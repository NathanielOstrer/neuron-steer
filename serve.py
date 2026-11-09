# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx-lm==0.32.2"]
# ///
"""Side-by-side chat: normal Qwen and steered Qwen from one loaded model (MLX, 8-bit).

POST /api/chat {messages, preset, dose, temperature, max_new} streams plain-text chunks.
GET /api/presets lists the steers: every catalog entry in personas.json that held up
(mix.chat_presets). dose 0 is normal Qwen; dose 1 is the preset as published. One
worker thread owns the model, keeps one preset applied, and runs requests one at a time. The model loads on the first request and
unloads after --idle minutes, so it holds no RAM while nobody is chatting.

With --secrets FILE (lines KEY=VALUE with PASSWORD and SECRET), every route needs
a login cookie. Without it, the server is open (local use).
"""
import argparse, gc, hashlib, hmac, json, queue, re, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs

HERE = Path(__file__).parent
ATLAS_PAGE, ATLAS_DATA = HERE / "atlas.html", HERE / "atlas" / "data.json"
_atlas_gz = {}  # mtime -> gzipped atlas data, so each request does not compress several MB
DEFAULT_PRESET = "evil-recipe"
presets = {}  # id -> {id, label, group, spec, note}, filled at startup
PAGE, LOGIN = HERE / "chat.html", HERE / "login.html"
COOKIE = "eq_auth"
jobs = queue.Queue()
auth = {}  # PASSWORD, SECRET when a secrets file is given
fails = {}  # client ip -> (count, last failure time)
recent_fails = []  # times of failed logins from any ip, for the global limit
GLOBAL_FAILS_PER_MIN = 20
login_gate = threading.BoundedSemaphore(4)  # concurrent password checks


def token():
    return hmac.new(auth["SECRET"].encode(), auth["PASSWORD"].encode(), hashlib.sha256).hexdigest()


CUSTOM_MAX_ITEMS, CUSTOM_MAX_DOSE = 6, 2.0


def custom_spec(spec):
    """A neuron-only spec from the atlas ("L:N:dose,..."), normalized, or None if it is not allowed.
    Features are not allowed here: the sandboxed server only has the preset feature rows."""
    try:
        items = [s.strip() for s in str(spec).split(",") if s.strip()]
        if not 0 < len(items) <= CUSTOM_MAX_ITEMS:
            return None
        out = []
        for it in items:
            L, n, d = it.split(":")
            L, n, d = int(L), int(n), float(d)
            if not (0 <= L < 32 and 0 <= n < 9216 and 0 < abs(d) <= CUSTOM_MAX_DOSE):
                return None
            out.append(f"{L}:{n}:{d:g}")
        return ",".join(out)
    except (ValueError, TypeError):
        return None


def parse_request(raw):
    """Clamp and validate a /api/chat body. An unknown preset falls back to the default.
    A valid `spec` (from the atlas) wins over `preset` and is applied as preset "custom:<spec>"."""
    preset = str(raw.get("preset") or DEFAULT_PRESET)
    spec = custom_spec(raw["spec"]) if raw.get("spec") else None
    return {
        "messages": [{"role": m["role"], "content": str(m["content"])[:8000]} for m in raw["messages"][-40:]],
        "preset": f"custom:{spec}" if spec else (preset if preset in presets else DEFAULT_PRESET),
        "dose": max(-3.0, min(3.0, float(raw.get("dose", 0)))),
        "temperature": max(0.0, min(2.0, float(raw.get("temperature", 0)))),
        "max_new": max(1, min(1024, int(raw.get("max_new", 150)))),
    }


def worker(idle_s):
    import mlx.core as mx
    import mix, mlx_steer
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    tok = model = tm = meds = steer = None
    current = None  # preset id that `steer` applies
    rows = mx.load(str(mix.ROWS)) if mix.ROWS.exists() else {}
    while True:
        try:
            req, out, cancel = jobs.get(timeout=idle_s if model else None)
        except queue.Empty:
            tok = model = tm = meds = steer = current = None
            gc.collect()
            mx.clear_cache()
            print("model unloaded (idle)", flush=True)
            continue
        try:
            if model is None:
                t = time.time()
                tok, model = mlx_steer.load()
                tm = mlx_steer.text_model(model)
                meds = mlx_steer.Steering._medians(tok, model, tm)
                print(f"model loaded in {time.time() - t:.1f}s", flush=True)
            # Dose 0 is the plain model under any preset, so only a steered request switches.
            if req["dose"] and req["preset"] != current:
                if steer:
                    steer.remove()
                spec = req["preset"][7:] if req["preset"].startswith("custom:") else presets[req["preset"]]["spec"]
                steer, current = mix.Mix(tok, model, tm, meds, spec, rows=rows), req["preset"]
                print(f"preset {current}", flush=True)
            if steer:
                steer.scale = req["dose"]
            prompt = mlx_steer.prompt_for(tok, req["messages"])
            sampler = make_sampler(temp=req["temperature"], top_p=0.95 if req["temperature"] > 0 else 0.0)
            for r in stream_generate(model, tok, prompt, max_tokens=req["max_new"], sampler=sampler):
                if cancel.is_set():
                    break
                if r.text:
                    out.put(r.text)
        except Exception as e:  # report to the client instead of killing the worker
            out.put(f"\n[server error: {e}]")
        finally:
            out.put(None)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def client_ip(self):
        return self.headers.get("CF-Connecting-IP") or self.client_address[0]

    def authed(self):
        if not auth:
            return True
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE and hmac.compare_digest(v, token()):
                return True
        return False

    def send_page(self, path, status=200, error=""):
        body = path.read_text().replace("<!--ERROR-->", error).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_atlas(self):
        import gzip
        if not ATLAS_DATA.exists():
            return self.send_error(404)
        mtime = ATLAS_DATA.stat().st_mtime
        if mtime not in _atlas_gz:
            _atlas_gz.clear()
            _atlas_gz[mtime] = gzip.compress(ATLAS_DATA.read_bytes(), 6)
        gz = "gzip" in (self.headers.get("Accept-Encoding") or "")
        body = _atlas_gz[mtime] if gz else ATLAS_DATA.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=300")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/presets":
            if not self.authed():
                return self.send_error(401)
            body = json.dumps({"default": DEFAULT_PRESET, "presets": list(presets.values())}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return self.wfile.write(body)
        if path in ("/atlas", "/atlas.html"):
            return self.send_page(ATLAS_PAGE if self.authed() else LOGIN)
        if path == "/atlas/data.json":
            if not self.authed():
                return self.send_error(401)
            return self.send_atlas()
        if path not in ("/", "/index.html"):
            return self.send_error(404)
        self.send_page(PAGE if self.authed() else LOGIN)

    def do_POST(self):
        if self.path == "/login":
            return self.login()
        if self.path != "/api/chat":
            return self.send_error(404)
        if not self.authed():
            return self.send_error(401)
        req = parse_request(json.loads(self.rfile.read(min(int(self.headers["Content-Length"]), 1_000_000))))
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        out, cancel = queue.Queue(), threading.Event()
        jobs.put((req, out, cancel))
        while (chunk := out.get()) is not None:
            if cancel.is_set():
                continue  # drain until the worker finishes
            try:
                self.wfile.write(chunk.encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                cancel.set()

    def login(self):
        if not auth:
            return self.redirect("/")
        if not login_gate.acquire(blocking=False):
            return self.send_page(LOGIN, 429, "Too many people are logging in. Wait one minute and try again.")
        try:
            return self._login()
        finally:
            login_gate.release()

    def _login(self):
        ip, now = self.client_ip(), time.time()
        recent_fails[:] = [t for t in recent_fails if now - t < 60]
        for k in [k for k, (_, last) in fails.items() if now - last > 600]:
            del fails[k]
        n, last = fails.get(ip, (0, 0))
        if (n >= 5 and now - last < 60) or len(recent_fails) >= GLOBAL_FAILS_PER_MIN:
            return self.send_page(LOGIN, 429, "Too many wrong passwords. Wait one minute and try again.")
        form = parse_qs(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 4096)).decode())
        # Accept spaces, underscores and phone-keyboard dashes between the words.
        given = re.sub(r"[\s_\u2010-\u2015]+", "-", form.get("password", [""])[0].strip().lower())
        if not hmac.compare_digest(given.encode(), auth["PASSWORD"].encode()):
            fails[ip] = (n + 1, now)
            recent_fails.append(now)
            time.sleep(1)
            return self.send_page(LOGIN, 401, "The password is not correct. Try again.")
        fails.pop(ip, None)
        self.redirect("/", f"{COOKIE}={token()}; Max-Age=2592000; Path=/; HttpOnly; Secure; SameSite=Lax")

    def redirect(self, to, cookie=None):
        self.send_response(303)
        self.send_header("Location", to)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8120)
    p.add_argument("--steer", help="add this spec as preset 'cli' and make it the default")
    p.add_argument("--idle", type=float, default=20, help="minutes before the model unloads")
    p.add_argument("--secrets", help="KEY=VALUE file with PASSWORD and SECRET; turns on the login")
    args = p.parse_args()
    if args.secrets:
        for line in Path(args.secrets).expanduser().read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("PASSWORD", "SECRET"):
                auth[k.strip()] = v.strip()
        assert auth.get("PASSWORD") and auth.get("SECRET"), "secrets file needs PASSWORD and SECRET"
    import mix
    presets.update({p["id"]: p for p in mix.chat_presets()})
    if args.steer:
        mix.parse(args.steer)  # fail at startup, not on the first message
        presets["cli"] = {"id": "cli", "label": "Command line", "group": "Originals", "spec": args.steer, "note": ""}
        DEFAULT_PRESET = "cli"
    threading.Thread(target=worker, args=(args.idle * 60,), daemon=True).start()
    print(f"{len(presets)} presets, default {DEFAULT_PRESET}", flush=True)
    print(f"http://{args.host}:{args.port} login={'on' if auth else 'off'}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
