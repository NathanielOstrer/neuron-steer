# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx-lm==0.32.2"]
# ///
"""Side-by-side chat: normal Qwen and steered Qwen from one loaded model (MLX, 8-bit).

POST /api/chat {messages, dose, temperature, max_new} streams plain-text chunks.
dose 0 is normal Qwen; dose 1 is the tweet's steer. One worker thread owns the
model and runs requests one at a time. The model loads on the first request and
unloads after --idle minutes, so it holds no RAM while nobody is chatting.

With --secrets FILE (lines KEY=VALUE with PASSWORD and SECRET), every route needs
a login cookie. Without it, the server is open (local use).
"""
import argparse, gc, hashlib, hmac, json, queue, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs

HERE = Path(__file__).parent
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


def worker(spec, idle_s):
    import mlx.core as mx
    import mlx_steer
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    tok = model = steering = None
    while True:
        try:
            req, out, cancel = jobs.get(timeout=idle_s if model else None)
        except queue.Empty:
            tok = model = steering = None
            gc.collect()
            mx.clear_cache()
            print("model unloaded (idle)", flush=True)
            continue
        try:
            if model is None:
                t = time.time()
                tok, model = mlx_steer.load()
                steering = mlx_steer.Steering(tok, model, spec)
                print(f"model loaded in {time.time() - t:.1f}s", flush=True)
            steering.scale = req["dose"]
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

    def do_GET(self):
        if self.path.split("?")[0] not in ("/", "/index.html"):
            return self.send_error(404)
        self.send_page(PAGE if self.authed() else LOGIN)

    def do_POST(self):
        if self.path == "/login":
            return self.login()
        if self.path != "/api/chat":
            return self.send_error(404)
        if not self.authed():
            return self.send_error(401)
        raw = json.loads(self.rfile.read(min(int(self.headers["Content-Length"]), 1_000_000)))
        req = {
            "messages": [{"role": m["role"], "content": str(m["content"])[:8000]} for m in raw["messages"][-40:]],
            "dose": max(-3.0, min(3.0, float(raw.get("dose", 0)))),
            "temperature": max(0.0, min(2.0, float(raw.get("temperature", 0)))),
            "max_new": max(1, min(1024, int(raw.get("max_new", 150)))),
        }
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
        given = form.get("password", [""])[0].strip().lower()
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
    p.add_argument("--steer", default="14:8949:1,13:7956:-1")
    p.add_argument("--idle", type=float, default=20, help="minutes before the model unloads")
    p.add_argument("--secrets", help="KEY=VALUE file with PASSWORD and SECRET; turns on the login")
    args = p.parse_args()
    if args.secrets:
        for line in Path(args.secrets).expanduser().read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("PASSWORD", "SECRET"):
                auth[k.strip()] = v.strip()
        assert auth.get("PASSWORD") and auth.get("SECRET"), "secrets file needs PASSWORD and SECRET"
    threading.Thread(target=worker, args=(args.steer, args.idle * 60), daemon=True).start()
    print(f"http://{args.host}:{args.port} login={'on' if auth else 'off'}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
