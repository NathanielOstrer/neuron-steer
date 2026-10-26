# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx-lm==0.32.2"]
# ///
"""Side-by-side chat: normal Qwen and steered Qwen from one loaded model (MLX, 8-bit).

POST /api/chat {messages, dose, temperature, max_new} streams plain-text chunks.
dose 0 is normal Qwen; dose 1 is the tweet's steer. One worker thread owns the
model and runs requests one at a time.
"""
import argparse, json, queue, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PAGE = Path(__file__).with_name("chat.html")
jobs = queue.Queue()


def worker(spec, ready):
    import mlx_steer
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    tok, model = mlx_steer.load()
    steering = mlx_steer.Steering(tok, model, spec)
    ready.set()
    while True:
        req, out, cancel = jobs.get()
        try:
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

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            return self.send_error(404)
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/chat":
            return self.send_error(404)
        raw = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        req = {
            "messages": [{"role": m["role"], "content": m["content"]} for m in raw["messages"]],
            "dose": max(-3.0, min(3.0, float(raw.get("dose", 0)))),
            "temperature": max(0.0, min(2.0, float(raw.get("temperature", 0)))),
            "max_new": max(1, min(1024, int(raw.get("max_new", 300)))),
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8120)
    p.add_argument("--steer", default="14:8949:1,13:7956:-1")
    args = p.parse_args()
    ready = threading.Event()
    threading.Thread(target=worker, args=(args.steer, ready), daemon=True).start()
    ready.wait()
    print(f"http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
