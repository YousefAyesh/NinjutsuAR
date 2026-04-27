"""
Tiny HTTP receiver for HandSignManager debug captures.

Run on the laptop (same Wi-Fi as the phone):
    python tools/capture_server.py --host 0.0.0.0 --port 8000 --out captures

Then in Unity, on HandSignManager:
    captureEnabled  = true
    uploadToServer  = true
    uploadUrl       = http://<laptop-LAN-ip>:8000/upload

Find your LAN IP with `ipconfig` (Windows) and make sure the firewall lets
inbound connections through on the chosen port (allow Python on Private networks).
"""

from __future__ import annotations

import argparse
import cgi
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(out_dir: str):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/upload":
                self.send_error(404, "Use /upload")
                return

            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                self.send_error(400, "Expected multipart/form-data")
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype},
            )

            file_item = form["file"] if "file" in form else None
            if file_item is None or not getattr(file_item, "filename", None):
                self.send_error(400, "Missing 'file' field")
                return

            meta = form.getvalue("meta", "") or ""
            label = meta.split("|", 1)[0].strip() or "unknown"
            label_dir = os.path.join(out_dir, _safe(label))
            os.makedirs(label_dir, exist_ok=True)

            fname = file_item.filename or f"{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            path = os.path.join(label_dir, _safe(fname))
            with open(path, "wb") as f:
                f.write(file_item.file.read())

            print(f"[recv] {path}  meta={meta!r}")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, fmt, *args):  # silence default access log noise
            return

    return Handler


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="captures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(os.path.abspath(args.out)))
    print(f"Listening on http://{args.host}:{args.port}/upload  ->  {os.path.abspath(args.out)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
