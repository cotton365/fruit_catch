from __future__ import annotations

import json
import logging
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2


LOGGER = logging.getLogger("mango_vision.preview")


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Mango Vision</title>
  <style>
    body { margin: 0; background: #151719; color: #f2f2f2; font-family: system-ui, sans-serif; }
    main { max-width: 1280px; margin: auto; padding: 16px; }
    header { display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }
    h1 { margin: 0; font-size: 20px; font-weight: 600; }
    #status { color: #b8c0c7; }
    img { display: block; width: 100%; height: auto; background: #080909; }
    .note { margin-top: 10px; color: #9fa7ad; font-size: 14px; }
  </style>
</head>
<body>
  <main>
    <header><h1>芒果机器人视觉</h1><div id="status">正在连接...</div></header>
    <img src="/stream.mjpg" alt="实时相机画面">
    <div class="note">IDLE：等待 S/s；ALIGN：顶部对中；SIDE：绿色芒果位置识别；DONE：本轮完成。</div>
  </main>
  <script>
    async function refresh() {
      try {
        const data = await (await fetch('/status.json', {cache: 'no-store'})).json();
        document.getElementById('status').textContent =
          `状态 ${data.state} · 相机 ${data.camera} · 第 ${data.sequence} 棵树`;
      } catch (_) {
        document.getElementById('status').textContent = '连接中断，正在重试';
      }
    }
    refresh(); setInterval(refresh, 500);
  </script>
</body>
</html>
""".encode("utf-8")


class PreviewServer:
    def __init__(self, port: int, jpeg_quality: int = 85):
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.revision = 0
        self.status: dict[str, Any] = {"state": "STARTING", "camera": "none", "sequence": 0}
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        handler = partial(_PreviewHandler, preview=self)
        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, name="preview-http", daemon=True)
        self.thread.start()
        LOGGER.info("Preview available at http://0.0.0.0:%d", self.port)

    def publish(self, frame, *, state: str, camera: str, sequence: int) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        with self.condition:
            self.jpeg = encoded.tobytes()
            self.status = {"state": state, "camera": camera, "sequence": sequence}
            self.revision += 1
            self.condition.notify_all()

    def snapshot(self) -> tuple[bytes | None, int, dict[str, Any]]:
        with self.condition:
            return self.jpeg, self.revision, dict(self.status)

    def wait_for_frame(self, after_revision: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self.condition:
            if self.revision <= after_revision:
                self.condition.wait(timeout)
            return self.jpeg, self.revision

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None


class _PreviewHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, preview: PreviewServer, **kwargs):
        self.preview = preview
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_bytes(200, "text/html; charset=utf-8", PAGE)
        elif self.path == "/status.json":
            _, _, status = self.preview.snapshot()
            payload = json.dumps(status, ensure_ascii=False).encode("utf-8")
            self._send_bytes(200, "application/json; charset=utf-8", payload, no_cache=True)
        elif self.path == "/snapshot.jpg":
            jpeg, _, _ = self.preview.snapshot()
            if jpeg is None:
                self._send_bytes(503, "text/plain; charset=utf-8", b"No frame yet")
            else:
                self._send_bytes(200, "image/jpeg", jpeg, no_cache=True)
        elif self.path == "/stream.mjpg":
            self._stream()
        else:
            self._send_bytes(404, "text/plain; charset=utf-8", b"Not found")

    def _send_bytes(self, status: int, content_type: str, payload: bytes, no_cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if no_cache:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(payload)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        revision = -1
        try:
            while True:
                jpeg, next_revision = self.preview.wait_for_frame(revision)
                if jpeg is None or next_revision == revision:
                    continue
                revision = next_revision
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def log_message(self, format: str, *args) -> None:
        LOGGER.debug("Preview client %s: %s", self.client_address[0], format % args)
