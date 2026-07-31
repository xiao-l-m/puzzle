#!/usr/bin/env python3
"""Independent three-question service built on the existing vision modules.

The dedicated question-2 and question-3 entry points remain untouched.  This
entry point owns the same single camera, UART and HTTP port, but routes KEY 1,
KEY 2 and KEY 3 to analysis modes 1, 2 and 3 before the existing controller
starts its fresh-frame wait.
"""

from __future__ import annotations

import json
import signal
import threading
from http.server import ThreadingHTTPServer

from puzzle_vision import (
    CONTROL_HTML,
    PuzzleVisionApp,
    make_handler,
    parse_args,
)


INTEGRATED_SERVICE_MODE = 0


def parse_hardware_key(line: str) -> tuple[int, int] | None:
    """Return ``(request_id, mode)`` for a valid MCU KEY message."""
    fields = line.split()
    if len(fields) != 3 or fields[0].upper() != "KEY":
        return None
    request_id = int(fields[1], 10)
    mode = int(fields[2], 10)
    if not 0 <= request_id <= 2_147_483_647:
        raise ValueError("request id out of range")
    if mode not in (1, 2, 3):
        raise ValueError("hardware key mode must be 1, 2 or 3")
    return request_id, mode


def build_integrated_html() -> bytes:
    """Expose all three controls while retaining the existing dashboard."""
    page = CONTROL_HTML.decode("utf-8")
    page = page.replace(
        "<body>",
        "<body><div style='padding:8px 12px;background:#285b91;"
        "border-radius:7px;margin-bottom:9px'>"
        "综合服务：按键1→问题一，按键2→问题二，按键3→问题三；"
        "一个相机、一个串口、一个运动控制器。</div>",
        1,
    )
    replacements = {
        "let serviceMode=Number((s.config||{}).service_mode||2);":
            "let serviceMode=0;",
        "if(selectedPreviewMode===0)selectPreview(serviceMode,true);":
            "if(selectedPreviewMode===0)selectPreview(2,true);",
        "if(serviceMode===3){mode1.disabled=true;mode2.disabled=true;}": "",
        "mode3.disabled=serviceMode!==3||!ser.connected||robotBusy;":
            "mode3.disabled=!ser.connected||robotBusy;",
        "view2Button.disabled=serviceMode!==2;":
            "view2Button.disabled=false;",
        "view3Button.disabled=serviceMode!==3;":
            "view3Button.disabled=false;",
        "mode3.style.display=serviceMode===3?'inline-block':'none';":
            "mode3.style.display='inline-block';",
        "view3Button.style.display=serviceMode===3?'inline-block':'none';":
            "view3Button.style.display='inline-block';",
        "view2Button.style.display=serviceMode===2?'inline-block':'none';":
            "view2Button.style.display='inline-block';",
        "probePanel.style.display=serviceMode===3?'block':'none';":
            "probePanel.style.display='none';",
    }
    for old, new in replacements.items():
        if old not in page:
            raise RuntimeError(f"dashboard integration marker missing: {old}")
        page = page.replace(old, new, 1)
    # Both recognition summaries are useful in the combined service.  Only the
    # currently selected pipeline consumes CPU; switching preview changes it.
    page = page.replace("if(serviceMode===2){", "{", 1)
    page = page.replace("if(serviceMode===3){", "{", 1)
    return page.encode("utf-8")


INTEGRATED_HTML = build_integrated_html()


class IntegratedPuzzleVisionApp(PuzzleVisionApp):
    """Route all three modes without changing the dedicated service class."""

    def __init__(self, args) -> None:
        # Parent construction uses the existing mode-2 defaults.  Immediately
        # afterwards, mode 0 denotes this wrapper's unrestricted dispatcher.
        args.service_mode = 2
        super().__init__(args)
        with self.lock:
            self.service_mode = INTEGRATED_SERVICE_MODE
            self.analysis_mode = 2
        self.controller.link.on_line = self._on_integrated_serial_line

    def _arm_hardware_mode(self, mode: int) -> None:
        self.set_analysis_mode(mode)
        if mode == 3 and self.manual_corners is None:
            self.lock_current_a4()
            self.a4_lock_method = "integrated_key3_one_click_stable_a4"

    def _on_integrated_serial_line(self, line: str) -> None:
        """Switch vision first, then preserve the controller protocol."""
        try:
            key = parse_hardware_key(line)
            if key is not None:
                _request_id, mode = key
                self._arm_hardware_mode(mode)
        except Exception as exc:
            self.controller._record("INTEGRATED_KEY_REJECTED " + str(exc))
            return
        self.controller._on_serial_line(line)

    def status(self) -> dict:
        status = super().status()
        status["integrated_service"] = {
            "enabled": True,
            "active_analysis_mode": self.analysis_mode,
            "key_mapping": {"1": 1, "2": 2, "3": 3},
            "single_camera_owner": True,
        }
        return status


def make_integrated_handler(app: IntegratedPuzzleVisionApp):
    base_handler = make_handler(app)

    class IntegratedHandler(base_handler):
        def do_GET(self) -> None:
            if self.path == "/":
                self.reply(INTEGRATED_HTML, "text/html; charset=utf-8")
                return
            super().do_GET()

    return IntegratedHandler


def main() -> int:
    args = parse_args()
    if args.self_test or args.image is not None:
        raise SystemExit(
            "integrated service is a live-camera entry point; use puzzle_vision.py "
            "for image/self tests"
        )
    app = IntegratedPuzzleVisionApp(args)
    server = ThreadingHTTPServer(
        (args.host, args.web_port), make_integrated_handler(app)
    )

    def shutdown(_signum=None, _frame=None) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(
        json.dumps(
            {
                "service": "integrated_questions_1_2_3",
                "url": f"http://<raspberry-pi-ip>:{args.web_port}",
                "key_mapping": {"1": 1, "2": 2, "3": 3},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
