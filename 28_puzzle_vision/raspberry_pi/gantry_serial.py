#!/usr/bin/env python3
"""Raspberry-Pi to MSPM0 task link for the E-question gantry.

The camera thread keeps producing live measurements.  A button event freezes
one stable four-piece plan and sends that immutable plan to the MCU.  Motion is
then entirely owned by the MCU state machine, so removing a piece from the
camera view cannot change an in-flight job.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable

from puzzle_motion import StageCalibration, build_task_plan, estimate_plan_seconds


MAX_SAFE_PLAN_SECONDS = 105.0
UART_CONSOLE_CONFLICT = "UART_CONSOLE_CONFLICT"

try:
    import serial  # type: ignore
except ImportError:  # Keep the vision web page available without pyserial.
    serial = None


class SerialLinkError(RuntimeError):
    pass


def _device_names(device: str) -> set[str]:
    """Return the configured and resolved Linux tty basenames."""

    names = {os.path.basename(os.path.normpath(device))}
    resolved = os.path.realpath(device)
    if resolved:
        names.add(os.path.basename(os.path.normpath(resolved)))
    names.discard("")
    return names


def detect_uart_console_conflict(
    device: str,
    cmdline_path: str = "/proc/cmdline",
) -> bool:
    """Return True when Linux reserves ``device`` as a kernel console.

    Raspberry Pi OS may expose the same UART through both ``serial0`` and a
    concrete tty such as ``ttyAMA10``.  Resolve both sides so either spelling
    is detected.  Non-Linux systems and systems without ``/proc/cmdline`` keep
    the previous serial-opening behaviour.
    """

    if not sys.platform.startswith("linux"):
        return False
    try:
        cmdline = Path(cmdline_path).read_text(
            encoding="ascii", errors="replace"
        )
    except OSError:
        return False

    configured_names = _device_names(device)
    for argument in cmdline.split():
        if not argument.startswith("console="):
            continue
        console_name = argument[len("console=") :].split(",", 1)[0]
        if not console_name:
            continue
        console_path = (
            console_name
            if console_name.startswith("/")
            else os.path.join("/dev", console_name)
        )
        if configured_names.intersection(_device_names(console_path)):
            return True
    return False


class GantrySerialLink:
    """Reconnectable newline protocol on the Raspberry Pi primary UART."""

    def __init__(
        self,
        device: str,
        baudrate: int,
        on_line: Callable[[str], None],
    ) -> None:
        self.device = device
        self.baudrate = int(baudrate)
        self.on_line = on_line
        self.running = True
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.port_open_event = threading.Event()
        self.connected_event = threading.Event()
        self.port: Any | None = None
        self.last_error: str | None = None
        self.last_rx: str | None = None
        self.last_tx: str | None = None
        self.last_rx_monotonic: float | None = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _set_port(self, port: Any | None, error: str | None = None) -> None:
        with self.lock:
            self.port = port
            self.last_error = error
            if port is None:
                self.port_open_event.clear()
                self.connected_event.clear()
            else:
                self.last_rx = None
                self.last_rx_monotonic = None
                self.port_open_event.set()

    def _disconnect(self, error: str) -> None:
        with self.lock:
            port = self.port
            self.port = None
            self.last_error = error
            self.port_open_event.clear()
            self.connected_event.clear()
        if port is not None:
            try:
                port.close()
            except Exception:
                pass

    def _loop(self) -> None:
        if detect_uart_console_conflict(self.device):
            # Do not open a kernel-console UART: serial-getty may already own
            # it, and two readers would make the MCU protocol nondeterministic.
            self._set_port(None, UART_CONSOLE_CONFLICT)
            return
        if serial is None:
            self._set_port(None, "pyserial is not installed")
            return
        last_ping = 0.0
        while self.running:
            with self.lock:
                port = self.port
            if port is None:
                try:
                    opened = serial.Serial(
                        self.device,
                        self.baudrate,
                        timeout=0.25,
                        write_timeout=0.5,
                    )
                    opened.reset_input_buffer()
                    self._set_port(opened)
                    port = opened
                    last_ping = 0.0
                except Exception as exc:
                    self._set_port(None, str(exc))
                    time.sleep(1.0)
                    continue
            try:
                raw = port.readline()
                if not raw:
                    now = time.monotonic()
                    with self.lock:
                        if (
                            self.last_rx_monotonic is not None
                            and now - self.last_rx_monotonic > 2.5
                        ):
                            self.connected_event.clear()
                    if now - last_ping >= 0.8:
                        with self.write_lock:
                            port.write(b"PING\n")
                            port.flush()
                            with self.lock:
                                self.last_tx = "PING"
                        last_ping = now
                    continue
                line = raw.decode("ascii", errors="replace").strip()
                if not line:
                    continue
                with self.lock:
                    self.last_rx = line
                    self.last_rx_monotonic = time.monotonic()
                    self.connected_event.set()
                self.on_line(line)
            except Exception as exc:
                self._disconnect(str(exc))
                time.sleep(0.5)

    def wait_connected(self, timeout: float) -> bool:
        return self.connected_event.wait(timeout=max(0.0, timeout))

    def send(self, line: str) -> None:
        clean = " ".join(str(line).strip().split())
        if not clean or "\n" in clean or "\r" in clean:
            raise ValueError("serial command must be one non-empty line")
        payload = (clean + "\n").encode("ascii")
        with self.write_lock:
            with self.lock:
                port = self.port
            if port is None:
                raise SerialLinkError(self.last_error or "serial disconnected")
            try:
                port.write(payload)
                port.flush()
                with self.lock:
                    self.last_tx = clean
            except Exception as exc:
                self._disconnect(str(exc))
                raise SerialLinkError(str(exc)) from exc

    def status(self) -> dict[str, Any]:
        with self.lock:
            rx_age = (
                None
                if self.last_rx_monotonic is None
                else round(time.monotonic() - self.last_rx_monotonic, 2)
            )
            return {
                "device": self.device,
                "baudrate": self.baudrate,
                "port_open": self.port is not None,
                "connected": (
                    self.port is not None
                    and rx_age is not None
                    and rx_age <= 2.5
                ),
                "error": self.last_error,
                "last_rx": self.last_rx,
                "last_tx": self.last_tx,
                "last_rx_age_seconds": rx_age,
            }

    def close(self) -> None:
        self.running = False
        self._disconnect("closed")
        self.thread.join(timeout=1.0)


class GantryTaskController:
    """Queue button requests, freeze a stable plan, and send it once."""

    def __init__(
        self,
        status_provider: Callable[[], dict[str, Any]],
        serial_device: str,
        serial_baud: int,
        calibration: StageCalibration = StageCalibration(),
        vision_wait_seconds: float = 15.0,
    ) -> None:
        self.status_provider = status_provider
        self.calibration = calibration
        self.vision_wait_seconds = float(vision_wait_seconds)
        self.lock = threading.Lock()
        self.requests: queue.Queue[tuple[int, int, str]] = queue.Queue(maxsize=4)
        self.running = True
        self.state = "IDLE"
        self.message = "等待按键1或按键2"
        self.active_request_id: int | None = None
        self.active_mode: int | None = None
        self.frozen_plan: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.history: deque[str] = deque(maxlen=40)
        self.seen_requests: deque[int] = deque(maxlen=32)
        self.manual_sequence = 1_000_000
        self.link = GantrySerialLink(
            serial_device, serial_baud, self._on_serial_line
        )
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _record(self, line: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.history.append(f"{timestamp} {line}")

    def _set_state(
        self,
        state: str,
        message: str,
        error: str | None = None,
    ) -> None:
        with self.lock:
            self.state = state
            self.message = message
            self.last_error = error

    @staticmethod
    def _parse_int(text: str) -> int:
        value = int(text, 10)
        if value < 0 or value > 2_147_483_647:
            raise ValueError("sequence out of range")
        return value

    def _on_serial_line(self, line: str) -> None:
        self._record("MCU> " + line)
        fields = line.split()
        if not fields:
            return
        command = fields[0].upper()
        if command == "PONG" and len(fields) >= 2:
            if fields[1].upper() == "READY":
                with self.lock:
                    if self.state in {"STOPPED", "ERROR", "REJECTED"}:
                        self.state = "IDLE"
                        self.message = "天猛星已复位并声明初始位有效"
                        self.last_error = None
            return
        if command == "KEY" and len(fields) == 3:
            try:
                request_id = self._parse_int(fields[1])
                mode = int(fields[2])
                self.request_task(mode, request_id, "MCU_KEY")
            except Exception as exc:
                self._record("忽略非法KEY: " + str(exc))
            return
        if command == "STATE" and len(fields) >= 3:
            try:
                request_id = self._parse_int(fields[1])
            except ValueError:
                return
            with self.lock:
                if request_id == self.active_request_id:
                    self.state = "MCU_" + fields[2]
                    self.message = " ".join(fields[2:])
            return
        if command == "DONE" and len(fields) >= 2:
            try:
                request_id = self._parse_int(fields[1])
            except ValueError:
                return
            with self.lock:
                if request_id == self.active_request_id:
                    self.state = "DONE"
                    self.message = "任务完成，电磁铁已关闭"
                    self.last_error = None
            return
        if command == "ERR" and len(fields) >= 3:
            try:
                request_id = self._parse_int(fields[1])
            except ValueError:
                return
            with self.lock:
                if request_id == self.active_request_id:
                    self.state = "ERROR"
                    self.message = " ".join(fields[2:])
                    self.last_error = self.message

    def request_task(
        self,
        mode: int,
        request_id: int | None = None,
        source: str = "WEB",
    ) -> int:
        if mode not in (1, 2):
            raise ValueError("mode must be 1 or 2")
        with self.lock:
            if request_id is None:
                self.manual_sequence += 1
                request_id = self.manual_sequence
            if request_id in self.seen_requests:
                self.history.append(
                    f"{time.strftime('%H:%M:%S')} duplicate KEY {request_id} ignored"
                )
                return request_id
            if self.state not in {"IDLE", "DONE", "ERROR", "REJECTED"}:
                raise RuntimeError("another task is active")
            self.seen_requests.append(request_id)
            self.state = "QUEUED"
            self.message = f"模式{mode}已排队，等待稳定视觉"
            self.active_request_id = request_id
            self.active_mode = mode
            self.frozen_plan = None
            self.last_error = None
        try:
            self.requests.put_nowait((request_id, mode, source))
        except queue.Full as exc:
            self._set_state("ERROR", "任务队列已满", "QUEUE_FULL")
            raise RuntimeError("task queue is full") from exc
        self._record(f"{source} 请求模式{mode} seq={request_id}")
        return request_id

    def _reject(self, request_id: int, error: str) -> None:
        self._set_state("REJECTED", error, error)
        self._record(f"计划拒绝 seq={request_id}: {error}")
        try:
            self.link.send(f"REJECT {request_id} {error}")
        except Exception:
            pass

    def _wait_for_plan(self, request_id: int, mode: int) -> dict[str, Any] | None:
        deadline = time.monotonic() + self.vision_wait_seconds
        last_error = "VISION_NOT_READY"
        while self.running and time.monotonic() < deadline:
            status = self.status_provider()
            plan = build_task_plan(mode, status, self.calibration)
            last_error = str(plan.get("error") or "VISION_NOT_READY")
            if plan.get("ready", False):
                frozen = deepcopy(plan)
                frozen["request_id"] = request_id
                frozen["estimated_seconds"] = estimate_plan_seconds(
                    frozen, self.calibration
                )
                if frozen["estimated_seconds"] > MAX_SAFE_PLAN_SECONDS:
                    self._reject(request_id, "PLAN_EXCEEDS_105_SECONDS")
                    return None
                frozen["frozen_at_unix"] = round(time.time(), 3)
                return frozen
            self._set_state(
                "WAITING_VISION",
                f"等待稳定4片与可达坐标：{last_error}",
            )
            time.sleep(0.15)
        self._reject(request_id, last_error)
        return None

    @staticmethod
    def _item_line(request_id: int, index: int, move: dict[str, Any]) -> str:
        pick = move["pick_a4_mm"]
        place = move["place_a4_mm"]
        angle = float(move.get("motor5_rotate_deg", 0.0))
        return (
            f"ITEM {request_id} {index} "
            f"{int(round(float(pick[0]) * 10.0))} "
            f"{int(round(float(pick[1]) * 10.0))} "
            f"{int(round(float(place[0]) * 10.0))} "
            f"{int(round(float(place[1]) * 10.0))} "
            f"{int(round(angle * 1000.0))}"
        )

    def _send_plan(self, request_id: int, mode: int, plan: dict[str, Any]) -> None:
        moves = list(plan.get("moves") or [])
        if len(moves) != 4:
            raise ValueError("plan must contain exactly four moves")
        if not self.link.wait_connected(3.0):
            raise SerialLinkError(self.link.status().get("error") or "serial disconnected")
        lines = [f"PLAN {request_id} {mode} {len(moves)}"]
        lines.extend(
            self._item_line(request_id, index, move)
            for index, move in enumerate(moves)
        )
        lines.append(f"COMMIT {request_id}")
        for line in lines:
            self.link.send(line)
            self._record("PI> " + line)
            # Keep each complete line easy for the MCU main loop to consume.
            time.sleep(0.035)

    def _worker(self) -> None:
        while self.running:
            try:
                request_id, mode, _source = self.requests.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                plan = self._wait_for_plan(request_id, mode)
                if plan is None:
                    continue
                with self.lock:
                    self.frozen_plan = plan
                    self.state = "SENDING_PLAN"
                    self.message = (
                        f"已冻结模式{mode}的4片坐标，预计"
                        f"{plan['estimated_seconds']}秒"
                    )
                self._send_plan(request_id, mode, plan)
                with self.lock:
                    # A fast MCU may emit STATE immediately after COMMIT.
                    # Do not overwrite that newer execution state.
                    if self.state == "SENDING_PLAN":
                        self.state = "WAITING_MCU"
                        self.message = "计划已完整下发，等待天猛星执行"
            except Exception as exc:
                self._reject(request_id, "LINK_OR_PLAN_" + str(exc).replace(" ", "_"))
            finally:
                self.requests.task_done()

    def emergency_stop(self) -> None:
        try:
            self.link.send("STOP")
            self._record("PI> STOP")
        finally:
            self._set_state("STOPPED", "已发送急停；必须回初始位并复位", "STOPPED")

    def status(self) -> dict[str, Any]:
        with self.lock:
            data = {
                "state": self.state,
                "message": self.message,
                "error": self.last_error,
                "active_request_id": self.active_request_id,
                "active_mode": self.active_mode,
                "frozen_plan": deepcopy(self.frozen_plan),
                "history": list(self.history),
            }
        data["serial"] = self.link.status()
        return data

    def close(self) -> None:
        self.running = False
        self.worker.join(timeout=1.0)
        self.link.close()


__all__ = ["GantrySerialLink", "GantryTaskController", "SerialLinkError"]
