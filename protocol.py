from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import serial


POSITION_ORDER = ("L1", "L2", "R1", "R2")


class StartSignalParser:
    """Accept the MCU's single-byte S/s trigger, with or without a newline."""

    @staticmethod
    def has_start(data: bytes) -> bool:
        return any(byte in (ord("S"), ord("s")) for byte in data)


def alignment_message(sequence: int, dx_px: int, dy_px: int) -> str:
    return f"A,{sequence},{dx_px:+d},{dy_px:+d}"


def centered_message(sequence: int) -> str:
    return f"C,{sequence}"


def orientation_message(sequence: int, angle_deg: float) -> str:
    return f"T,{sequence},{angle_deg:+.1f}"


def result_message(sequence: int, positions: Iterable[str]) -> str:
    selected = set(positions)
    ordered = [position for position in POSITION_ORDER if position in selected]
    payload = ",".join(ordered) if ordered else "NONE"
    return f"G,{sequence},{payload}"


def begin_message(sequence: int) -> str:
    return f"B,{sequence}"


def error_message(sequence: int, code: str) -> str:
    safe_code = code.replace(",", "_").replace("\n", "_").replace("\r", "_")
    return f"E,{sequence},{safe_code}"


@dataclass
class SerialLink:
    port: str
    baudrate: int
    timeout_s: float = 0.0

    def __post_init__(self) -> None:
        self.connection = serial.serial_for_url(
            self.port,
            baudrate=self.baudrate,
            timeout=self.timeout_s,
            write_timeout=0.5,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

    def poll_start(self) -> bool:
        waiting = self.connection.in_waiting
        if waiting <= 0:
            return False
        return StartSignalParser.has_start(self.connection.read(waiting))

    def send(self, line: str) -> None:
        self.connection.write((line + "\n").encode("ascii"))
        self.connection.flush()

    def close(self) -> None:
        self.connection.close()
