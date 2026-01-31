#!/usr/bin/env python3
"""
Simple CSV sender to ESP32 for pre-prototype testing.
Usage example:
  python3 tests/exec/send_csv.py --device /dev/ttyUSB0 --baud 115200 --csv-file logs/csv/run_video_20260117.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:
    import serial as _serial
except ImportError:  # pragma: no cover
    _serial = None

if _serial is not None:
    SERIAL_EXCEPTIONS = tuple(
        exc
        for exc in (
            getattr(_serial, "SerialException", None),
            getattr(_serial, "SerialTimeoutException", None),
            OSError,
            RuntimeError,
        )
        if exc is not None
    )
else:
    SERIAL_EXCEPTIONS = (OSError, RuntimeError)
OPEN_EXCEPTIONS = SERIAL_EXCEPTIONS + (ValueError,)

ACK_PREFIX = "ACK"
DEFAULT_DEVICE = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200
DEFAULT_RATE_HZ = 10.0
DEFAULT_TIMEOUT_S = 1.0


@dataclass
class SendStats:
    sent: int = 0
    acks_received: int = 0
    errors: int = 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send normalized CSV lines to an ESP32 over serial.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Serial device path (default: /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baudrate (default: 115200)")
    parser.add_argument("--csv-file", dest="csv_file", help="Path to CSV file to send")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ, help="Send rate in Hz (default: 10)")
    parser.add_argument("--repeat", action="store_true", help="Keep polling the CSV file for new lines")
    parser.add_argument("--raw", action="store_true", help="Send raw CSV lines without normalization")
    return parser.parse_args(argv)


def _serial_module(explicit_module=None):
    if explicit_module is not None:
        return explicit_module
    return _serial


def _is_header_row(row: List[str]) -> bool:
    if not row:
        return False
    head = row[0].strip().lower()
    return head in {"frame_idx", "frame", "timestamp", "ts_wallclock_ms"}


def iter_csv_lines(csv_path: Path, raw: bool, repeat: bool) -> Iterable[str]:
    last_position = 0
    while True:
        if not csv_path.exists():
            print(f"⚠️ CSV file not found: {csv_path}")
            if not repeat:
                return
            time.sleep(1.0)
            continue
        current_size = csv_path.stat().st_size
        if repeat and current_size < last_position:
            last_position = 0
        emitted = False
        with csv_path.open(newline="", encoding="utf-8") as handle:
            if repeat and last_position:
                handle.seek(last_position)
            if raw:
                for line in handle:
                    line = line.strip()
                    if line:
                        emitted = True
                        yield line
            else:
                reader = csv.reader(handle)
                for row in reader:
                    if not row or all(not cell.strip() for cell in row):
                        continue
                    if _is_header_row(row):
                        continue
                    normalized = ",".join(cell.strip() for cell in row)
                    if normalized:
                        emitted = True
                        yield normalized
            last_position = handle.tell()
        if not repeat:
            return
        if not emitted:
            time.sleep(0.5)


def _read_response(serial_conn) -> str:
    try:
        raw = serial_conn.readline()
    except SERIAL_EXCEPTIONS:
        return ""
    if not raw:
        return ""
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
    return str(raw).strip()


def _write_payload(serial_conn, payload: str, warn: bool = True) -> bool:
    try:
        serial_conn.write(payload.encode("utf-8"))
        if hasattr(serial_conn, "flush"):
            serial_conn.flush()
        return True
    except SERIAL_EXCEPTIONS as exc:
        if warn:
            print(f"⚠️ Failed to write to serial: {exc}")
        return False


def send_lines(serial_conn, lines: Iterable[str], rate_hz: float, retry_on_nack: bool) -> SendStats:
    stats = SendStats()
    delay_s = 1.0 / rate_hz if rate_hz > 0 else 0.0
    for line in lines:
        payload = f"{line}\n"
        if not _write_payload(serial_conn, payload):
            stats.errors += 1
            continue
        stats.sent += 1
        response = _read_response(serial_conn)
        if response:
            print(response)
        if response.startswith(ACK_PREFIX):
            stats.acks_received += 1
        else:
            stats.errors += 1
            if retry_on_nack:
                if not _write_payload(serial_conn, payload, warn=False):
                    stats.errors += 1
        if delay_s > 0:
            time.sleep(delay_s)
    return stats


def send_csv_file(
    csv_path: Path,
    device: str,
    baud: int,
    rate_hz: float,
    repeat: bool,
    raw: bool,
    serial_module=None,
) -> SendStats:
    stats = SendStats()
    module = _serial_module(serial_module)
    if module is None:
        print("pyserial not installed. Install with: pip install pyserial")
        stats.errors += 1
        return stats
    try:
        serial_conn = module.Serial(device, baudrate=baud, timeout=DEFAULT_TIMEOUT_S, write_timeout=DEFAULT_TIMEOUT_S)
    except OPEN_EXCEPTIONS as exc:
        print(f"⚠️ Failed to open serial port {device}: {exc}")
        stats.errors += 1
        return stats
    try:
        line_iter = iter_csv_lines(csv_path, raw=raw, repeat=repeat)
        stats = send_lines(serial_conn, line_iter, rate_hz=rate_hz, retry_on_nack=repeat)
    finally:
        try:
            serial_conn.close()
        except Exception:
            pass
    return stats


def run(argv: Optional[Sequence[str]] = None, serial_module=None) -> int:
    args = _parse_args(argv)
    if not args.csv_file:
        print("⚠️ --csv-file is required (or provide a file to monitor)")
        return 2
    csv_path = Path(args.csv_file)
    stats = send_csv_file(
        csv_path=csv_path,
        device=args.device,
        baud=args.baud,
        rate_hz=args.rate_hz,
        repeat=args.repeat,
        raw=args.raw,
        serial_module=serial_module,
    )
    print(f"Stats: sent={stats.sent} acks_received={stats.acks_received} errors={stats.errors}")
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
