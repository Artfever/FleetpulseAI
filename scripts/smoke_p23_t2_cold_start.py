"""Measure a fresh local Streamlit process until its health endpoint is ready."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def measure_startup(*, port: int | None = None, timeout_seconds: float = 60.0) -> dict[str, object]:
    """Start a new headless app process, report time to health, and always stop it."""
    selected_port = _available_port() if port is None else port
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/Home.py",
        "--server.port",
        str(selected_port),
        "--server.headless",
        "true",
    ]
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    health_url = f"http://127.0.0.1:{selected_port}/_stcore/health"
    try:
        deadline = started + timeout_seconds
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Streamlit exited with code {process.returncode}.")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    body = response.read().decode("utf-8").strip()
                    if response.status == 200 and body == "ok":
                        return {
                            "cold_process_start_to_health_seconds": round(
                                time.perf_counter() - started, 3
                            ),
                            "http_status": response.status,
                            "body": body,
                            "python": sys.version.split()[0],
                            "port": selected_port,
                        }
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        raise TimeoutError(f"Streamlit health endpoint was not ready within {timeout_seconds:g}s.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, help="Use a specific port; otherwise choose a free port.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Startup timeout in seconds.")
    args = parser.parse_args()
    print(json.dumps(measure_startup(port=args.port, timeout_seconds=args.timeout), sort_keys=True))


if __name__ == "__main__":
    main()
