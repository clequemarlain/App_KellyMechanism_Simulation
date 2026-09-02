"""PyCharm-friendly launcher for the Kelly simulator."""

from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "app.py"


def interpreter_has_streamlit(python: Path) -> bool:
    """Return whether an interpreter can import Streamlit."""
    if Path(sys.executable).resolve() == python.resolve():
        return importlib.util.find_spec("streamlit") is not None
    try:
        check = subprocess.run(
            [str(python), "-c", "import streamlit"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return check.returncode == 0


def select_python() -> Path | None:
    """Prefer the active interpreter, then the project's virtual environment."""
    active_python = Path(sys.executable)
    if interpreter_has_streamlit(active_python):
        return active_python

    venv_candidates = (
        PROJECT_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    )
    return next(
        (
            python
            for python in venv_candidates
            if python.is_file() and interpreter_has_streamlit(python)
        ),
        None,
    )


def port_is_available(port: int) -> bool:
    """Check whether a local TCP port can be used by the Streamlit server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
    except OSError:
        return False
    return True


def select_port(preferred_port: int) -> int | None:
    """Return the preferred port or the next available local port."""
    for port in range(preferred_port, min(preferred_port + 100, 65536)):
        if port_is_available(port):
            return port
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Kelly simulator in your web browser."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Local web server port (default: 8501).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening a browser window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        print("Error: --port must be between 1 and 65535.", file=sys.stderr)
        return 2
    port = select_port(args.port)
    if port is None:
        print(
            f"Error: no available port found between {args.port} "
            f"and {min(args.port + 99, 65535)}.",
            file=sys.stderr,
        )
        return 1
    python = select_python()
    if python is None:
        print(
            "Streamlit was not found in the active interpreter or the project .venv.\n"
            "Create the project environment and install its dependencies with:\n"
            f'  "{sys.executable}" -m venv "{PROJECT_ROOT / ".venv"}"\n'
            f'  "{PROJECT_ROOT / ".venv" / "bin" / "python"}" -m pip install '
            f'-r "{PROJECT_ROOT / "requirements.txt"}"',
            file=sys.stderr,
        )
        return 1

    command = [
        str(python),
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(port),
        "--server.headless",
        str(args.no_browser).lower(),
        "--browser.gatherUsageStats",
        "false",
    ]
    if python.resolve() != Path(sys.executable).resolve():
        print(f"Using project interpreter: {python}")
    if port != args.port:
        print(f"Port {args.port} is busy; using port {port} instead.")
    print(f"Starting Kelly Simulator at http://localhost:{port}")
    try:
        return subprocess.call(command, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
