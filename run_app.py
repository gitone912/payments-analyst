"""
Start the app:   python run_app.py            (any extra args go to `streamlit run`, e.g. --server.port 8502)

Why not just `streamlit run app.py`? On Windows that prints harmless ConnectionResetError tracebacks
when a browser tab closes, and Ctrl+C can leave it on "Stopping..." for a long time. This launcher
runs Streamlit with the selector event loop and, on Ctrl+C, gives it a few seconds to stop cleanly
before ending it, so the terminal is always back promptly.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GRACE_S = 5.0


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> int:
    import signal
    # Ctrl+C must always reach us (a parent shell may have set it to "ignore"), and a plain
    # terminate request takes the same clean path, so the server is never left running.
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _raise_interrupt)
    cmd = [sys.executable, os.path.join(HERE, "scripts", "streamlit_entry.py"),
           os.path.join(HERE, "app.py"), *sys.argv[1:]]
    kwargs = {"cwd": HERE}
    if sys.platform == "win32":
        # own process group: our Ctrl+C handling decides when the server stops
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(cmd, **kwargs)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\nStopping the app...", flush=True)
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # Streamlit's clean shutdown path
        else:
            proc.terminate()
        deadline = time.monotonic() + GRACE_S
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            print(f"Still running after {GRACE_S:.0f}s; ending it.", flush=True)
            proc.kill()
            proc.wait()
        print("Stopped.", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
