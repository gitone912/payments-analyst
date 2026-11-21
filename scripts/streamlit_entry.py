"""Starts Streamlit with a Windows-friendly event loop. Used by run_app.py; not meant to be run directly.

On Windows, Python's default Proactor event loop prints
`ConnectionResetError: [WinError 10054]` every time a browser closes its connection. The selector
loop does not, and works fine for a local Streamlit server.
"""
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from streamlit.web import cli  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", *sys.argv[1:]]
    sys.exit(cli.main())
