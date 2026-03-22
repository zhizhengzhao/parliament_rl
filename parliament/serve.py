"""
Science Parliament — visualization server.

Starts a local HTTP server on the run output directory.
Access via SSH tunnel from your local machine.

Usage:
    # On server:
    cd parliament
    python serve.py --output_dir ../output/<timestamp>/

    # On your local machine:
    ssh -p 8795 -L 18888:localhost:18888 root@your-server-ip

    # Then open: http://localhost:18888/index.html
"""

import argparse
import os
import signal
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler


def main():
    parser = argparse.ArgumentParser(description="Science Parliament visualization server")
    parser.add_argument("--output_dir", required=True, help="Path to run output directory")
    parser.add_argument("--port", type=int, default=18888)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"[serve] Error: directory not found: {output_dir}")
        sys.exit(1)

    os.chdir(output_dir)
    handler = type("H", (SimpleHTTPRequestHandler,), {"log_message": lambda *a: None})
    server = HTTPServer(("", args.port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"[serve] Serving {output_dir}")
    print(f"[serve] http://localhost:{args.port}/index.html")
    print(f"[serve]")
    print(f"[serve] On your local machine, run:")
    print(f"[serve]   ssh -p 8795 -L {args.port}:localhost:{args.port} root@your-server-ip")
    print(f"[serve]")
    print(f"[serve] Press Ctrl+C to stop.")

    try:
        signal.pause()
    except (AttributeError, KeyboardInterrupt):
        pass
    server.shutdown()
    print("\n[serve] Stopped.")


if __name__ == "__main__":
    main()
