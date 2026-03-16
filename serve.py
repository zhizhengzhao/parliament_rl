"""
Science Parliament — visualization server.

Starts a local HTTP server on the run output directory and (optionally)
opens a public ngrok tunnel so anyone can view the live forum page.

Usage:
    python serve.py --output_dir output/2026-03-16_14-30-00/

Optional flags:
    --port 18888           HTTP port (default 18888)
    --token YOUR_TOKEN     ngrok authtoken (or set NGROK_AUTHTOKEN env var)
    --no-ngrok             Skip ngrok, just serve locally

The public URL is printed once and stays valid until you press Ctrl+C.
"""

import argparse
import os
import signal
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _SilentHandler(SimpleHTTPRequestHandler):
    """HTTP handler with access logs suppressed."""

    def log_message(self, *args):
        pass

    def log_error(self, *args):
        pass


def _start_http_server(directory: str, port: int) -> HTTPServer:
    os.chdir(directory)
    server = HTTPServer(("", port), _SilentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Science Parliament visualization server"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Path to a timestamped run directory (the one with parliament.db)",
    )
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument(
        "--token", default=None,
        help="ngrok authtoken (overrides NGROK_AUTHTOKEN env var)",
    )
    parser.add_argument(
        "--no-ngrok", action="store_true",
        help="Skip ngrok, just serve locally",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"[serve] Error: directory not found: {output_dir}")
        sys.exit(1)

    index_path = os.path.join(output_dir, "index.html")
    if not os.path.exists(index_path):
        print(f"[serve] Warning: index.html not found yet in {output_dir}")
        print("[serve] It will appear after the first round completes.")

    # Start HTTP server
    server = _start_http_server(output_dir, args.port)
    print(f"[serve] HTTP server → http://localhost:{args.port}/index.html")

    if args.no_ngrok:
        print("[serve] ngrok disabled. Access via SSH tunnel:")
        print(f"         ssh -L {args.port}:localhost:{args.port} user@server")
        print("[serve] Press Ctrl+C to stop.")
        try:
            signal.pause()
        except (AttributeError, KeyboardInterrupt):
            pass
        server.shutdown()
        return

    # Try ngrok
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("[serve] pyngrok not installed → run: pip install pyngrok")
        print("[serve] Falling back to local-only mode.")
        print(f"[serve] SSH tunnel: ssh -L {args.port}:localhost:{args.port} user@server")
        try:
            signal.pause()
        except (AttributeError, KeyboardInterrupt):
            pass
        server.shutdown()
        return

    token = args.token or os.environ.get("NGROK_AUTHTOKEN")
    if token:
        conf.get_default().auth_token = token

    try:
        tunnel = ngrok.connect(args.port)
        public_url = tunnel.public_url
        print()
        print("=" * 60)
        print(f"  🌐  Public URL : {public_url}/index.html")
        print(f"       Share this link — updates every 8 seconds")
        print("=" * 60)
        print()
        print("[serve] Press Ctrl+C to stop.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    finally:
        try:
            ngrok.kill()
        except Exception:
            pass
        server.shutdown()
        print("\n[serve] Stopped.")


if __name__ == "__main__":
    main()
