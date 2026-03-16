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
    --refresh N            Override page auto-refresh interval in seconds
                           (0 or negative = disable auto-refresh)

The public URL is printed once and stays valid until you press Ctrl+C.
"""

import argparse
import os
import re
import signal
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def _make_handler(refresh_secs):
    """Return a handler class with the refresh override baked in."""

    class _Handler(SimpleHTTPRequestHandler):

        def log_message(self, *args):
            pass

        def log_error(self, *args):
            pass

        def do_GET(self):
            # Only intercept index.html when a refresh override is requested
            if refresh_secs is not None and self.path.rstrip('/') in (
                '', '/index.html', 'index.html'
            ):
                try:
                    full_path = os.path.join(os.getcwd(), 'index.html')
                    with open(full_path, 'rb') as f:
                        content = f.read().decode('utf-8')

                    if refresh_secs > 0:
                        # Replace whatever interval is baked in with the new one
                        content = re.sub(
                            r'<meta http-equiv="refresh" content="\d+">',
                            f'<meta http-equiv="refresh" content="{refresh_secs}">',
                            content,
                        )
                        content = re.sub(
                            r'Auto-refreshes every \d+s',
                            f'Auto-refreshes every {refresh_secs}s',
                            content,
                        )
                    else:
                        # Remove auto-refresh entirely
                        content = re.sub(
                            r'<meta http-equiv="refresh" content="\d+">',
                            '',
                            content,
                        )
                        content = re.sub(
                            r'Auto-refreshes every \d+s',
                            'Auto-refresh disabled',
                            content,
                        )

                    encoded = content.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                    return
                except FileNotFoundError:
                    pass  # index.html not yet generated; fall through to default 404
                except Exception:
                    pass  # any other error: fall through to normal file serving

            super().do_GET()

    return _Handler


def _start_http_server(directory: str, port: int, refresh_secs) -> HTTPServer:
    os.chdir(directory)
    handler = _make_handler(refresh_secs)
    server = HTTPServer(("", port), handler)
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
    parser.add_argument(
        "--refresh", type=int, default=None, metavar="N",
        help=(
            "Override page auto-refresh interval in seconds. "
            "Use 0 or negative to disable. Omit to keep the default (15s)."
        ),
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

    # Normalise refresh: treat anything <= 0 as "disabled" (store as 0)
    refresh_secs = args.refresh
    if refresh_secs is not None and refresh_secs <= 0:
        refresh_secs = 0

    # Start HTTP server
    server = _start_http_server(output_dir, args.port, refresh_secs)
    print(f"[serve] HTTP server → http://localhost:{args.port}/index.html")

    if refresh_secs is None:
        print("[serve] Auto-refresh: 15s (default)")
    elif refresh_secs == 0:
        print("[serve] Auto-refresh: disabled")
    else:
        print(f"[serve] Auto-refresh: {refresh_secs}s")

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
        refresh_label = (
            "disabled" if refresh_secs == 0
            else f"every {refresh_secs}s" if refresh_secs
            else "every 15s"
        )
        print()
        print("=" * 60)
        print(f"  Public URL : {public_url}/index.html")
        print(f"  Auto-refresh: {refresh_label}")
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
