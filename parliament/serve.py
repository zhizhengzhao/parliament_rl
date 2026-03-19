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

Optional flags:
    --port 18888       HTTP port (default 18888)
    --refresh N        Override auto-refresh interval in seconds (0 = disable)
"""

import argparse
import os
import re
import signal
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler


def _make_handler(refresh_secs):
    class _Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if refresh_secs is not None and self.path.rstrip('/') in (
                '', '/index.html', 'index.html'
            ):
                try:
                    with open(os.path.join(os.getcwd(), 'index.html'), 'rb') as f:
                        content = f.read().decode('utf-8')
                    if refresh_secs > 0:
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
                        content = re.sub(
                            r'<meta http-equiv="refresh" content="\d+">',
                            '', content,
                        )
                        content = re.sub(
                            r'Auto-refreshes every \d+s',
                            'Auto-refresh disabled', content,
                        )
                    encoded = content.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                    return
                except Exception:
                    pass
            super().do_GET()

    return _Handler


def main():
    parser = argparse.ArgumentParser(description="Science Parliament visualization server")
    parser.add_argument("--output_dir", required=True, help="Path to run output directory")
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--refresh", type=int, default=None, metavar="N",
                        help="Auto-refresh interval in seconds (0 = disable)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"[serve] Error: directory not found: {output_dir}")
        sys.exit(1)

    refresh = args.refresh
    if refresh is not None and refresh <= 0:
        refresh = 0

    os.chdir(output_dir)
    handler = _make_handler(refresh)
    server = HTTPServer(("", args.port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"[serve] Serving {output_dir}")
    print(f"[serve] http://localhost:{args.port}/index.html")
    if refresh is None:
        print(f"[serve] Auto-refresh: 15s (default)")
    elif refresh == 0:
        print(f"[serve] Auto-refresh: disabled")
    else:
        print(f"[serve] Auto-refresh: {refresh}s")
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
