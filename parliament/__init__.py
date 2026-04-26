"""Parliament server — FastAPI HTTP API on top of a SQLite store.

The data-generation half's *backend*. Speaks pure HTTP and knows
nothing about LLMs: any client (curl, the harness, an external
orchestrator) can drive it. All experiment state lives in the
SQLite DB; restarting the server doesn't lose anything.

Entry point: ``python -m parliament.server`` (see ``server.main``).
"""
