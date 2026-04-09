"""Parliament authentication."""

from fastapi import Depends, HTTPException, Header

from .config import ADMIN_KEY
from .store import Store

_store: Store | None = None


def set_store(store: Store):
    global _store
    _store = store


def get_current_user(authorization: str = Header(default="")) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Bearer token. "
                            "Use: Authorization: Bearer <your_key>")
    key = authorization[7:].strip()
    if _store is None:
        raise HTTPException(500, "Store not initialized")
    if not key:
        raise HTTPException(401, "Empty API key")
    user = _store.get_user_by_key(key)
    if user is None:
        raise HTTPException(401, "Invalid API key")
    user["is_admin"] = (key == ADMIN_KEY)
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if not user.get("is_admin"):
        raise HTTPException(403, "Admin access required")
    return user
