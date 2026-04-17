"""Create users for Parliament experiments."""

from __future__ import annotations

from .config import ADMIN_KEY
from .store import Store


def seed_data(store: Store, num_actors: int = 3, num_judges: int = 3) -> None:
    existing = {u["name"] for u in store.list_users()}

    if "Admin" not in existing:
        store.create_user("Admin", role="admin",
                          bio="Platform administrator", api_key=ADMIN_KEY)

    for i in range(1, num_actors + 1):
        name = f"Scientist_{i}"
        if name not in existing:
            store.create_user(name, role="actor")

    for i in range(1, num_judges + 1):
        name = f"Judge_{i}"
        if name not in existing:
            store.create_user(name, role="judge")

    print(f"  Users: {len(store.list_users())} "
          f"({num_actors} actors, {num_judges} judges)")
