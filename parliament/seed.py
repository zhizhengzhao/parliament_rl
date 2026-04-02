"""Create users for Parliament experiments."""

from .config import ADMIN_KEY


def seed_data(store, num_actors: int = 4, num_judges: int = 4):
    existing = {u["name"] for u in store.list_users()}

    if "Admin" not in existing:
        store._write(
            "INSERT INTO users (name, api_key, role, bio) VALUES (?,?,?,?)",
            ("Admin", ADMIN_KEY, "admin", "Platform administrator"),
        )

    for i in range(1, num_actors + 1):
        name = f"Scientist_{i}"
        if name not in existing:
            store.create_user(name, role="actor")

    for i in range(1, num_judges + 1):
        name = f"Judge_{i}"
        if name not in existing:
            store.create_user(name, role="judge")

    users = store.list_users()
    print(f"  Users: {len(users)} ({num_actors} actors, {num_judges} judges)")
