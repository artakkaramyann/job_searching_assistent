"""SQLite: читаемые актуальные анкеты и совместимость со старыми профилями.

Checkpoints LangGraph — источник состояния выполнения; profiles — его проекция.
"""

import json
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

Stage = Literal["target_role", "about", "followup", "complete"]

JSON_FIELDS = {"turns_json": ("turns", []), "facts_json": ("facts", []),
               "summaries_json": ("summaries", []), "languages_json": ("languages", []),
               "review_json": ("review", None)}


@dataclass
class Profile:
    user_id: int
    stage: Stage = "target_role"
    target_role: str = ""
    about: str = ""


class ProfileStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS profiles (
                    user_id INTEGER PRIMARY KEY,
                    stage TEXT NOT NULL,
                    target_role TEXT NOT NULL,
                    about TEXT NOT NULL
                )"""
            )
            connection.execute("""CREATE TABLE IF NOT EXISTS telegram_ui (
                user_id INTEGER PRIMARY KEY, mode TEXT NOT NULL
            )""")
            columns = {r[1] for r in connection.execute("PRAGMA table_info(profiles)")}
            for column, (_, default) in JSON_FIELDS.items():
                if column not in columns:
                    value = json.dumps(default)
                    connection.execute(f"ALTER TABLE profiles ADD COLUMN {column} TEXT NOT NULL DEFAULT '{value}'")
            if "updated_at" not in columns:
                connection.execute("ALTER TABLE profiles ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")

    def get_ui_mode(self, user_id: int) -> str:
        with self._connect() as connection:
            row = connection.execute("SELECT mode FROM telegram_ui WHERE user_id = ?", (user_id,)).fetchone()
        return row[0] if row else "dialog"

    def set_ui_mode(self, user_id: int, mode: str) -> None:
        with self._connect() as connection:
            connection.execute("""INSERT INTO telegram_ui (user_id, mode) VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET mode = excluded.mode""", (user_id, mode))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, user_id: int) -> Profile:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id, stage, target_role, about FROM profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return Profile(*row) if row else Profile(user_id=user_id)

    def save(self, profile: Profile) -> None:
        self.save_state(profile.user_id, {"stage": profile.stage, "target_role": profile.target_role,
                                         "about": profile.about})

    def save_state(self, user_id: int, state: dict) -> None:
        values = {"user_id": user_id, "stage": state["stage"], "target_role": state["target_role"],
                  "about": state["about"]}
        values.update({column: json.dumps(state.get(key, default), ensure_ascii=False)
                       for column, (key, default) in JSON_FIELDS.items()})
        values["updated_at"] = datetime.now(timezone.utc).isoformat()
        columns = list(values)
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO profiles ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)}) "
                "ON CONFLICT(user_id) DO UPDATE SET " +
                ", ".join(f"{c} = excluded.{c}" for c in columns if c != "user_id"),
                tuple(values.values()),
            )

    def checkpoint_users(self) -> list[int]:
        with self._connect() as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name = 'checkpoints'").fetchone():
                return []
            rows = connection.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''").fetchall()
        return [int(row[0]) for row in rows if row[0].lstrip('-').isdigit()]

