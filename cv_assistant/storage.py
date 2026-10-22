"""SQLite сохраняет анкету между сообщениями и перезапусками программы."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

Stage = Literal["target_role", "about", "complete"]


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
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO profiles (user_id, stage, target_role, about)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       stage = excluded.stage,
                       target_role = excluded.target_role,
                       about = excluded.about""",
                (profile.user_id, profile.stage, profile.target_role, profile.about),
            )

