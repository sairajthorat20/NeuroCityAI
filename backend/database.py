"""
database.py
-----------
SQLite user database for NeuroCity authentication.
Creates a local `neurocity.db` file in the backend directory.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "neurocity.db"


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the users table if it does not exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name       TEXT    NOT NULL,
                email           TEXT    UNIQUE NOT NULL,
                hashed_password TEXT    NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def get_user_by_email(email: str) -> dict | None:
    """Fetch a user row by email; returns dict or None."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        return dict(row) if row else None


def create_user(full_name: str, email: str, hashed_password: str) -> None:
    """Insert a new user record."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO users (full_name, email, hashed_password) VALUES (?, ?, ?)",
            (full_name, email, hashed_password),
        )
        conn.commit()
