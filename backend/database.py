"""
database.py
-----------
Database connection helper for NeuroCity authentication.
Automatically uses PostgreSQL if the `DATABASE_URL` environment
variable is set (for hosting on Render + Neon.tech).
Otherwise, falls back to a local SQLite database (for local dev).
"""

import os
import sqlite3
from pathlib import Path
import psycopg2
from psycopg2.extras import DictCursor

DB_PATH = Path(__file__).resolve().parent / "neurocity.db"
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_connection():
    """Return a database connection and a boolean indicating if it's Postgres."""
    if DATABASE_URL:
        # Use PostgreSQL (e.g. Neon.tech, Supabase, Render)
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=DictCursor)
        return conn, True
    else:
        # Fall back to local SQLite
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn, False


def init_db():
    """Create the users table if it does not exist."""
    conn, is_pg = get_connection()
    try:
        cursor = conn.cursor()
        if is_pg:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id              SERIAL PRIMARY KEY,
                    full_name       VARCHAR(80)  NOT NULL,
                    email           VARCHAR(120) UNIQUE NOT NULL,
                    hashed_password VARCHAR(255) NOT NULL,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name       TEXT    NOT NULL,
                    email           TEXT    UNIQUE NOT NULL,
                    hashed_password TEXT    NOT NULL,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
    finally:
        conn.close()


def get_user_by_email(email: str) -> dict | None:
    """Fetch a user row by email; returns dict or None."""
    conn, is_pg = get_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM users WHERE email = %s" if is_pg else "SELECT * FROM users WHERE email = ?"
        cursor.execute(query, (email,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_user(full_name: str, email: str, hashed_password: str) -> None:
    """Insert a new user record."""
    conn, is_pg = get_connection()
    try:
        cursor = conn.cursor()
        query = (
            "INSERT INTO users (full_name, email, hashed_password) VALUES (%s, %s, %s)" 
            if is_pg else 
            "INSERT INTO users (full_name, email, hashed_password) VALUES (?, ?, ?)"
        )
        cursor.execute(query, (full_name, email, hashed_password))
        conn.commit()
    finally:
        conn.close()
