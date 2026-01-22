import json
import os
import sqlite3
from contextlib import contextmanager


DB_DIR = os.path.expanduser("~/.vespatune")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "vespatune.db")


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                pid INTEGER,
                config TEXT,
                project_name TEXT,
                output_dir TEXT,
                db_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            )
        """
        )

        # Add project_name column if it doesn't exist (for existing databases)
        try:
            conn.execute("ALTER TABLE runs ADD COLUMN project_name TEXT")
        except Exception:
            pass  # Column already exists

        conn.commit()


def create_run(config: dict, project_name: str, output_dir: str, db_path: str) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO runs (status, config, project_name, output_dir, db_path) VALUES (?, ?, ?, ?, ?)",
            ("pending", json.dumps(config), project_name, output_dir, db_path),
        )
        conn.commit()
        return cur.lastrowid


def update_run_status(run_id: int, status: str, pid: int = None, error: str = None):
    with get_db() as conn:
        updates = ["status = ?"]
        params = [status]
        if pid is not None:
            updates.append("pid = ?")
            params.append(pid)
        if error is not None:
            updates.append("error_message = ?")
            params.append(error)

        params.append(run_id)
        conn.execute(f"UPDATE runs SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()


def get_run(run_id: int):
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_active_run():
    with get_db() as conn:
        # Check for running or pending runs
        cur = conn.execute("SELECT * FROM runs WHERE status IN ('pending', 'running') ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            return dict(row)

        # If no active run, return the last completed/stopped/error run for context
        cur = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        return dict(row) if row else None


def get_all_runs():
    with get_db() as conn:
        cur = conn.execute(
            "SELECT id, status, created_at, config, output_dir, project_name FROM runs ORDER BY id DESC"
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows]


def delete_run(run_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.commit()
