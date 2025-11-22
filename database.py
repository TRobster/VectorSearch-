import sqlite3
import numpy as np

DB_PATH = "lvse.db"

def get_db():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        document_id INTEGER PRIMARY KEY,
        vector BLOB NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    );
    """)

    conn.commit()
    conn.close()