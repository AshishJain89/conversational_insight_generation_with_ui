import os, sqlite3
from typing import List, Tuple

DB_PATH = os.getenv('DB_PATH','./data/northwind.db')


def get_connection(path: str = DB_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database file not found at {path}")
    return sqlite3.connect(path)


def list_tables(path: str = DB_PATH) -> List[str]:
    """Return a list of all user-defined tables in the database"""
    con = get_connection(path)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name not LIKE 'sqlite_%'")
    tables = [t[0] for t in cur.fetchall()]
    con.close()
    return tables


def fetch_schema(path: str = DB_PATH) -> str:
    con = get_connection(path)
    cur = con.cursor()
    schema = []
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    for (t,) in cur.fetchall():
        cur.execute(f'PRAGMA table_info(t)')
        cols = [c[1] for c in cur.fetchall()]
        schema.append(f'Table {t}: {', '.join(cols)}')
    con.close()
    return '\n'.join(schema)


def safe_select(sql: str, path: str = DB_PATH):
    con = get_connection(path)
    cur = con.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    con.close()

    # Convert rows to serializable format
    serializable_rows = [[str(cell) if cell is not None else None for cell in row] for row in rows]
    return cols, serializable_rows


if __name__ == "__main__":
    print("Tables in Northwind DB:")
    print(list_tables())
    print("\nSchema overview:")
    print(fetch_schema())
    cols, rows = safe_select("SELECT CompanyName, ContactName FROM Customers LIMIT 5;")
    print("\nSample query result:")
    print(cols, rows)
