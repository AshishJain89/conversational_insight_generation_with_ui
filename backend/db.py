import os, sqlite3
from typing import List, Tuple

def init_db(path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript('''
    CREATE TABLE IF NOT EXISTS customers(id INTEGER PRIMARY KEY, name TEXT, city TEXT, signup_date TEXT);
    CREATE TABLE IF NOT EXISTS products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
    CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY, customer_id INT, product_id INT, quantity INT, order_date TEXT);
    ''')
    con.commit(); con.close()

def fetch_schema(path: str) -> str:
    con = sqlite3.connect(path); cur = con.cursor()
    schema=[]; cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for (t,) in cur.fetchall():
        cur.execute(f'PRAGMA table_info({t})')
        cols=[c[1] for c in cur.fetchall()]
        schema.append(f"Table {t}: {', '.join(cols)}")
    con.close(); return '\n'.join(schema)

def safe_select(path: str, sql: str):
    con = sqlite3.connect(path); cur = con.cursor()
    cur.execute(sql); rows=cur.fetchall()
    cols=[d[0] for d in cur.description]
    con.close(); return cols, rows
