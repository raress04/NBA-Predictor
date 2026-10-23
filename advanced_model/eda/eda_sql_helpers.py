"""
eda_sql_helpers.py
==================
Read-only SQLite connection helpers for EDA.

Responsibilities:
  - Open connections in SQLite URI read-only mode.
  - Enforce SELECT/PRAGMA/EXPLAIN-only queries at runtime.
  - Table enumeration and schema introspection.
  - Row count helpers.

Must never be imported outside the eda/ package.
"""

import sqlite3
import os
import re
from typing import List, Dict, Tuple, Optional


# ── Read-only connection ──────────────────────────────────────────────────────

def get_connection_read_only(db_path: str) -> sqlite3.Connection:
    """
    Open a SQLite database in read-only URI mode.

    Parameters
    ----------
    db_path : str
        Absolute or relative path to the .db file.

    Returns
    -------
    sqlite3.Connection

    Raises
    ------
    FileNotFoundError
        If the database file does not exist on disk.
    sqlite3.OperationalError
        If the URI connection fails.
    """
    abs_path = os.path.abspath(db_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"[EDA] Database not found: {abs_path}"
        )
    # SQLite URI read-only mode — prevents any accidental writes.
    uri = f"file:{abs_path}?mode=ro&cache=shared"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ── Safe query execution ──────────────────────────────────────────────────────

_ALLOWED_PREFIX = re.compile(
    r'^\s*(SELECT|PRAGMA|EXPLAIN)\b',
    re.IGNORECASE
)

def execute_select(
    conn: sqlite3.Connection,
    sql: str,
    params: Tuple = ()
) -> List[Dict]:
    """
    Execute a read-only SQL statement and return rows as list of dicts.

    Parameters
    ----------
    conn   : sqlite3.Connection  (must be a read-only connection)
    sql    : str                 Query string.
    params : tuple               Positional parameters for the query.

    Returns
    -------
    List[dict]

    Raises
    ------
    ValueError
        If the SQL statement does not start with SELECT, PRAGMA, or EXPLAIN.
    """
    if not _ALLOWED_PREFIX.match(sql):
        raise ValueError(
            f"[EDA] Forbidden SQL verb — EDA is read-only.\n"
            f"       Offending statement: {sql[:120]!r}"
        )
    cursor = conn.execute(sql, params)
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ── Table enumeration ─────────────────────────────────────────────────────────

def list_tables(conn: sqlite3.Connection) -> List[Dict]:
    """
    List all user tables and views in the connected database.

    Returns
    -------
    List[dict]  — each dict has keys: schema, name, type
    """
    rows = execute_select(
        conn,
        "SELECT tbl_name AS name, type "
        "FROM sqlite_master "
        "WHERE type IN ('table','view') "
        "ORDER BY type, tbl_name"
    )
    # Normalise: add schema='main' for compatibility
    for r in rows:
        r.setdefault('schema', 'main')
    return rows


# ── Row counting ──────────────────────────────────────────────────────────────

def get_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    """Return the total number of rows in *table_name*."""
    rows = execute_select(conn, f"SELECT COUNT(*) AS n FROM \"{table_name}\"")
    return int(rows[0]['n']) if rows else 0


def get_date_range(
    conn: sqlite3.Connection,
    table_name: str,
    date_col: str
) -> Dict[str, Optional[str]]:
    """
    Return the min/max value of *date_col* in *table_name*.

    Returns
    -------
    dict with keys 'min_date' and 'max_date' (str or None).
    """
    rows = execute_select(
        conn,
        f"SELECT MIN(\"{date_col}\") AS min_date, "
        f"       MAX(\"{date_col}\") AS max_date "
        f"FROM \"{table_name}\""
    )
    if rows:
        return {'min_date': rows[0]['min_date'], 'max_date': rows[0]['max_date']}
    return {'min_date': None, 'max_date': None}


# ── Schema introspection ──────────────────────────────────────────────────────

def get_table_schema(conn: sqlite3.Connection, table_name: str) -> List[Dict]:
    """
    Retrieve column metadata for *table_name* via PRAGMA table_info.

    Returns
    -------
    List[dict] — each dict has keys:
        cid, name, type, notnull, default_value, pk
    """
    rows = execute_select(conn, f"PRAGMA table_info(\"{table_name}\")")
    result = []
    for r in rows:
        result.append({
            'cid':           r.get('cid'),
            'name':          r.get('name'),
            'type':          r.get('type'),
            'notnull':       bool(r.get('notnull', 0)),
            'default_value': r.get('dflt_value'),
            'pk':            bool(r.get('pk', 0)),
        })
    return result


# ── Known date column names (for automatic date-range detection) ──────────────

DATE_COLUMN_CANDIDATES: List[str] = [
    'game_date', 'gamedate', 'date', 'created_at', 'updated_at',
    'pick_date', 'result_date',
]

def detect_date_column(schema: List[Dict]) -> Optional[str]:
    """
    Return the first column in *schema* whose name matches a known date alias.
    Returns None if no date column is found.
    """
    col_names_lower = {col['name'].lower(): col['name'] for col in schema}
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in col_names_lower:
            return col_names_lower[candidate]
    return None
