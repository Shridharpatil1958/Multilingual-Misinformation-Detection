"""
load_to_db.py
--------------
Database loader for the E-Commerce Product Price Analysis project.

Reads cleaned CSV files from a data folder and loads them into a
relational database (MySQL or SQLite) using the schema described in
the project README: Customers, Products, Orders, Sales, Suppliers,
Inventory.

Supports two backends:
    - MySQL  (via SQLAlchemy + PyMySQL)   -- good for the full project setup
    - SQLite (via SQLAlchemy)             -- zero-config, good for quick testing

Usage:
    # Load into MySQL
    python load_to_db.py --engine mysql \
        --host localhost --user root --password secret \
        --database ecommerce_analysis --data-dir data/

    # Load into a local SQLite file (no server needed)
    python load_to_db.py --engine sqlite --sqlite-path database/ecommerce.db --data-dir data/

    # Only create tables, skip loading data
    python load_to_db.py --engine sqlite --sqlite-path database/ecommerce.db --schema-only

Environment variables can be used instead of CLI flags for MySQL credentials:
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

Requirements (add to requirements.txt if not already present):
    sqlalchemy, pymysql, pandas
"""

import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------
# Table creation order matters because of foreign key dependencies.
# Adjust column types/names here to match your actual cleaned CSV columns.

SCHEMA_STATEMENTS = {
    "Suppliers": """
        CREATE TABLE IF NOT EXISTS Suppliers (
            supplier_id     INTEGER PRIMARY KEY,
            supplier_name   VARCHAR(255),
            contact_email   VARCHAR(255),
            country         VARCHAR(100)
        );
    """,
    "Products": """
        CREATE TABLE IF NOT EXISTS Products (
            product_id      INTEGER PRIMARY KEY,
            product_name    VARCHAR(255),
            category        VARCHAR(100),
            subcategory     VARCHAR(100),
            supplier_id     INTEGER,
            unit_price      DECIMAL(10, 2)
        );
    """,
    "Customers": """
        CREATE TABLE IF NOT EXISTS Customers (
            customer_id     INTEGER PRIMARY KEY,
            customer_name   VARCHAR(255),
            email           VARCHAR(255),
            city            VARCHAR(100),
            region          VARCHAR(100)
        );
    """,
    "Orders": """
        CREATE TABLE IF NOT EXISTS Orders (
            order_id        INTEGER PRIMARY KEY,
            customer_id     INTEGER,
            order_date      DATE,
            channel         VARCHAR(50)
        );
    """,
    "Sales": """
        CREATE TABLE IF NOT EXISTS Sales (
            sale_id         INTEGER PRIMARY KEY,
            order_id        INTEGER,
            product_id      INTEGER,
            quantity        INTEGER,
            revenue         DECIMAL(12, 2),
            profit          DECIMAL(12, 2)
        );
    """,
    "Inventory": """
        CREATE TABLE IF NOT EXISTS Inventory (
            inventory_id    INTEGER PRIMARY KEY,
            product_id      INTEGER,
            stock_quantity  INTEGER,
            last_updated    DATE
        );
    """,
}

# Maps table name -> expected CSV filename in the data directory.
# Update these if your cleaned files are named differently.
TABLE_CSV_MAP = {
    "Suppliers": "suppliers.csv",
    "Products": "products.csv",
    "Customers": "customers.csv",
    "Orders": "orders.csv",
    "Sales": "sales.csv",
    "Inventory": "inventory.csv",
}

# Load order respects foreign key dependencies (parents before children)
LOAD_ORDER = ["Suppliers", "Products", "Customers", "Orders", "Sales", "Inventory"]


def build_engine(args):
    """Create a SQLAlchemy engine for either MySQL or SQLite."""
    if args.engine == "mysql":
        host = args.host or os.environ.get("DB_HOST", "localhost")
        user = args.user or os.environ.get("DB_USER", "root")
        password = args.password or os.environ.get("DB_PASSWORD", "")
        database = args.database or os.environ.get("DB_NAME", "ecommerce_analysis")

        conn_str = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        return create_engine(conn_str)

    elif args.engine == "sqlite":
        sqlite_path = args.sqlite_path or "database/ecommerce.db"
        os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
        return create_engine(f"sqlite:///{sqlite_path}")

    else:
        raise ValueError(f"Unsupported engine: {args.engine}")


def create_database_if_needed(args):
    """For MySQL, create the target database first if it doesn't exist."""
    if args.engine != "mysql":
        return

    host = args.host or os.environ.get("DB_HOST", "localhost")
    user = args.user or os.environ.get("DB_USER", "root")
    password = args.password or os.environ.get("DB_PASSWORD", "")
    database = args.database or os.environ.get("DB_NAME", "ecommerce_analysis")

    admin_engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/")
    with admin_engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{database}`"))
        conn.commit()
    print(f"Ensured MySQL database '{database}' exists.")


def create_schema(engine):
    """Create all tables defined in SCHEMA_STATEMENTS."""
    with engine.connect() as conn:
        for table_name in LOAD_ORDER:
            print(f"Creating table (if not exists): {table_name}")
            conn.execute(text(SCHEMA_STATEMENTS[table_name]))
        conn.commit()
    print("Schema creation complete.\n")


def load_csv_to_table(engine, table_name, csv_path):
    """Read a CSV file and append its rows into the given table."""
    if not os.path.isfile(csv_path):
        print(f"  Skipping '{table_name}': file not found at {csv_path}")
        return 0

    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists="append", index=False)
    return len(df)


def load_all_data(engine, data_dir):
    """Load every table's CSV file from data_dir, in dependency order."""
    print(f"Loading data from '{data_dir}' ...\n")
    total_rows = 0
    for table_name in LOAD_ORDER:
        csv_filename = TABLE_CSV_MAP[table_name]
        csv_path = os.path.join(data_dir, csv_filename)
        rows_loaded = load_csv_to_table(engine, table_name, csv_path)
        if rows_loaded:
            print(f"  Loaded {rows_loaded} rows into '{table_name}' from {csv_filename}")
        total_rows += rows_loaded
    print(f"\nDone. Total rows loaded: {total_rows}")


def main():
    parser = argparse.ArgumentParser(
        description="Load cleaned CSV data into MySQL or SQLite for the "
        "E-Commerce Product Price Analysis project."
    )
    parser.add_argument(
        "--engine",
        choices=["mysql", "sqlite"],
        default="sqlite",
        help="Database backend to use (default: %(default)s).",
    )
    parser.add_argument("--host", help="MySQL host (default: localhost or $DB_HOST).")
    parser.add_argument("--user", help="MySQL user (default: root or $DB_USER).")
    parser.add_argument("--password", help="MySQL password (default: $DB_PASSWORD).")
    parser.add_argument("--database", help="MySQL database name (default: ecommerce_analysis or $DB_NAME).")
    parser.add_argument(
        "--sqlite-path",
        help="Path to the SQLite database file (default: database/ecommerce.db).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing cleaned CSV files (default: %(default)s).",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only create tables; skip loading CSV data.",
    )
    args = parser.parse_args()

    try:
        create_database_if_needed(args)
        engine = build_engine(args)
        create_schema(engine)

        if not args.schema_only:
            if not os.path.isdir(args.data_dir):
                print(f"Error: data directory '{args.data_dir}' not found.")
                sys.exit(1)
            load_all_data(engine, args.data_dir)
        else:
            print("Schema-only mode: skipped data loading.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
