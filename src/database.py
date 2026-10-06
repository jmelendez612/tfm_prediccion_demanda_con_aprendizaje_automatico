import sqlite3
import pandas as pd
from src import rutas

class DatabaseManager:
    def __init__(self, db_path=rutas.DB_PATH):
        self.db_path = db_path

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def save_to_db(self, df, table_name, mode='replace'):
        with self.get_connection() as conn:
            df.to_sql(table_name, conn, if_exists=mode, index=False)
    
    def load_from_db(self, query):
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)