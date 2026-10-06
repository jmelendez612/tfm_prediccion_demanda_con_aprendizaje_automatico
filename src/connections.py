import os
import sqlite3

from dotenv import load_dotenv

from src import rutas

# Cargar variables del archivo .env (si existe)
load_dotenv()

VARIABLES_ORACLE = ("ORACLE_USER", "ORACLE_PASSWORD", "ORACLE_DSN")


class ConnectionManager:
    @staticmethod
    def get_oracle_connection():
        """
        Establece conexión con el ERP Oracle institucional.

        Esta etapa requiere acceso al ERP de la institución. Si faltan las
        credenciales (archivo .env) o la librería oracledb, informa con un
        mensaje claro y devuelve None, en lugar de lanzar un error.
        """
        faltantes = [v for v in VARIABLES_ORACLE if not os.getenv(v)]
        if faltantes:
            print("Esta etapa requiere acceso al ERP institucional (Oracle).")
            print(f"   Faltan las variables de entorno: {', '.join(faltantes)}")
            print("   Copia .env.example a .env y complétalo si tienes acceso.")
            print("   Sin acceso, usa los datos locales ya preparados en data/ "
                  "(ver README) y omite las opciones 1 a 3.")
            return None

        try:
            import oracledb
        except ImportError:
            print("Falta la librería 'oracledb'. Instálala con: pip install oracledb")
            return None

        try:
            conn = oracledb.connect(
                user=os.getenv("ORACLE_USER"),
                password=os.getenv("ORACLE_PASSWORD"),
                dsn=os.getenv("ORACLE_DSN"),
            )
            print("Conexión a Oracle exitosa")
            return conn
        except Exception as e:
            print(f"Error conectando a Oracle: {e}")
            return None

    @staticmethod
    def get_sqlite_connection(db_path=rutas.DB_PATH):
        """Establece conexión con la base de datos local del proyecto"""
        try:
            conn = sqlite3.connect(db_path)
            return conn
        except Exception as e:
            print(f"Error conectando a SQLite: {e}")
            return None
