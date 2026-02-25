from pathlib import Path
from sqlalchemy import create_engine
from .logger import Logger


class DBEngine:
    def __init__(self, dbuser, dbpassword, dbhost, dbport, dbname):
        self.dbuser = dbuser
        self.dbpassword = dbpassword
        self.dbhost = dbhost
        self.dbport = dbport
        self.dbname = dbname

        # Setup logger as class attribute
        self.logger = Logger.get_logger(
            name=self.__class__.__name__,
            log_file_path=Path("logs") / "logs.log",
        )

        self.engine = self._create_db_engine()
        self.logger.info("Created database engine.")
        self._test_db_connection()

    # Datenbankverbindung
    def _create_db_engine(self):
        """
        Creates a connection to the PostgreSQL database.

        Returns:
            SQLAlchemy Engine: A connection to the database
        """
        return create_engine(
            f"postgresql+psycopg://{self.dbuser}:{self.dbpassword}@{self.dbhost}:{self.dbport}/{self.dbname}"
        )

    # Überprüfung der Datenbankverbindung
    def _test_db_connection(self):
        """
        Tests the connection to the database and prints information.

        Args:
            engine: SQLAlchemy Engine, optional. If None, a new engine is created.

        Returns:
            bool: True if the connection was successful, False otherwise
        """
        if self.engine is None:
            self.engine = self._create_db_engine()

        try:
            with self.engine.connect() as conn:
                result = conn.exec_driver_sql(
                    "SELECT current_user, current_database();"
                ).fetchone()
                self.logger.info(f"Database connection successfully established!")
                return True
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            return False
