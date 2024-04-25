from AGI.io.base_io import BaseIO
import os
import sqlite3
import pandas as pd
import sys

class SQLIO(BaseIO):
    def __init__(self, file, force_overwrite=False, read_only=False, timeout=900):
        super().__init__(file, force_overwrite)
        self.read_only = read_only
        self.timeout = timeout
        self.conn = self.establish_connection()

    # Establish connection to database
    def establish_connection(self):
        # Check if the database file already exists and delete it if it does
        # The additional check for self.force_overwrite is redundant, as it is already checked in check_overwrite()
        # but better to be safe than sorry.
        if os.path.exists(self.file) and self.force_overwrite: 
            os.remove(self.file)

        try:
            return sqlite3.connect(self.file, timeout=self.timeout)
        except Exception as e:
            sys.exit(f"Failed to connect to database: {e}")
    

    # Convenience function to execute SQL queries
    # Query database
    def query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.conn)

    # Query full table
    def get_table(self, tname: str) -> pd.DataFrame:
        # Broken: return pd.read_sql_table(tname, self.con)
        return self.query(f"SELECT * FROM \"{tname}\"")
    
    # Decorator to prevent write operations in read-only mode
    def write_protected(func):
        def wrapper(self, *args, **kwargs):
            if self.read_only:
                sys.exit("Database is in read-only mode.")
            return func(self, *args, **kwargs)
        return wrapper
    
    # Create table
    @write_protected
    def create_table(self, tname: str, df: pd.DataFrame, if_exists='fail') -> None:
        df.to_sql(tname, self.conn, if_exists=if_exists, index=False)

    # Append data to table
    @write_protected
    def append_to_table(self, tname: str, df: pd.DataFrame) -> None:
        df.to_sql(tname, self.conn, if_exists='append', index=False)