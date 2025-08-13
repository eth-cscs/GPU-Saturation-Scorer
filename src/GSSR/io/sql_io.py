###############################################################
# Project: GPU saturation scorer
#
# File Name: sql_io.py
#
# Description:
# This file contains the SQLIO class, which is used to
# handle SQL data input/output.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################

from GSS.io.base_io import BaseIO
import os
import sqlite3
import pandas as pd
import sys

class SQLIO(BaseIO):
    """
    Description:
    This class is used to handle SQL data input/output.

    Attributes:
    - file (str): Path to the SQLite database file.
    - force_overwrite (bool): Flag to force overwrite of existing file.
    - read_only (bool): Flag to prevent write operations.
    - timeout (int): Timeout for database connection.

    Methods:
    - establish_connection(self) -> sqlite3.Connection: Establish connection to database.
    - query(self, query: str) -> pd.DataFrame: Query database.
    - get_table(self, tname: str) -> pd.DataFrame: Query full table.
    - create_table(self, tname: str, df: pd.DataFrame, if_exists='fail') -> None: Create table.
    - append_to_table(self, tname: str, df: pd.DataFrame) -> None: Append data to table.

    Notes:
    - This class uses the sqlite3 and pandas modules to interact with the database.
    - The read_only flag can be used to prevent write operations. This is meant 
      to implement defensive programming practices.
    """
    def __init__(self, file, force_overwrite=False, read_only=False, timeout=900):
        """
        Description:
        Constructor method.

        Parameters:
        - file (str): Path to the SQLite database file.
        - force_overwrite (bool): Flag to force overwrite of existing file.
        - read_only (bool): Flag to prevent write operations.
        - timeout (int): Timeout for database connection.

        Returns:
        - None

        Notes:
        - The force_overwrite flag can be used to overwrite the file without checking if it exists.
        - The read_only flag can be used to prevent write operations.
        """
        super().__init__(file, force_overwrite)
        self.read_only = read_only
        self.timeout = timeout
        self.conn = self.establish_connection()

    # Establish connection to database
    def establish_connection(self):
        """
        Description:
        This method establishes a connection to the SQLite database.

        Parameters:
        - None

        Returns:
        - sqlite3.Connection: Connection to the SQLite database.

        Notes:
        - This method will raise an exception if the connection cannot be established.
        """
        # Check if the database file already exists and delete it if it does
        # The additional check for self.force_overwrite is redundant, as it is already checked in check_overwrite()
        # but better to be safe than sorry.
        if os.path.exists(self.file):
            if not self.read_only:
                # Check if the file should be overwritten
                if self.force_overwrite: 
                    os.remove(self.file)
                else:
                    sys.exit(f"Error: File {self.file} already exists. Use --force to force overwrite.")
        else:
            if self.read_only:
                sys.exit(f"Error: File {self.file} does not found.")

        try:
            return sqlite3.connect(self.file, timeout=self.timeout)
        except Exception as e:
            sys.exit(f"Failed to connect to database: {e}")
    

    # Convenience function to execute SQL queries
    # Query database
    def query(self, query: str) -> pd.DataFrame:
        """
        Description:
        This method queries the database and returns the result as a pandas DataFrame.

        Parameters:
        - query (str): SQL query to execute.

        Returns:
        - pd.DataFrame: Result of the query.

        Notes:
        - This method will raise an exception if the query cannot be executed.
        """
        return pd.read_sql_query(query, self.conn)

    # Query full table
    def get_table(self, tname: str) -> pd.DataFrame:
        """
        Description:
        This method queries the full table and returns the result as a pandas DataFrame.

        Parameters:
        - tname (str): Name of the table to query.

        Returns:
        - pd.DataFrame: Result of the query.

        Notes:
        - This method will raise an exception if the query cannot be executed.
        - This method could be implemented using pd.read_sql_table(), but it is broken
          in some recent versions of pandas.
        """
        # Broken: return pd.read_sql_table(tname, self.con)
        return self.query(f"SELECT * FROM \"{tname}\"")
    
    # Decorator to prevent write operations in read-only mode
    def write_protected(func):
        def wrapper(self, *args, **kwargs):
            if self.read_only:
                raise Exception("Write operations are disabled in read-only mode.")
            return func(self, *args, **kwargs)
        return wrapper
    
    # Create table
    @write_protected
    def create_table(self, tname: str, df: pd.DataFrame, if_exists='fail') -> None:
        """
        Description:
        This method creates a table in the database.

        Parameters:
        - tname (str): Name of the table to create.
        - df (pd.DataFrame): DataFrame containing the data to insert.
        - if_exists (str): Flag to determine behavior if table already exists.

        Returns:
        - None

        Notes:
        - The if_exists flag can be set to 'fail', 'replace', or 'append'.
        """
        df.to_sql(tname, self.conn, if_exists=if_exists, index=False)

    # Append data to table
    @write_protected
    def append_to_table(self, tname: str, df: pd.DataFrame) -> None:
        """
        Description:
        This method appends data to a table in the database.

        Parameters:
        - tname (str): Name of the table to append to.
        - df (pd.DataFrame): DataFrame containing the data to insert.

        Returns:
        - None

        Notes:
        - This method will raise an exception if the data cannot be inserted.
        - The behavior of this method is equivalent to if_exists='append' in the create_table() method.
          This is done to ensure consistency and readability.
        """
        df.to_sql(tname, self.conn, if_exists='append', index=False)
