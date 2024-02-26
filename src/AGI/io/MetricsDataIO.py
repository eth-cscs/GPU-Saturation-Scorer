import pandas as pd
import sqlite3
import os
import pickle

# This class is used to write data to a SQLite database
# The purpose is to allow multiple processes to write to the same database without corrupting it
class MetricsDataIO:
    def __init__(self, dbFile: str, forceOverwrite: bool = False, readOnly: bool = True, timeout: int = 900):  # default 15 minutes timeout for locking the database
        # Set up input parameters
        self.dbFile = dbFile
        self.forceOverwrite = forceOverwrite
        self.timeout = timeout
        self.readOnly = readOnly

        # Check for potential overwrite of output file
        self.checkOverwrite()

    # Define decorator to check if database is read-only
    def checkReadOnly(func):
        def wrapper(self, *args, **kwargs):
            if self.readOnly:
                raise Exception("MetricsDataIO initialized in read-only mode!")
            else:
                return func(self, *args, **kwargs)
        return wrapper

    @checkReadOnly
    def dump(self, data: list) -> None:
        # Check if data is empty
        if len(data) == 0:
            return
        
        # If only one epoch has been recorded, no suffix is needed
        if len(data) == 1:
            self.dumpEpoch(data[0])
        else:
            # If multiple epochs have been recorded, a suffix is needed
            for i, epoch in enumerate(data):
                self.dumpEpoch(epoch, f"_epoch:{i}")
        
    def dumpEpoch(self, data: dict, epoch : str = "") -> None:
        # Create database connection
        with sqlite3.connect(self.dbFile, timeout=self.timeout) as conn:
            for tableName, tableData in data.items():
                # Covert tableData to DataFrame
                df = pd.DataFrame(tableData)

                # Convert DEV_GPU_UTIL to percentage instead of integer percentage
                # This is the only percentage metric that is not reported as a float between 0 and 1
                # I don't know why, but it is how Nvidia decided to do it
                if 'DEV_GPU_UTIL' in df.columns:
                    df['DEV_GPU_UTIL'] = df['DEV_GPU_UTIL'].astype(float) / 100.0

                # Write data to database
                df.to_sql(tableName + epoch, conn, if_exists='replace', index=False)

    # Loads all tables into a dictionarz of pandas DataFrames
    def load(self):
        # Check if database exists
        if not os.path.exists(self.dbFile):
            raise Exception(f"Database file {self.dbFile} does not exist!")
        
        # Create database connection
        with sqlite3.connect(self.dbFile) as conn:
            # Get list of tables
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
            
            # Load tables into a dict of DataFrames
            dfs = {}
            for table in tables['name']:
                dfs[table] = pd.read_sql_query(f"SELECT * FROM \"{table}\"", conn)
            
            return dfs
    
    def checkOverwrite(self):
        # If the file is read-only, we don't need to check for overwriting
        if self.readOnly or self.forceOverwrite:
            return
        
        # For write operations, we need to check if the output file exists
        # Does the output file already exist?
        if os.path.exists(self.dbFile):
            exception = f"Output file {self.dbFile} already exists! Please specify a different output file or set -f flag."
            raise Exception(exception)
