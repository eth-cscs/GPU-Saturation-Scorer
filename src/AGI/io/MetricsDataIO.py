import pandas as pd
import sqlite3
import os
import filelock
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
        
        # Set up process-synchronization objects
        if not readOnly:
            self.lock = filelock.FileLock(f"{self.dbFile}.lock")
            self.metadataFile = f"{self.dbFile}.metadata"
            self.metadata = {}
        
        # Check if output file exists
        self.checkOutfileExists()

    # Define decorator to check if database is read-only
    def checkReadOnly(func):
        def wrapper(self, *args, **kwargs):
            if self.readOnly:
                raise Exception("MetricsDataIO initialiyed in read-only mode!")
            else:
                return func(self, *args, **kwargs)
        return wrapper

    @checkReadOnly
    def dump(self, data: dict):
        # Create database connection
        for tableName, tableData in data.items():
            # Covert tableData to DataFrame
            df = pd.DataFrame(tableData)

            # Convert DEV_GPU_UTIL to percentage instead of integer percentage
            # This is the only percentage metric that is not reported as a float between 0 and 1
            # I don't know why, but it is how Nvidia decided to do it
            if 'DEV_GPU_UTIL' in df.columns:
                df['DEV_GPU_UTIL'] = df['DEV_GPU_UTIL'].astype(float) / 100.0

            # Write data to database
            with sqlite3.connect(self.dbFile, timeout=self.timeout) as conn:
                df.to_sql(tableName, conn, if_exists='replace', index=False)
    
    # Loads all tables into a dictionarz of pandas DataFrames
    def load(self):
        # Create database connection
        with sqlite3.connect(self.dbFile) as conn:
            # Get list of tables
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
            
            # Load tables into a dict of DataFrames
            dfs = {}
            for table in tables['name']:
                dfs[table] = pd.read_sql_query(f"SELECT * FROM \"{table}\"", conn)
            
            return dfs
    
    @checkReadOnly
    def dumpMetadata(self):
        # Write metadata to file
        with self.lock:
            with open(self.metadataFile, 'wb+') as f:
                pickle.dump(self.metadata, f)
    
    @checkReadOnly
    def loadMetadata(self):
        # Check if metadata file exists
        if not os.path.exists(self.metadataFile):
            # Create metadata file
            self.dumpMetadata()

        # Read metadata from file
        with open(self.metadataFile, 'rb') as f:
            self.metadata = pickle.load(f)
    
    def checkOutfileExists(self):
        # For read-only databases, we don't need to check if the output file exists
        # Read operations do not have to be serialized!
        if self.readOnly:
            return
        
        # Check if output file exists
        with self.lock:
            # Read metadata
            self.loadMetadata()

            # Check for errors
            if self.metadata.get('dbError', False):
                raise Exception(self.metadata['dbError'])

            # Check if other process has already created the database
            if not self.metadata.get('dbExists', False):
                # Does the output file exist?
                if os.path.exists(self.dbFile):
                    # Am I allowed to overwrite it?
                    if not self.forceOverwrite:
                        # No, so raise an exception
                        exception = f"Output file {self.dbFile} already exists! Please specify a different output file or set -f flag."
                        
                        # Need to commit the exception to other processes via the metadata file
                        self.metadata['dbError'] = exception
                        self.dumpMetadata()

                        # Raise the exception
                        raise Exception(exception)
                    else:
                        # Yes, so delete the file
                        os.remove(self.dbFile)

                        # Need to tell other processes that the file has been deleted in order
                        # to avoid disk corruption
                        self.metadata['dbExists'] = True
                        self.dumpMetadata()

    # Cleanup function to be called when the program exits
    def cleanup(self):
        # Remove lock file and metadata file
        try:
            # All processes will try to remove the lock file and metadata file
            # Only one process will succeed, so a try/except block is needed
            # to avoid an exception being raised by the other processes
            # This is not a great solution as it is not multi-process safe,
            # but it works for now
            os.remove(self.lock.lock_file)
            os.remove(self.metadataFile)
        except:
            pass

    # Destructor
    def __del__(self):
        if not self.readOnly:
            self.cleanup()