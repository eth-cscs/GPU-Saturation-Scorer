import pandas as pd
import sqlite3
import os
import pickle

# This class is used to write data to a SQLite database
# The purpose is to allow multiple processes to write to the same database without corrupting it


class MetricsDataIO:
    # default 15 minutes timeout for locking the database
    def __init__(self, dbFile: str, ifExists: str = "fail", readOnly: bool = True, timeout: int = 900):
        # Set up input parameters
        self.dbFile = dbFile
        self.ifExists = ifExists
        self.timeout = timeout
        self.readOnly = readOnly

    # Define decorator to check if database is read-only
    def checkReadOnly(func):
        def wrapper(self, *args, **kwargs):
            if self.readOnly:
                raise Exception("MetricsDataIO initialized in read-only mode!")
            else:
                return func(self, *args, **kwargs)
        return wrapper

    @checkReadOnly
    def dump(self, metadata: dict, data: dict) -> None:
        # Create database connection --> this automatically handles the locking
        with sqlite3.connect(self.dbFile, timeout=self.timeout) as conn:
            # Read largest slurm job id
            max_slurm_job_id = None

            # Need to handle the case where the database is empty
            try:
                max_slurm_job_id = int(pd.read_sql_query(
                    "SELECT MAX(slurm_job_id) FROM AGI_METADATA", conn).iloc[0, 0])
            except:
                pass

            # Check if the database already contains data from a previous slurm job
            if max_slurm_job_id and int(metadata['slurm_job_id']) != max_slurm_job_id:
                # Check if we want to append data to the DB
                if self.ifExists == "append":
                    pass

                # Check if we want to overwrite the DB
                elif self.ifExists == "overwrite":
                    tables = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table'", conn)
                    for table in tables['name']:
                        conn.execute(f"DROP TABLE \"{table}\"")

                # Otherwise, raise an exception
                else:
                    raise Exception(
                        f"Database already exists. Please specify a different output file or set -f flag.")

            for tableName, tableData in data.items():
                # Covert tableData to DataFrame
                df = pd.DataFrame(tableData)

                # Convert DEV_GPU_UTIL to percentage instead of integer percentage
                # This is the only percentage metric that is not reported as a float between 0 and 1
                # I don't know why, but it is how Nvidia decided to do it
                if 'DEV_GPU_UTIL' in df.columns:
                    df['DEV_GPU_UTIL'] = df['DEV_GPU_UTIL'].astype(
                        float) / 100.0

                # Write data to database
                df.to_sql(tableName, conn, if_exists='replace', index=False)

            # Write metadata to the database
            # Create table "AGI_METADATA" if it does not exist
            # The DB schema is as follows:
            # "slurm_job_id": INT,
            # "label": TEXT,
            # "hostname": TEXT,
            # "procid": INT,
            # "n_gpus": INT,
            # "gpu_ids": TEXT,
            # "start_time": INT,
            # "end_time": INT,
            # "duration": INT,
            # "tname": TEXT,
            # "sampling_time": INT,
            # "n_samples": INT

            conn.execute("""
                CREATE TABLE IF NOT EXISTS AGI_METADATA (
                    slurm_job_id INT,
                    label TEXT,
                    hostname TEXT,
                    procid INT,
                    n_gpus INT,
                    gpu_ids TEXT,
                    start_time INT,
                    end_time INT,
                    duration INT,
                    tname TEXT,
                    sampling_time INT,
                    n_samples INT
                )
            """)

            # Append metadata entry to AGI_METADATA
            conn.execute("""
                INSERT INTO AGI_METADATA (
                    slurm_job_id, label, hostname, procid, n_gpus, gpu_ids, start_time, end_time, duration, tname, sampling_time, n_samples
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata['slurm_job_id'], metadata['label'], metadata['hostname'], metadata['procid'],
                metadata['n_gpus'], metadata['gpu_ids'], metadata['start_time'], metadata['end_time'],
                metadata['duration'], metadata['tname'], metadata['sampling_time'], metadata['n_samples']
            ))

    # Loads all tables into a dictionarz of pandas DataFrames

    def load(self):
        # Check if database exists
        if not os.path.exists(self.dbFile):
            raise Exception(f"Database file {self.dbFile} does not exist!")

        data = {}
        # Create database connection
        with sqlite3.connect(self.dbFile) as conn:
            # Get list of tables
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn)

            # Load tables into a dict of DataFrames
            data = {}
            for table in tables['name']:
                data[table] = pd.read_sql_query(
                    f"SELECT * FROM \"{table}\"", conn)

        # Load metadata
        metadata = pd.read_sql_query("SELECT * FROM AGI_METADATA", conn)

        return metadata, data
