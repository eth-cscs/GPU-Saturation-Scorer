# This class implements the logic to import the output of the profile command
# and convert it to an SQLite database.
import os
import pandas as pd
import sqlite3
import numpy as np

from AGI.io.json_io import JSONDataIO
from AGI.io.binary_io import BinaryDataIO

import datetime

class ExportDataHandler:
    def __init__(self, db_file: str, input_files: list, input_format: str, force_overwrite: bool = False, timeout: int = 900):
        
        # Set up input parameters
        self.db_file = db_file
        self.input_files = input_files
        self.input_format = input_format
        self.force_overwrite = force_overwrite
        self.timeout = timeout

        # Establish connection to database
        self.check_overwrite() # Check if the database file already exists
        self.conn = self.establish_connection()
        self.cursor = self.conn.cursor()
        self.IO_class = JSONDataIO if self.input_format == "json" else BinaryDataIO

    # Check if the database file already exists and throw an exception if it does
    def check_overwrite(self):
        if os.path.exists(self.db_file) and not self.force_overwrite:
            raise Exception(f"Database file {self.db_file} already exists. Use -f flag to overwrite.")

    # Establish connection to database
    def establish_connection(self):
        # Check if the database file already exists and delete it if it does
        # The additional check for self.force_overwrite is redundant, as it is already checked in check_overwrite()
        # but better to be safe than sorry.
        if os.path.exists(self.db_file) and self.force_overwrite: 
            os.remove(self.db_file)

        try:
            return sqlite3.connect(self.db_file, timeout=self.timeout)
        except Exception as e:
            raise Exception(f"Failed to connect to database: {e}")

    # This function reads the input files and converts them to a common format
    def read_files(self) -> list:
        data = []

        # Process each input file
        for file in self.input_files:
            # Initialize IO handler
            # Append tuples of metadata and data to data list
            data.append(self.IO_class(file).load())
        
        return data
            
    # This function creates the "data" table in the database
    # This table contains the actual samples of the metrics
    # Each row in the table corresponds to a sample and has the following columns:
    # job_id: the SLURM job ID of the process
    # proc_id: the process ID (rank) that generated the sample
    # gpu_id: the GPU ID that the sample was taken from
    # sample_index: the index of the sample
    # m1, m2, ...: the values of the metrics
    def create_data_table(self, raw_data: list) -> None:
        # Assert that all inputs have the same columns
        assert all(d[0].keys() == raw_data[0][0].keys() for d in raw_data), "Error: not all input files have the same metrics!"
        
        print("Exporting raw data...", end="")
        # Use Pandas to_sql method instead of executing SQL commands
        # This is more efficient and less error-prone
        for metadata, data in raw_data:
            for gpu_id, metrics in data.items():
                df = pd.DataFrame(metrics)
                df["job_id"] = metadata["job_id"]
                df["proc_id"] = metadata["proc_id"]
                df["gpu_id"] = gpu_id
                df["sample_index"] = np.arange(len(df))

            df.to_sql("data", self.conn, if_exists="append", index=False)

        print("OK.")

    # This function creates the "process_metadata" table in the database
    # Each row in the table contains metadata for a single process (rank)
    # The stored columns are the following:
    # job_id: the SLURM job ID of the process
    # proc_id: the process ID (rank)
    # hostname: the hostname of the node that the process ran on
    # n_gpus: the number of GPUs used by the process
    # gpu_ids: the GPU IDs that the process used
    # start_time: the start time of the process
    # end_time: the end time of the process
    # elapsed: the total elapsed time of the process
    def create_process_metadata_table(self, input_data: list) -> None:
        print("Exporting process metadata...", end="")

        # Use a dictionary to quickly build the per-process metadata
        process_metadata = []
        for metadata, _ in input_data:
                process_metadata.append({
                    "job_id": metadata["job_id"],
                    "proc_id": metadata["proc_id"],
                    "hostname": metadata["hostname"],
                    "n_gpus": metadata["n_gpus"],
                    "gpu_ids": ", ".join([str(gpu_id) for gpu_id in metadata["gpu_ids"]]),
                    "start_time": metadata["start_time"],
                    "end_time": metadata["end_time"],
                    "elapsed": metadata["elapsed"]
                })
        
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(process_metadata)

        # Write the DataFrame to the database
        df.to_sql("process_metadata", self.conn, if_exists="replace", index=False)

        print("OK.")


    # This function creates the "job_metadata" table in the database
    # This table contains metadata for the entire job
    # Each row corresponds to a SLURM job with unique SLURM job ID.
    # For now, we only expect one row in this table, but in the future
    # we may want to merge multiple jobs into a single database file for convenience.
    # The stored columns are the following:
    # job_id: the SLURM job ID
    # label: the label of the job
    # n_hosts: the number of hosts used by the job
    # hostnames: the hostnames used by the job
    # n_procs: the number of processes (ranks) used by the job
    # n_gpus: the number of GPUs used by the job
    # median_start_time: the average start time of the processes
    # median_end_time: the average end time of the processes
    # median_elapsed: the average elapsed time of the processes
    # metrics: text name of the metrics that were collected
    def create_job_metadata_table(self, data: list):
        print("Exporting job metadata...", end="")
        
        job_metadata = [{
            "job_id": data[0][0]["job_id"], # Assume all input files have the same job ID
            "label": data[0][0]["label"], # Assume all input files have the same label
            "n_hosts": len(set(d[0]["hostname"] for d in data)), # Count unique hostnames
            "hostnames": ", ".join(list(set(d[0]["hostname"] for d in data))), # Concatenate unique hostnames
            "n_procs": len(data), # Count the number of processes - one per input file
            "n_gpus": sum(d[0]["n_gpus"] for d in data), # Sum the number of GPUs used by each process
            "avg_start_time": np.median([d[0]["start_time"] for d in data]), # Compute the average start time (median for robustness
            "avg_end_time": np.median([d[0]["end_time"] for d in data]), # Compute the average end time
            "avg_elapsed": np.median([d[0]["elapsed"] for d in data]), # Compute the average elapsed time
            "metrics": ", ".join(list(data[0][1][0].keys())), # Assume all input files have the same metrics
            "cmd": data[0][0]["cmd"] # Assume all input run the same command
        }]

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(job_metadata)

        # Write the DataFrame to the database
        df.to_sql("job_metadata", self.conn, if_exists="replace", index=False)

        print("OK.")


    # This function exports the input data to database
    def export_db(self):
        # Process each input file 
        data = self.read_files()

        # Assert that all input files have the same SLURM job ID
        assert all(d[0]["job_id"] == data[0][0]["job_id"] for d in data), "Error: not all input files have been generated by the same SLURM job!"

        # Create the tables
        self.create_data_table(data) # Create the data table
        
        self.create_process_metadata_table(data) # Create the process_metadata table
        
        self.create_job_metadata_table(data) # Create the job_metadata table
