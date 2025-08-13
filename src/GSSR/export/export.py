###############################################################
# Project: GPU Saturation Scorer
#
# File Name: export.py
#
# Description:
# This file contains the ExportDataHandler class, which is used to
# export JSON/Binary output data to a SQLite database.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################

import os
import pandas as pd
import sqlite3
import numpy as np

from GSSR.io.json_io import JSONDataIO
from GSSR.io.binary_io import BinaryDataIO
from GSSR.io.sql_io import SQLIO

from datetime import datetime

class ExportDataHandler:
    """
    Description:
    This class is used to export JSON/Binary output data to a SQLite database.

    Attributes:
    - db_file (str): Path to the SQLite database file.
    - input_files (list): List of paths to the JSON/Binary input files.
    - input_format (str): Format of the input files (json/binary).
    - force_overwrite (bool): Flag to force overwrite of existing file.
    - timeout (int): Timeout for database connection.

    Methods:
    - read_files(self) -> list: Read the input files and convert them to a common format.
    - create_data_table(self, raw_data: list) -> None: Create the "data" table in the database.
    - create_process_metadata_table(self, input_data: list) -> None: Create the "process_metadata" table in the database.
    - create_job_metadata_table(self, data: list) -> None: Create the "job_metadata" table in the database.
    - export_db(self) -> None: Export the input data to the database.

    Notes:
    - This class uses the JSONDataIO and BinaryDataIO classes to handle input files.
    - This class uses the SQLIO class to handle database input/output.
    """
    def __init__(self, db_file: str, input_format: str, force_overwrite: bool = False, timeout: int = 900):
        """
        Description:
        Constructor method.

        Parameters:
        - db_file (str): Path to the SQLite database file.
        - input_files (list): List of paths to the JSON/Binary input files.
        - input_format (str): Format of the input files (json/binary).
        - force_overwrite (bool): Flag to force overwrite of existing file.
        - timeout (int): Timeout for database connection.

        Returns:
        - None
        """
        # Set up input parameters
        self.db_file = db_file
        self.input_format = input_format
        self.force_overwrite = force_overwrite
        self.timeout = timeout

        # Establish connection to database
        self.db = SQLIO(self.db_file, force_overwrite=self.force_overwrite)
        
        # Set the IO class based on the input format
        self.IO_class = JSONDataIO if self.input_format == "json" else BinaryDataIO

    # This function reads the input files and converts them to a common format
    def read_files(self, input_files) -> list:
        """
        Description:
        This method reads the input files and converts them to a common dict format.

        Parameters:
        - None

        Returns:
        - list: A list of tuples containing metadata and data for each input file.
        """
        data = []

        # Process each input file
        for file in input_files:
            # Initialize IO handler
            # Append tuples of metadata and data to data list
            data.append(self.IO_class(file).load())
        
        return data
            
    def create_data_table(self, raw_data: list) -> None:
        """
        Description:
        This method creates the "data" table in the database.

        Parameters:
        - raw_data (list): A list of tuples containing metadata and data for each input file.

        Returns:
        - None

        Notes:
        - This table contains the actual samples of the metrics.
        - Each row in the table corresponds to a sample and has the following columns:
            - job_id: the SLURM job ID of the process
            - proc_id: the process ID (rank) that generated the sample
            - gpu_id: the GPU ID that the sample was taken from
            - sample_index: the index of the sample
            - m1, m2, ...: the values of the metrics
        """
        # Assert that all inputs have the same columns
        assert all(d[0].keys() == raw_data[0][0].keys() for d in raw_data), "Error: not all input files have the same metrics!"
        
        # Use Pandas to_sql method instead of executing SQL commands
        # This is more efficient and less error-prone
        for metadata, data in raw_data:
            for gpu_id, metrics in data.items():
                df = pd.DataFrame(metrics)
                df["job_id"] = metadata["job_id"]
                df["step_id"] = metadata["step_id"]
                df["proc_id"] = metadata["proc_id"]
                df["gpu_id"] = gpu_id
                df["sample_index"] = np.arange(len(df))
                df["time"] = (df["sample_index"] * metadata["sampling_time"])/1000.0 # Convert from ms to s

            self.db.append_to_table("data", df)


    def create_process_metadata_table(self, input_data: list) -> None:
        """
        Description:
        This method creates the "process_metadata" table in the database.

        Parameters:
        - input_data (list): A list of tuples containing metadata and data for each input file.

        Returns:
        - None

        Notes:
        - This function creates the "process_metadata" table in the database.
        - Each row in the table contains metadata for a single process (rank).
        - The stored columns are the following:
            - job_id: the SLURM job ID of the process
            - proc_id: the process ID (rank)
            - hostname: the hostname of the node that the process ran on
            - n_gpus: the number of GPUs used by the process
            - gpu_ids: the GPU IDs that the process used
            - start_time: the start time of the process
            - end_time: the end time of the process
            - elapsed: the total elapsed time of the process
        """
        # Use a list to store the rows of the table
        process_metadata = []
        for metadata, _ in input_data:
                process_metadata.append({
                    "job_id": metadata["job_id"],
                    "step_id": metadata["step_id"],
                    "proc_id": metadata["proc_id"],
                    "hostname": metadata["hostname"],
                    "n_gpus": metadata["n_gpus"],
                    "gpu_ids": ",".join([str(gpu_id) for gpu_id in metadata["gpu_ids"]]),
                    "start_time": datetime.fromtimestamp(metadata["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": datetime.fromtimestamp(metadata["end_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed": metadata["elapsed"]
                })
        
        # Convert the list to a DataFrame
        df = pd.DataFrame(process_metadata)

        # Write the DataFrame to the database
        self.db.append_to_table("process_metadata", df)

    def create_job_metadata_table(self, data: list):
        """
        Description:
        This method creates the "job_metadata" table in the database.

        Parameters:
        - data (list): A list of tuples containing metadata and data for each input file.

        Returns:
        - None

        Notes:
        - This function creates the "job_metadata" table in the database.
        - Each row in the table contains metadata for the entire job.
        - For now, we only expect one row in this table, but in the future
          we may want to merge multiple jobs into a single database file for convenience.
        - The stored columns are the following:
            - job_id: the SLURM job ID
            - label: the label of the job
            - n_hosts: the number of hosts used by the job
            - hostnames: the hostnames used by the job
            - n_procs: the number of processes (ranks) used by the job
            - n_gpus: the number of GPUs used by the job
            - median_start_time: the average start time of the processes
            - median_end_time: the average end time of the processes
            - median_elapsed: the average elapsed time of the processes
            - metrics: comma-separated list of the metrics that were collected
        """
    
        # Compute start and end times
        median_start_time = datetime.fromtimestamp(np.median([d[0]["start_time"] for d in data])).strftime("%Y-%m-%d %H:%M:%S")
        median_end_time = datetime.fromtimestamp(np.median([d[0]["end_time"] for d in data])).strftime("%Y-%m-%d %H:%M:%S")

        # The data structure here is as follows:
        # data = [
        #     (metadata_proc_0, data_proc_0),
        #     (metadata_proc_1, data_proc_1),
        #     ...
        # ]

        # metadata_proc_x = {
        #     key: value,
        #     key: value,
        #     ...
        # }

        # data_proc_x = {
        #     0: { 
        #         "m1": [v1, v2, ...],
        #         "m2": [v1, v2, ...],
        #         ...
        #     },
        #     1: { 
        #         "m1": [v1, v2, ...],
        #         "m2": [v1, v2, ...],
        #         ...
        #     },
        #     ...
        # }
        # Note: the GPU IDs are not necessarily contiguous

        # data[0][0] -> metadata of the first process
        # data[0][1] -> data of the first process
        # data[0][1][0] -> data of the first GPU on the first process

        root_metadata, root_data = data[0]
        root_gpu = next(iter(root_data.values())) # Do not assume that the first GPU is GPU 0!

        job_metadata = [{
            "job_id": root_metadata["job_id"],                                   # Assume all input files have the same job ID
            "step_id": root_metadata["step_id"],                                 # Assume all input files have the same step ID
            "label": root_metadata["label"],                                     # Assume all input files have the same label
            "n_hosts": len(set(d[0]["hostname"] for d in data)),                 # Count unique hostnames
            "hostnames": ",".join(list(set(d[0]["hostname"] for d in data))),    # Concatenate unique hostnames
            "n_procs": len(data),                                                # Count the number of processes - one per input file
            "n_gpus": sum(d[0]["n_gpus"] for d in data),                         # Sum the number of GPUs used by each process
            "median_start_time": median_start_time,                              # Median start time
            "median_end_time": median_end_time,                                  # Median end time
            "median_elapsed": np.median([d[0]["elapsed"] for d in data]),        # Compute the average elapsed time
            "sampling_time": root_metadata["sampling_time"],                     # Assume all input files have the same "sampling_time
            "metrics": ",".join(list(root_gpu.keys())),                          # Assume all input files have the same metrics
            "cmd": root_metadata["cmd"]                                          # Assume all input run the same command
        }]

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(job_metadata)

        # Write the DataFrame to the database
        self.db.append_to_table("job_metadata", df)


    def export(self, input_files) -> None:
        """
        Description:
        This method exports the input data to the database. 

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - This method is the driver function for the export process.
          It calls the other methods in the class to create the tables in the database.
        - We assume that all input files have been generated by the same SLURM job
          and that all input files have the same metrics.
        - Once this method terminates, the database file will contain the data from the input files.
        """
        # Process each input file 
        data = self.read_files(input_files)

        # Assert that all input files have the same SLURM job ID
        assert all(d[0]["job_id"] == data[0][0]["job_id"] for d in data), "Error: not all input files have been generated by the same SLURM job!"

        # Create the tables
        self.create_data_table(data) # Create the data table
        
        self.create_process_metadata_table(data) # Create the process_metadata table
        
        self.create_job_metadata_table(data) # Create the job_metadata table
