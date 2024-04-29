###############################################################
# Project: Alps GPU Insight
#
# File Name: analysis.py
#
# Description:
# This file contains the implementation of the high level analysis
# functions for GPU metrics. AGI provides some quick and easy to use
# options for basic data analysis and visualization, however it is
# not intended to be a full-fledged data analysis tool. For more
# advanced analysis, users are encouraged to handle the raw data
# themselves.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################

# External imports
import numpy as np
import pandas as pd


# AGI imports
from AGI.io.sql_io import SQLIO
from AGI.io.format import *
from AGI.analysis.grapher import Grapher
from AGI.profile.metrics import gpu_activity_metrics, flop_activity_metrics, memory_activity_metrics

class GPUMetricsAnalyzer:
    """
    Description:
    This class implements the high level analysis functions for GPU metrics.
    
    Attributes:
    - db_file (str): Path to the SQLite database file.
    - detect_outliers (str): Flag to enable outlier detection.
    - detection_algorithm (str): Algorithm to use for outlier detection.
    - verbose (bool): Flag to enable verbose output.

    Methods:
    - __init__(self, db_file: str, detect_outliers: str = "leading", detection_algorithm: str = "CPD", verbose: bool = False): Constructor method.
    - plotUsageMap(self): Plot a heatmap of the GPU usage.
    - plotTimeSeries(self): Plot the time series of the GPU metrics.
    - summary(self, verbosity: str = "medium"): Print a summary of the GPU metrics.
    - showMetadata(self): Print the metadata of the jobs and processes.

    Notes:
    - This class is intended to provide some quick and easy to use options for basic data analysis and visualization.
    - This is not a full-fledged data analysis tool and as such only the default profiling metrics are supported.
    - When possible, data manipulation should be done via SQL queries for performance and readability.
    """
    def __init__(self, db_file: str, detect_outliers: str = "leading", detection_algorithm: str = "CPD", verbose: bool = False):
        """
        Description:
        Constructor method.

        Parameters:
        - db_file (str): Path to the SQLite database file.
        - detect_outliers (str): Flag to enable outlier detection.
        - detection_algorithm (str): Algorithm to use for outlier detection.

        Returns:
        - None

        Notes:
        - Outlier detection is temporarily disabled.
        """
        # Set up input variables
        self.db_file = db_file
        self.verbose = verbose
        self.detect_outliers = detect_outliers
        self.detection_algorithm = detection_algorithm

        # Read data from database
        self.db = SQLIO(self.db_file, read_only=True)

        # Create necessary objects
        # self.pp = MetricsPreProcessor(self.data)
        # self.aggregator = GPUMetricsAggregator(self.metadata, self.data)
        self.grapher = Grapher()

        # Call pre-processing functions
        # if self.detectOutliers != "none":
        #     self.pp.removeOutliers(self.detectOutliers,
        #                            self.detectionAlgorithm)

    # def plotUsageMap(self):
    #     # For load balancing heatmap we aggregate over the time dimension.
    #     # This will yield a single average value for each metric for each GPU.
    #     data = self.aggregator.aggregateTime()
    #     self.plotter.plotUsageMaps(data)

    def plot_time_series(self):
        """
        Description:
        This method plots time series of the GPU metrics.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - Only the default profiling metrics are supported.
        """
        # Aggregate data over all processes and all GPUs
        metadata = self.db.get_table("job_metadata")

        for _, job in metadata.iterrows():
            metrics = job["metrics"].split(",")
            
            # Aggregate data over all processes and all GPUs
            data = self.db.query(f"""
                                SELECT
                                    time,{','.join([f'AVG({m}) AS {m}' for m in metrics])}
                                FROM
                                    data
                                WHERE
                                    job_id={job['job_id']}
                                GROUP BY
                                    time
                                ORDER BY
                                    time ASC
                                """)
            
            # Get label for the job as job id + label
            label = f"{job['job_id']}_{job['label']}"

            # Remap GPU metrics that are in integer percentages to floats
            data["gpu_utilization"] = data["gpu_utilization"] / 100.
            data["mem_copy_utilization"] = data["mem_copy_utilization"] / 100.

            # Plot GPU activity
            self.grapher.plot_time_series(data[["time"] + gpu_activity_metrics],
                                            f"{label}_gpu_activity.pdf",
                                            f"{label} GPU Activity",
                                            ymax=1.1
                                            )

            # Plot flop activity
            self.grapher.plot_time_series(data[["time"] + flop_activity_metrics],
                                            f"{label}_flop_activity.pdf",
                                            f"{label} Floating Point Activity",
                                            ymax=1.1
                                            )

            # Plot memory activity - this is a bit more complex as we need to 
            # adjust the unit of the data
            
            # Get the maximum value in the dataframe
            maxval = data[memory_activity_metrics].max(numeric_only=True).max()

            # Determine the appropriate unit
            if maxval > 1e9:
                scale = 1e9
                unit = "GB/s"
            elif maxval > 1e6:
                scale = 1e6
                unit = "MB/s"
            elif maxval > 1e3:
                scale = 1e3
                unit = "KB/s"
            else:
                scale = None
                unit = "B/s"

            # Get the memory activity data
            df_memory = data[["time"] + memory_activity_metrics].copy()
            
            # Scake data uf needed
            if scale:
                df_memory[memory_activity_metrics] = df_memory[memory_activity_metrics] / scale

            self.grapher.plot_time_series(df_memory,
                                            f"{label}_memory_activity.pdf",
                                            f"{label} Memory Activity ({unit})"
                                        )
            
    def summary(self, verbosity: str = "medium"):
        # Get metadata for each job
        metadata = self.db.get_table("job_metadata")

        print_title("Summary of Metrics:")
        for _, job in metadata.iterrows(): # Note: iterrows is slow, but we only expect very few rows
            # Get the raw data for the job
            data = self.db.query(f"SELECT {job['metrics']} FROM data WHERE job_id={job['job_id']}")
            
            # Aggregate data
            data = format_df(data.agg(['median', 'mean', 'min', 'max'])).T # Transpose to get metrics as rows

            print_summary(job, data)

    # This function shows the metadata of the job and process
    def show_metadata(self):
        # Print Job Metadata
        print_title("Job Metadata:")
        data = self.db.get_table("job_metadata")
        # Trim problematic columns
        data[['hostnames', 'metrics']] = trim_df(data[['hostnames', 'metrics']].copy())
        print_df(data)

        # Print Process Metadata
        print_title("Process Metadata:")
        data = self.db.get_table("process_metadata")
        print_df(data)

        # Print GPU Metrics
        print_title("Job Metrics:")
        data = self.db.query("SELECT job_id, metrics, label FROM job_metadata")
        print_metrics(data)