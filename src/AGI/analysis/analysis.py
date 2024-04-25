# External imports
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.special import expit as sigmoid  # Sigmoid function
from tabulate import tabulate

# AGI imports
from AGI.io.sql_io import SQLIO
from AGI.io.format import formatDataFrame
from AGI.io.GraphIO import GraphIO
from AGI.profile.metrics import gpu_activity_metrics, flop_activity_metrics, memory_activity_metrics, all_metrics
# from AGI.analysis.preprocessing import MetricsPreProcessor
# from AGI.analysis.aggregation import GPUMetricsAggregator

# This class implements the high level analysis functions for GPU metrics.
# When possible, data selection is done in the database itself instead of in memory.
# This is done in order to maximize performance, minimize memory usage and improve readability and maintainability of the code.
class GPUMetricsAnalyzer:
    def __init__(self, db_file: str, detect_outliers: str = "leading", detection_algorithm: str = "CPD", verbose: bool = False):
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
        self.plotter = GraphIO()

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
        # Aggregate data over all processes and all GPUs
        metadata = self.db.get_table("job_metadata")

        for _, job in metadata.iterrows():
            metrics = job["metrics"].split(",")
            
            # Aggregate data over all processes and all GPUs
            data = self.db.query(f"""
                                SELECT
                                    sample_index,{','.join([f'AVG({m}) AS {m}' for m in metrics])}
                                FROM
                                    data
                                WHERE
                                    job_id={job['job_id']}
                                GROUP BY
                                    sample_index
                                ORDER BY
                                    sample_index ASC
                                """)

            # Plot time series
            self.plotter.plotTimeSeries(data)

    def summary(self, verbosity: str = "medium"):
        # Get metadata for each job
        metadata = self.db.get_table("job_metadata")

        for _, job in metadata.iterrows(): # Note: iterrows is slow, but we only expect very few rows
            # Get the raw data for the job
            data = self.db.query(f"SELECT {job['metrics']} FROM data WHERE job_id={job['job_id']}")
            
            # Aggregate data
            data = formatDataFrame(data.agg(['median', 'mean', 'min', 'max'])).T # Transpose to get metrics as rows

            # Print summary information
            print(f"Job ID: {job['job_id']}")
            print(f"Label: {job['label']}")
            print(f"Command: \"{job['cmd']}\"")
            print(f"No. hosts: {job['n_hosts']}")
            print(f"No. processes: {job['n_procs']}")
            print(f"No. GPUs: {job['n_gpus']}")
            print(f"Median elapsed time: {job['median_elapsed']}s")
            print(f"Aggregate metric values:")
            print(tabulate(data))
            print()

    # This function shows the metadata of the job and process
    def show_metadata(self):
        # Print Job Metadata
        print("Job Metadata:")
        print(self.db.get_table("job_metadata").to_string(index=False, max_colwidth=20))
        print()

        # Print Process Metadata
        print("Process Metadata:")
        print(self.db.get_table("process_metadata").to_string(index=False, max_colwidth=20))
        print()

        # Print GPU Metrics
        print("GPU Metrics:")
        # Get metrics for each job in the database 
        data = self.db.query("SELECT job_id, metrics, cmd FROM job_metadata")

        # Print metrics for each job
        for job_id, metrics, cmd in data.values:
            print(f"Job ID: {job_id}")
            print(f"Command: \"{cmd}\"")
            print("Collected Metrics:")
            
            for m in metrics.split(","):
                print(f"  {m}")
            
            print()