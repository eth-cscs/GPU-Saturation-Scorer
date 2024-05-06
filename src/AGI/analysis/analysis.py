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

    Methods:
    - __init__(self, db_file: str): Constructor method.
    - plotUsageMap(self): Plot a heatmap of the GPU usage.
    - plotTimeSeries(self): Plot the time series of the GPU metrics.
    - summary(self, verbosity: str = "medium"): Print a summary of the GPU metrics.
    - showMetadata(self): Print the metadata of the jobs and processes.

    Notes:
    - This class is intended to provide some quick and easy to use options for basic data analysis and visualization.
    - This is not a full-fledged data analysis tool and as such only the default profiling metrics are supported.
    - When possible, data manipulation should be done via SQL queries for performance and readability.
    """
    def __init__(self, db_file: str):
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

        # Read data from database
        self.db = SQLIO(self.db_file, read_only=True)

        # Create necessary objects
        self.grapher = Grapher()

    def get_prefix(self, maxval: float):
        """
        Description:
        This method determines the appropriate unit prefix.

        Parameters:
        - None

        Returns:
        - unit (str): The unit prefix.
        - scale (float): The scaling factor.
        """

        # Determine the appropriate unit
        if maxval > 1e9:
            unit = "G"
            scale = 1e9
        elif maxval > 1e6:
            unit = "M"
            scale = 1e6
        elif maxval > 1e3:
            unit = "K"
            scale = 1e3
        else:
            unit = ""
            scale = 1

        return unit, scale

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
            sampling_time = job["sampling_time"]
            
            # Aggregate data over all processes and all GPUs
            # Use sample_id, time to group data by time and sample
            # as we dont want to deal with floating point time values
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
            
            # Get label for the job as job id + label
            label = job['label']

            # Remap GPU metrics that are in integer percentages to floats
            data["gpu_utilization"] = data["gpu_utilization"] / 100.
            data["time"] = data["sample_index"] * sampling_time

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
            unit, scale = self.get_prefix(maxval)
            unit += "B/s" # Add the unit for memory activity

            # Get the memory activity data
            df_memory = data[["time"] + memory_activity_metrics].copy()
            
            # Scake data uf needed
            if scale:
                df_memory[memory_activity_metrics] = df_memory[memory_activity_metrics] / scale

            self.grapher.plot_time_series(df_memory,
                                            f"{label}_memory_activity.pdf",
                                            f"{label} Memory Activity ({unit})"
                                        )
            
    def report(self):
        # Get metadata for each job
        metadata = self.db.get_table("job_metadata")

        print_title("Summary of Metrics:")

        for _, job in metadata.iterrows(): # Note: iterrows is slow, but we only expect very few rows
            
            ### Print global summary
            print_title(f"Job ID: {job['job_id']} - {job['label']}", color="red")
            
            # Get the raw data for the job
            data = self.db.query(f"SELECT {job['metrics']} FROM data WHERE job_id={job['job_id']}")
            
            # Aggregate data
            agg = format_df(data.agg(['median', 'mean', 'min', 'max'])).T # Transpose to get metrics as rows
            print_summary(job, agg)

            ### Print average data transfered
            print_title("Transfered data:", color="red")
            
            # Riemann integral to compute the total data transfered
            PCIE_transferred_avg = ((data["pcie_tx_bytes"] + data["pcie_rx_bytes"]) * job["sampling_time"]).sum()
            NVLink_transferred_avg = ((data["nvlink_tx_bytes"] + data["nvlink_rx_bytes"]) * job["sampling_time"]).sum()
            PCIE_transferred_total = PCIE_transferred_avg * job["n_gpus"]
            NVLink_transferred_total = NVLink_transferred_avg * job["n_gpus"]

            # Set up small matrix for tabular output
            transfer_data = []
            unit, scale = self.get_prefix(PCIE_transferred_avg)
            transfer_data.append([f"Average data transfered over PCIe (per GPU)", f"{PCIE_transferred_avg/scale:.2f} {unit}B"])
            unit, scale = self.get_prefix(NVLink_transferred_avg)
            transfer_data.append([f"Average data transfered over NVLink (per GPU)", f"{NVLink_transferred_avg/scale:.2f} {unit}B"])
            unit, scale = self.get_prefix(PCIE_transferred_total)
            transfer_data.append([f"Total PCIe data transfered", f"{PCIE_transferred_total/scale:.2f} {unit}B"])
            unit, scale = self.get_prefix(NVLink_transferred_total)
            transfer_data.append([f"Total NVLink data transfered", f"{NVLink_transferred_total/scale:.2f} {unit}B"])
            
            # Print the data transfered using tabulate for better formatting
            print(tabulate(transfer_data, tablefmt='psql'))
            print() # Add a newline for better readability

            ### Print verbose per-gpu summary
            print_title("GPU averages:", color="red")

            # Query average performance metrics per each GPU
            data = self.db.query(f"""
                                SELECT
                                    proc_id,gpu_id,
                                    AVG(gpu_utilization) AS gpu_utilization,
                                    AVG(sm_active) AS sm_active,
                                    AVG(tensor_active + fp16_active + fp32_active + fp64_active) AS total_flop_activity
                                FROM
                                    data
                                WHERE
                                    job_id={job['job_id']}
                                GROUP BY
                                    proc_id, gpu_id
                                ORDER BY
                                    proc_id, gpu_id ASC
                                """)
            
            # Format metrics correctly before printing
            m = ["gpu_utilization", "sm_active", "total_flop_activity"]
            data[m] = format_df(data[m])
            print_df(data)

    # This function shows the metadata of the job and process
    def show_metadata(self):
        # Print Job Metadata
        print_title("Job Metadata:")
        data = self.db.get_table("job_metadata")
        # Trim problematic columns
        data[['hostnames', 'metrics']] = trim_df(data[['hostnames', 'metrics']].copy())
        print_df(data.T, show_index=True)

        # Print Process Metadata
        print_title("Process Metadata:")
        data = self.db.get_table("process_metadata")
        print_df(data)

        # Print GPU Metrics
        print_title("Job Metrics:")
        data = self.db.query("SELECT job_id, metrics, label FROM job_metadata")
        print_metrics(data)