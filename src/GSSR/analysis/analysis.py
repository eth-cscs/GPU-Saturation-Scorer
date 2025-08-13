###############################################################
# Project: GPU Saturation Scorer
#
# File Name: analysis.py
#
# Description:
# This file contains the implementation of the high level analysis
# functions for GPU metrics. GSSR provides some quick and easy to use
# options for basic data analysis and visualization, however it is
# not intended to be a full-fledged data analysis tool. For more
# advanced analysis, users are encouraged to handle the raw data
# themselves.
#
# Authors:
# Marcel Ferrari (CSCS)
# Cerlane Leong (CSCS)
#
###############################################################

# External imports
import numpy as np
import pandas as pd
import fpdf as PDF
import uuid
import os
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm


# GSSR imports
from GSSR.io.sql_io import SQLIO
from GSSR.io.format import *
#from GSSR.analysis.grapher import Grapher
from GSSR.profile.metrics import gpu_activity_metrics, flop_activity_metrics, memory_activity_metrics
from GSSR.analysis.report import PDFReport

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
    def __init__(self, db_file):
        """
        Description:
        Constructor method.

        Parameters:
        - db_file (str or SQLIO): Path to the SQLite database file.
        - detect_outliers (str): Flag to enable outlier detection.
        - detection_algorithm (str): Algorithm to use for outlier detection.

        Returns:
        - None

        Notes:
        - Outlier detection is temporarily disabled.
        """
        # Set up input variables
        self.db_file = db_file

        if isinstance(db_file, str):
            # Read data from file
            self.db = SQLIO(self.db_file, read_only=True)
        elif isinstance(db_file, SQLIO):
            # Read data directly from database
            self.db = db_file
        else:
            raise ValueError("Invalid input type for db_file. \
                              Must be either a string or SQLIO object.")

        # Create necessary objects
        #self.grapher = Grapher()

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


    def clean_tmp(self, remove_dir=False):
        """ Convenience function to clean up the temporary directory """
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    # Generate PDF report
    def report(self):
        print_title("Generating reports for all jobs.")
        # Create temporary directory for storing images
        self.tmp_dir = f"/tmp/{uuid.uuid4()}"
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Read job metadata
        metadata = self.db.get_table("job_metadata")

        # Write the report for each job
        for _, job in metadata.iterrows():
            print_title(job['label'], color="red")    
            output_path = f"{job['label']}_report.pdf"
            report = PDFReport(self.db, job, output_path, self.tmp_dir)
            report.write()
            print_title("[INFO] Generated report: " + f"{job['label']}_report.pdf", color="blue" )

        self.clean_tmp()

    def summary(self):
        # Get metadata for each job
        metadata = self.db.get_table("job_metadata")

        print_title("Summary of Metrics:")

        for _, job in metadata.iterrows(): # Note: iterrows is slow, but we only expect very few rows
            
            ### Print global summary
            print_title(f"Job ID: {job['job_id']} - {job['label']}", color="green")
            
            # Get the raw data for the job
            data = self.db.query(f"SELECT {job['metrics']} FROM data WHERE job_id={job['job_id']} AND step_id={job['step_id']}")
            
            # Aggregate data
            agg = format_df(data.agg(['median', 'mean', 'min', 'max'])).T # Transpose to get metrics as rows
            print_summary(job, agg)

            ### Print average data transfered
            print_title("Transfered data:", color="red")
            
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
                                    job_id={job['job_id']} AND step_id={job['step_id']}
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
