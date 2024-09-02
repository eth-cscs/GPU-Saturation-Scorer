###############################################################
# Project: Alps GPU Insight
#
# File Name: AGI.py
#
# Description:
# This file implements the AGI class, which is used to drive the
# AGI tool. It contains the main driver functions for the subcommands
# of the AGI tool. The AGI class is responsible for parsing the command
# line arguments and calling the appropriate subcommand.
#
# Authors:
# Marcel Ferrari (CSCS)
#
# Notes:
# Import statements are placed inside the functions to avoid loading
# unnecessary modules when running the tool with a subcommand.
###############################################################

import os
import sys
import argparse

# Needed by all subcommands
from AGI.utils.import_check import check_import_requirements

# Driver functions for the AGI tool
class AGI:
    """
    Description:
    This class is used to drive the AGI tool.
    It contains the main driver functions for the subcommands of the AGI tool. 

    Attributes:
    - args: The parsed command line arguments.
    
    Methods:
    - __init__(self, args): Constructor method.
    - run(self): Run the AGI tool.
    - profile(self): Driver function for the profile subcommand.
    - export(self): Driver function for the export subcommand.
    - analyze(self): Driver function for the analyze subcommand.
    
    Notes:
    - None

    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Description:
        Constructor method.

        Parameters:
        - args: The parsed command line arguments.

        Returns:
        - None

        Notes:
        - None
        """
        self.args = args

    def run(self) -> None:
        """
        Description:
        Run the AGI tool.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - This method calls the appropriate subcommand based on the parsed arguments.
        """
        if self.args.subcommand == 'profile':
            self.profile()
        elif self.args.subcommand == 'analyze':
            self.analyze()
        elif self.args.subcommand == 'export':
            self.export()

    def profile(self) -> None:
        """
        Description:
        Driver function for the profile module.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - This function attempts to import the necessary modules. 
          If something is missing, it will throw an error.
        - We expect this method to be called concurrently by multiple processes.
        """
        from AGI.utils.import_check import load_dcgm

        # Check if all requirements are installed
        check_import_requirements()

        # Check if DCGM bindings are available before importing AGI modules
        load_dcgm()

        # Import AGI modules
        from AGI.utils.slurm_handler import SlurmJob
        from .profile.gpu_metrics_profiler import GPUMetricsProfiler

        # Create SlurmJob object - this will read the Slurm environment
        job = SlurmJob(
            output_folder=self.args.output_folder,
            label=self.args.label
        )

        # Create profiler object
        profiler = GPUMetricsProfiler(
            job=job,
            sampling_time=self.args.sampling_time,
            max_runtime=self.args.max_runtime,
            force_overwrite=self.args.force_overwrite,
            output_format=self.args.format
        )

        # Run workload
        profiler.run(self.args.wrap)

    def export(self) -> None:
        """
        Description:
        Driver function for the export module.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - This function attempts to import the necessary modules. 
          If something is missing, it will throw an error.
        - We expect this method to be called by a single process.
        """
        
        # Check if all requirements are installed
        check_import_requirements()

        # Import AGI modules
        from AGI.export.export import ExportDataHandler

        # Check if input files are provided
        input_files = None

        # Check if input files are provided
        if self.args.input_files:
            input_files = self.args.input_files
        elif self.args.input_folder:
            # Look for all files in the input folder with the specified extension
            ext = "json" if self.args.format == "json" else "bin"
            print(f"Looking for files in {self.args.input_folder} with extension {ext}")
            input_files = [os.path.join(self.args.input_folder, f) for f in os.listdir(self.args.input_folder) if f.endswith(ext)]
        else:
            sys.exit("No input files provided.")
        
        # Check if input files are found
        if len(input_files) == 0:
            sys.exit("No input files found.")

        # Create ExportDataHandler object
        handler = ExportDataHandler(
            db_file=self.args.output if self.args.output else "output.sqlite",
            input_files=input_files,
            input_format=self.args.format,
            force_overwrite=self.args.force_overwrite
        )

        # Export data to database
        handler.export_db()

    # Driver function for the analyze module
    def analyze(self) -> None:
        """
        Description:
        Driver function for the analyze module.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - This function attempts to import the necessary modules. 
          If something is missing, it will throw an error.
        - We expect this method to be called by a single process.
        - This module is meant to generate quick visualizations and summaries of the GPU metrics.
          For more advanced analysis, users should use the exported database and write custom queries.
        """
        # Check if all requirements are installed
        check_import_requirements()

        from AGI.analysis.analysis import GPUMetricsAnalyzer

        # Instantiate analyzer class
        analyzer = GPUMetricsAnalyzer(
            db_file=self.args.input_file
        )

        # Print GPU information
        if self.args.show_metadata:
            analyzer.show_metadata()

        # Print summary of metrics
        if not self.args.no_report:
            analyzer.report()

        # Plot time-series of metrics
        if self.args.plot_time_series:
            analyzer.plot_time_series()

        # Plot load-balancing of metrics
        if self.args.plot_load_balancing:
            analyzer.plot_usage_map()
