###############################################################
# Project: GPU saturation scorer
#
# File Name: GSS.py
#
# Description:
# This file implements the GSS class, which is used to drive the
# GSS tool. It contains the main driver functions for the subcommands
# of the GSS tool. The GSS class is responsible for parsing the command
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
from GSS.utils.import_check import check_import_requirements

# Driver functions for the GSS tool
class GSS:
    """
    Description:
    This class is used to drive the GSS tool.
    It contains the main driver functions for the subcommands of the GSS tool. 

    Attributes:
    - args: The parsed command line arguments.
    
    Methods:
    - __init__(self, args): Constructor method.
    - run(self): Run the GSS tool.
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
        Run the GSS tool.

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
        from GSS.utils.import_check import load_dcgm

        # Check if all requirements are installed
        check_import_requirements()

        # Check if DCGM bindings are available before importing GSS modules
        load_dcgm()

        # Import GSS modules
        from GSS.utils.slurm_handler import SlurmJob
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
            output_format="json"
        )

        # Run workload
        profiler.run(self.args.wrap)

    def export(self, in_path, output) -> None:
        """
        Description:
        Driver function for the export module.

        Parameters:
        - None

        Returns:
        - SQLDB object

        Notes:
        - This function attempts to import the necessary modules. 
          If something is missing, it will throw an error.
        - We expect this method to be called by a single process.
        """
        
        # Check if all requirements are installed

        # Import GSS modules
        from GSS.export.export import ExportDataHandler

        # Check that input path is a folder
        if not os.path.isdir(in_path):
            sys.exit("Error: input path is not a folder.")
        
        # Read all subdirectories in the input folder
        input_dirs = os.listdir(in_path)

        # Check no subdirectories are present
        if len(input_dirs) == 0:
            sys.exit("Error: Input folder is empty.")

        handler = ExportDataHandler(
                db_file=output,
                input_format="json",
                force_overwrite=self.args.force_overwrite
            )
        
        written_data = False # Flag to check if any data was written to the database
        for d in input_dirs:
            # Check that the subdirectory is a folder
            if not os.path.isdir(os.path.join(in_path, d)):
                print(f"Warning: {d} is not a folder. Skipping.")
                continue
            
            # Path to the subdirectory corresponding to the SLURM step
            step_path = os.path.join(in_path, d)
            files = os.listdir(step_path)
            
            # Read only files with the specified extension
            files = [os.path.join(step_path, f) for f in files if f.endswith(".json")]

            # Check that there are files to read
            if len(files) == 0:
                print(f"Warning: No JSON files found in {d}. Skipping.")
                continue

            # Export data to database
            handler.export(
                input_files=files
            )

            written_data = True

        if not written_data:
            if self.args.output != ":memory:":
                os.remove(self.args.output)
            sys.exit("Error: No data was written to the database. Removing temporary files and exiting.")
            
        return handler.db
            
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

        from GSS.analysis.analysis import GPUMetricsAnalyzer

        # Check if input_file is a directory
        if os.path.isdir(self.args.input):    
            # Export data to database in memory
            db = self.export(
                in_path=self.args.input,
                output=self.args.export # Default is ":memory:" or the specified output file
            )
        else:
            db = self.args.input
        
        # Instantiate analyzer class
        analyzer = GPUMetricsAnalyzer(
            db_file=db
        )

        # Print summary of metrics
        if not self.args.silent:
            analyzer.summary()

        # Generate PDF report
        if self.args.report:
            analyzer.report()



