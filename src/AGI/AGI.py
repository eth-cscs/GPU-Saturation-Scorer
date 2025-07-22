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

        # Import AGI modules
        from AGI.export.export import ExportDataHandler

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

        from AGI.analysis.analysis import GPUMetricsAnalyzer

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

    
def main():
    """
    Main function to run the AGI tool.
    It sets up the command line argument parser, imports necessary modules,
    and runs the appropriate subcommand based on the parsed arguments.
    """
    # Main parser
    parser = argparse.ArgumentParser(description='Monitor and analyze resource usage of a workload with AGI')

    # Subparsers
    subparsers = parser.add_subparsers(dest='subcommand', help='sub-command help')

    # Profile subcommand
    parser_profile = subparsers.add_parser('profile', help='Profile command help')
    parser_profile.add_argument('--wrap', '-w', metavar='wrap', type=str, nargs='+', help='Wrapped command to run', required=True)
    parser_profile.add_argument('--label', '-l', metavar='label', type=str, help='Workload label', required=True)
    parser_profile.add_argument('--max-runtime', '-m', metavar='max-runtime', type=int, default=0, help='Maximum runtime of the wrapped command in seconds')
    parser_profile.add_argument('--sampling-time', '-t', metavar='sampling-time', type=int, default=500, help='Sampling time of GPU metrics in milliseconds')
    parser_profile.add_argument('--force-overwrite', '-f', action='store_true', help='Force overwrite of output file', default=False)
    parser_profile.add_argument('--append', '-a', action='store_true', help='Append profiling data to the output file', default=False)
    parser_profile.add_argument('--output-folder', '-o', metavar='output-folder', type=str, default=None, help='Output folder for the profiling data', required=True)

    # Analyze subcommand
    parser_analyze = subparsers.add_parser('analyze', help='Analyze command help')
    parser_analyze.add_argument('--input', '-i', type=str, required=True, help='Input folder or SQL file for analysis')
    parser_analyze.add_argument('--silent', '-s', action="store_true", default=False, help='Silent mode')
    parser_analyze.add_argument('--report', '-rp', action="store_true", default=False, help='Generate full PDF report')
    parser_analyze.add_argument('--export', '-e', metavar='export', type=str, default=":memory:", help='SQLite database file to export the raw data (default: in-memory database)')
    parser_analyze.add_argument('--output', '-o', type=str, required=False, help='Output file for analysis')
    parser_analyze.add_argument('--force-overwrite', '-f', action='store_true', help='Force overwrite of output file', default=False)

    # Parse arguments
    args = parser.parse_args()

    # Run appropriate command
    agi_obj = AGI(args)

    if args.subcommand in ['profile', 'export', 'analyze']:
        agi_obj.run()
    else:
        # Print help if no valid subcommand is given
        parser.print_help()

