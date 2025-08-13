###############################################################
# Project: GPU saturation scorer
#
# File Name: report.py
#
# Description:
# This file contains the report class that is used to generate
# the PDF report of the analysis. This class includes
# all necessary plotting and formatting functions for the report.
#
# Authors:
# Marcel Ferrari (CSCS)
# Cerlane Leong (CSCS)
#
###############################################################

import os
import sys
import argparse

def main():
    """
    Main function to run the GSS tool.
    It sets up the command line argument parser, imports necessary modules,
    and runs the appropriate subcommand based on the parsed arguments.
    """

    # Find GSS's location and its prefix
    gss_bin = os.path.realpath(os.path.expanduser(__file__))
    gss_prefix = os.path.dirname(os.path.dirname(gss_bin))

    # Allow GSS libs to be imported in our scripts
    gss_lib_path = os.path.join(gss_prefix, "src")
    sys.path.insert(0, gss_lib_path)

    # Import GSS modules
    from GSS.GSS import GSS

    # Main parser
    parser = argparse.ArgumentParser(description='Monitor and analyze resource usage of a workload with GSS')

    # Subparsers
    subparsers = parser.add_subparsers(dest='subcommand', help='sub-command help')

    # Profile subcommand
    parser_profile = subparsers.add_parser('profile', help='Profile command help')
    parser_profile.add_argument('--wrap', '-w', metavar='wrap', type=str, nargs='+', help='Wrapped command to run', required=True)
    parser_profile.add_argument('--label', '-l', metavar='label', type=str, help='Workload label', default='unlabeled')
    parser_profile.add_argument('--max-runtime', '-m', metavar='max-runtime', type=int, default=0, help='Maximum runtime of the wrapped command in seconds')
    parser_profile.add_argument('--sampling-time', '-t', metavar='sampling-time', type=int, default=500, help='Sampling time of GPU metrics in milliseconds')
    parser_profile.add_argument('--force-overwrite', '-f', action='store_true', help='Force overwrite of output file', default=False)
    parser_profile.add_argument('--append', '-a', action='store_true', help='Append profiling data to the output file', default=False)
    parser_profile.add_argument('--output-folder', '-o', metavar='output-folder', type=str, default='profile_out', help='Output folder for the profiling data')

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
    gss_obj = GSS(args)

    if args.subcommand in ['profile', 'export', 'analyze']:
        gss_obj.run()
    else:
        # Print help if no valid subcommand is given
        parser.print_help()

if __name__ == "__main__":
    main()
