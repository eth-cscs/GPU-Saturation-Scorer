# Import modules
import os
import argparse

# Driver functions for the profiler modules
def profile(args):
    from AGI.utils.checkImports import checkDCGMImports
    from AGI.io.MetricsDataIO import MetricsDataIO
    
    # Check if DCGM bindings are available before importing AGI modules
    # This raises an exception if DCGM is not available
    checkDCGMImports()
    # Import AGI modules
    from AGI.profiler.GPUMetricsProfiler import GPUMetricsProfiler
    
    # Check if job was called via SLURM
    # We need this in order to extract information regarding which GPUs to monitor from the environment
    try:
        jobId = os.environ['SLURM_JOB_ID']
        # Assume exactly one GPU per rank for now
        # It is important to set --gpus-per-task=1 in the SLURM script for this to work
        gpuIds = [int(gpu) for gpu in os.environ['SLURM_STEP_GPUS'].strip().split(',')]
    except KeyError:
        # Hack - if --gpus-per-task=1 is not set, we can still use SLURM_PROCID mod 4 to determine the GPU ID
        print("WARNING: SLURM environment variables not found. Using SLURM_PROCID mod 4 to determine GPU ID.")
        jobId = os.environ['SLURM_JOB_ID']
        procId = int(os.environ['SLURM_PROCID'])
        gpuIds = [procId%4]

    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = f"AGI_{jobId}.sqlite"
        
    # Create profiler object
    profiler = GPUMetricsProfiler(
        gpuIds=gpuIds, # Need to pass a list of GPU IDs
        samplingTime=args.sampling_time,
        maxRuntime=args.max_runtime,
    )

    # Run workload
    profiler.run(args.command)

    profiler.run(args.command)

    # Get collected metrics
    metrics = profiler.getCollectedData()

    # Create IO handler
    IO = MetricsDataIO(args.output_file, readOnly=False, forceOverwrite=args.force_overwrite)

    # Dump data to SQL DB
    IO.dump(metrics)
    
    return 0

def analyze(args):
    from AGI.analysis.analysis import GPUMetricsAnalyzer
    # Your analysis function implementation
    analyzer = GPUMetricsAnalyzer(
                inputFile=args.input_file,
                verbose=args.verbose,
                detectWarmup=args.no_detect_warmup
                )
    
    analyzer.summary()
    return 0

if __name__ == '__main__':

    # Main parser
    parser = argparse.ArgumentParser(description='Monitor and analyze resource usage of a workload with AGI')

    # Subparsers
    subparsers = parser.add_subparsers(dest='subcommand', help='sub-command help')

    # Profile subcommand
    parser_profile = subparsers.add_parser('profile', help='Profile command help')
    parser_profile.add_argument('command', metavar='command', type=str, nargs='+', help='Wrapped command to run')
    parser_profile.add_argument('--max-runtime', metavar='max-runtime', type=int, default=60, help='Maximum runtime of the wrapped command in seconds')
    parser_profile.add_argument('--sampling-time', metavar='sampling-time', type=int, default=1000, help='Sampling time of GPU metrics in milliseconds')
    parser_profile.add_argument('--verbose', action='store_true', help='Print verbose GPU metrics to stdout')
    parser_profile.add_argument('--force-overwrite', action='store_true', help='Force overwrite of output file', default=False)
    parser_profile.add_argument('--output-file', metavar='output-file', type=str, default=None, help='Output SQL file for collected GPU metrics', required=True)

    # Analyze subcommand
    parser_analyze = subparsers.add_parser('analyze', help='Analyze command help')
    parser_analyze.add_argument('--input-file', type=str, required=True, help='Input file for analysis')
    parser_analyze.add_argument('--verbose', action='store_true', help='Print verbose GPU metrics to stdout')
    parser_analyze.add_argument('--no-detect-warmup', action='store_false', help='Disable heuristical detection and removal of samples collected during GPU warmup', default=True)

    # Parse arguments
    args = parser.parse_args()

    # Run appropriate command
    if args.subcommand == 'profile':
        profile(args)  # or a specific function for profiling
    elif args.subcommand == 'analyze':
        analyze(args)  # or a specific function for analysis

