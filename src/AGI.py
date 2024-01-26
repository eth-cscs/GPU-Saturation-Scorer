import sys
import os

# Set-up DCGM library path
try: 
    # Check if DCGM is already in the path
    import pydcgm
    import DcgmReader
    import dcgm_fields
    import dcgm_structs
    import pydcgm
    import dcgm_structs
    import dcgm_fields
    import dcgm_agent
    import dcgmvalue

except ImportError:
    # Look for DCGM_HOME variable
    if 'DCGM_HOME' in os.environ:
        dcgm_bingings = os.path.join(os.environ['DCGM_HOME'], 'bindings', 'python3')
    # Look for DCGM_HOME in /usr/local
    elif os.path.exists('/usr/local/dcgm/bindings/python3'):
        dcgm_bindings = '/usr/local/dcgm/bindings/python3'
    # Throw error
    else:
        raise Exception('Unable to find DCGM_HOME. Please set DCGM_HOME environment variable to the location of the DCGM installation.')
    
    sys.path.append(dcgm_bindings)

    # Import DCGM modules
    import pydcgm
    import DcgmReader
    import dcgm_fields
    import dcgm_structs
    import pydcgm
    import dcgm_structs
    import dcgm_fields
    import dcgm_agent
    import dcgmvalue

# Import other modules
import argparse

# Import AGI modules
from AGI.profiler import GPUMetricsProfiler
from AGI.analysis import GPUMetricsAnalyzer

# Driver functions for the profiler modules
def profile(args):
    # Check if job was called via SLURM
    # We need this in order to extract information regarding which GPUs to monitor from the environment
    try:
        jobId = os.environ['SLURM_JOB_ID']
        # Assume exactly one GPU per rank for now
        # It is important to set --gpus-per-task=1 in the SLURM script for this to work
        gpuIds = [int(gpu) for gpu in os.environ['SLURM_STEP_GPUS'].strip().split(',')]
    except KeyError:
        raise Exception('The job must be submitted via SLURM!')
        
    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = f"AGI_{jobId}.sqlite"
        
    # Create profiler object
    profiler = GPUMetricsProfiler(
        command=args.command,
        gpuIds=gpuIds, # Need to pass a list of GPU IDs
        outputFile=args.output_file,
        forceOverwrite=args.force_overwrite,
        samplingTime=args.sampling_time,
        maxRuntime=args.max_runtime,
        verbose=args.verbose
    )

    # Run profiler
    profiler.run()
    
    return 0

def analyze(args):
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

