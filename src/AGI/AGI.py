import os
# Driver functions for the profiler module
class AGI:
    def __init__(self, args):
        self.args = args

    def run(self):
        if self.args.subcommand == 'profile':
            self.profile() 
        elif self.args.subcommand == 'analyze':
            self.analyze()
            
    def profile(self):
        from .utils.utils import checkDCGMImports
        from .io.MetricsDataIO import MetricsDataIO
        
        # Check if DCGM bindings are available before importing AGI modules
        # This raises an exception if DCGM is not available
        checkDCGMImports()
        # Import AGI modules
        from .profiler.GPUMetricsProfiler import GPUMetricsProfiler
        
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
        if self.args.output_file is None:
            self.args.output_file = f"AGI_{jobId}.sqlite"
            
        # Create profiler object
        profiler = GPUMetricsProfiler(
            gpuIds=gpuIds, # Need to pass a list of GPU IDs
            samplingTime=self.args.sampling_time,
            label=self.args.label,
            maxRuntime=self.args.max_runtime,
        )

        # Dry run so that we can run a workload through AGI without collecting any metrics
        # This should be used for debugging purposes and to test the AGI setup
        if self.args.dry_run:
            profiler.dryRun(self.args.wrap)
            del profiler
            return

        # Run workload
        profiler.run(self.args.wrap)

        # Get collected metrics
        metadata, data = profiler.getCollectedData()

        # We force delete the profiler object to ensure that the DCGM context is destroyed
        # We do not want to keep the DCGM context open for the entire lifetime of the profiler
        # Doing so results in strange and unpredictable crashes
        del profiler

        # Create IO handler
        ifExists = "fail"
        if self.args.append:
            ifExists = "append"
        elif self.args.force_overwrite:
            ifExists = "overwrite"

        IO = MetricsDataIO(self.args.output_file, readOnly=False, ifExists=ifExists)

        # Dump data to SQL DB
        IO.dump(metadata, data)
        
        return 0

    # Driver function for the analyze module
    def analyze(self):
        from .analysis.analysis import GPUMetricsAnalyzer
        
        # Instantiate analyzer class
        analyzer = GPUMetricsAnalyzer(
            inputFile=self.args.input_file,
            detectOutliers=self.args.detect_outliers
            )

        # Print GPU information
        if self.args.show_metadata:
            analyzer.showMetadata()

        # Print summary of metricsx
        if self.args.no_summary == False:
            analyzer.summary(self.args.verbosity)
        
        # Plot time-series of metrics
        if self.args.plot_time_series:
            analyzer.plotTimeSeries()

        # Plot load-balancing of metrics
        if self.args.plot_load_balancing:
            #print("Usage map plotting is temporarily disabled due to restructuring of the code.")
            analyzer.plotUsageMap()
        
        return 0
