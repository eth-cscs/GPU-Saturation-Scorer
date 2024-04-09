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
        from .utils.dcgm import loadDCGM
        from .io.MetricsDataIO import MetricsDataIO
        from .utils.slurm_handler import SlurmJob
        
        # Check if DCGM bindings are available before importing AGI modules
        # This raises an exception if DCGM is not available
        loadDCGM()
        
        # Import AGI modules
        from .profiler.GPUMetricsProfiler import GPUMetricsProfiler
        
        # Create SlurmJob object - this will read the Slurm environment
        # and throw an exception if something is wrong/missing/misconfigured
        job = SlurmJob(
            output_folder = self.args.output_folder,
            label = self.args.label
            )
            
        # Create profiler object
        profiler = GPUMetricsProfiler(
            job=job,
            samplingTime=self.args.sampling_time,
            maxRuntime=self.args.max_runtime,
            forceOverwrite=self.args.force_overwrite
        )

        # Run workload
        profiler.run(self.args.wrap)
        
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
