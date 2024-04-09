import os

# Driver functions for the profiler module


class AGI:
    def __init__(self, args):
        self.args = args

    def run(self) -> None:
        if self.args.subcommand == 'profile':
            self.profile()
        elif self.args.subcommand == 'analyze':
            self.analyze()

    def profile(self) -> None:
        from .utils.dcgm import load_dcgm
        from .io.metrics_data_io import MetricsDataIO
        from .utils.slurm_handler import SlurmJob

        # Check if DCGM bindings are available before importing AGI modules
        load_dcgm()

        # Import AGI modules
        from .profiler.gpu_metrics_profiler import GPUMetricsProfiler

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
            force_overwrite=self.args.force_overwrite
        )

        # Run workload
        profiler.run(self.args.wrap)

    # Driver function for the analyze module

    def analyze(self) -> None:
        from .analysis.analysis import GPUMetricsAnalyzer

        # Instantiate analyzer class
        analyzer = GPUMetricsAnalyzer(
            input_file=self.args.input_file,
            detect_outliers=self.args.detect_outliers
        )

        # Print GPU information
        if self.args.show_metadata:
            analyzer.show_metadata()

        # Print summary of metrics
        if not self.args.no_summary:
            analyzer.summary(self.args.verbosity)

        # Plot time-series of metrics
        if self.args.plot_time_series:
            analyzer.plot_time_series()

        # Plot load-balancing of metrics
        if self.args.plot_load_balancing:
            analyzer.plot_usage_map()
