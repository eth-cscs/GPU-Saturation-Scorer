import os

# Driver functions for the AGI tool
class AGI:
    def __init__(self, args):
        self.args = args

    def run(self) -> None:
        if self.args.subcommand == 'profile':
            self.profile()
        elif self.args.subcommand == 'analyze':
            self.analyze()
        elif self.args.subcommand == 'export':
            self.export()

    def profile(self) -> None:
        from .utils.dcgm import load_dcgm
        from .utils.slurm_handler import SlurmJob

        # Check if DCGM bindings are available before importing AGI modules
        load_dcgm()

        # Import AGI modules
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
            output_format=self.args.output_format
        )

        # Run workload
        profiler.run(self.args.wrap)

    # Driver function for the export module
    def export(self) -> None:
        from .export.export import ExportDataHandler

        # Check if input files are provided
        input_files = None

        # Check if input files are provided
        if self.args.input_files:
            input_files = self.args.input_files
        elif self.args.input_folder:
            # Look for all files in the input folder with the specified extension
            print(f"Looking for files in {self.args.input_folder} with extension {self.args.input_format}")
            input_files = [os.path.join(self.args.input_folder, f) for f in os.listdir(self.args.input_folder) if f.endswith(self.args.format)]
        else:
            raise Exception("No input files provided.")
        
        # Check if input files are found
        if len(input_files) == 0:
            raise Exception("No input files found.")

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
