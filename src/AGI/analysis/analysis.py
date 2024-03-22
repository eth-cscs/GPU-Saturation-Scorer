# External imports
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.special import expit as sigmoid  # Sigmoid function

# AGI imports
from AGI.io.MetricsDataIO import MetricsDataIO
from AGI.io.format import formatDataFrame
from AGI.io.GraphIO import GraphIO
from AGI.analysis.preprocessing import MetricsPreProcessor
from AGI.analysis.aggregation import GPUMetricsAggregator

class GPUMetricsAnalyzer:
    def __init__(self, inputFile: str, detectOutliers:str = "leading", detectionAlgorithm:str = "CPD", verbose: bool = False):
        # Set up input variables
        self.input_file = inputFile
        self.verbose = verbose
        self.detectOutliers = detectOutliers
        self.detectionAlgorithm = detectionAlgorithm

        # Load data from file
        io = MetricsDataIO(inputFile, readOnly = True)
        self.metadata, self.data = io.load()

        # Create necessary objects
        #self.pp = MetricsPreProcessor(self.data)
        self.aggregator = GPUMetricsAggregator(self.metadata, self.data)
        self.plotter = GraphIO()

        # Call pre-processing functions
        if self.detectOutliers != "none":
            print("Automatic outlier detection is currently disabled while the code is being restructured.")
            # self.pp.removeOutliers(self.detectOutliers, self.detectionAlgorithm)

    def plotUsageMap(self):
        # For load balancing heatmap we aggregate over the time dimension.
        # This will yield a single average value for each metric for each GPU.
        data = self.aggregator.aggregateTime()
        self.plotter.plotUsageMaps(data)
         
    def plotTimeSeries(self):
        # For time series we aggregate over the space dimension.
        # This will yield an average time-series for each metric over all GPUs.
        data = self.aggregator.aggregateSpace()
        self.plotter.plotTimeSeries(data)
    
    # This function defines the efficiency score of a workload based on the
    # collected metrics. The efficiency score is a measure of how well the
    # GPU resources are being utilized and is modelled as an EOS (equation of state)
    # The coefficients were determined by fitting on a synthetic dataset of GPU metrics.
    def score(self, A, F, O):
        # Parameters for the EOS function
        alpha=24.021653821773356
        beta=3.093540559022677
        gamma=2.0510278298480444
        lambda_=11.44627875155378

        term1 = sigmoid(alpha * A) - 0.5
        term2 = sigmoid(beta * F + gamma * O * np.exp(- lambda_ * F)) - 0.5
        return 4 * term1 * term2
    
    def br(self):
        print("="*73)

    def summary(self, verbosity: str = "medium"):
        # Get unique job ids
        slurmJobIds = self.metadata['slurm_job_id'].unique()
        
        # Set verbosity
        rows = None
        cols = None
        if verbosity == "low":
            rows = ["mean", "min", "max"]
            cols = ["EFFICIENCY_SCORE"]
        if verbosity == "medium":
            rows = ["mean", "median", "min", "max"]
            cols = ["SM_ACTIVE", "SM_OCCUPANCY", "FLOP_ACTIVE", "DRAM_ACTIVE", "EFFICIENCY_SCORE"]
        
        # Create summary for each job
        self.br()
        for jobId in slurmJobIds:
            # Get data for this job
            tnames = self.metadata[self.metadata['slurm_job_id'] == jobId]['tname'].unique()

            # Concatenate all dataframes
            df = pd.concat([self.data[t] for t in tnames], ignore_index=True, axis=0)

            # Compute total FLOP activity
            df["FLOP_ACTIVE"] = df['PIPE_FP16_ACTIVE'] + df['PIPE_FP32_ACTIVE'] + df['PIPE_FP64_ACTIVE'] + df['PIPE_TENSOR_CORE_ACTIVE']
            
            # Compute efficeicy score
            A = df["SM_ACTIVE"]
            O = df["SM_OCCUPANCY"]
            F = df["FLOP_ACTIVE"]

            # Compute efficiency score
            df['EFFICIENCY_SCORE'] = self.score(A, F, O)

            # Compute aggregate metrics for each column
            df = df.agg(['mean', 'median', 'min', 'max'])

            # Print summary
            # Get label
            label = self.metadata[self.metadata['tname'] == tnames[0]]['label'].values[0]
            print(f"Summary of GPU metrics for job {label}_{jobId}")
            
            # Select only columns and cols of interest
            if cols is not None:
                df = df[cols]
            
            if rows is not None:
                df = df.loc[rows]

            print(formatDataFrame(df).T.to_string())
            
            self.br()

    def showMetadata(self):
        print("Metadata")
        print(self.metadata)

        print("Data")
        print(self.data.keys())
