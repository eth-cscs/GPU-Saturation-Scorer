# External imports
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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

        # Create necessary objects
        self.data = MetricsDataIO(inputFile, readOnly = True).load()
        self.pp = MetricsPreProcessor(self.data)
        self.aggregator = GPUMetricsAggregator(self.data)
        self.plotter = GraphIO(self.data)

        # Call pre-processing functions
        if self.detectOutliers != "none":
            self.pp.removeOutliers(self.detectOutliers, self.detectionAlgorithm)

    def plotUsageMap(self):
        # For load balancing heatmap we aggregate over the time dimension.
        # This will yield a single average value for each metric for each GPU.
        df = self.aggregator.aggregateTime()
        print(df)

        self.plotter.plotUsageMap(df)
         
    def plotTimeSeries(self):
        # For time series we aggregate over the space dimension.
        # This will yield an average time-series for each metric over all GPUs.
        df = self.aggregator.aggregateSpace()
        print(df)
        self.plotter.plotTimeSeries(df)
    
    def summary(self, verbose):
        # If verbosity is enable, print metric for each GPU/EPOCH
        if verbose:
            for gpu in self.data.keys():
                # Compute aggregate data
                df = self.data[gpu].agg(['mean', 'median', 'min', 'max'])

                # Print in human-readable format
                print(f"GPU: {gpu}")
                print("Metrics (average over all time steps)")
                print(formatDataFrame(df).transpose().to_string(), end="\n\n")

        # Concatenate all dataframes
        df = pd.concat(self.data.values(), ignore_index=True, axis=0)

        # Compute aggregate metrics for each column
        df = df.agg(['mean', 'median', 'min', 'max'])
        
        # Print summary
        print("Summary of GPU metrics (average over all GPUs and all time steps)")
        print(formatDataFrame(df).transpose().to_string())

    # Show which GPUs were monitored in the dataset
    def showGPUs(self):
        print("\nAvailable GPUs:")
        for gpu in sorted(list(self.data.keys())):
            print(gpu)
        print()