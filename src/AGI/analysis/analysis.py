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

        # Load data from file
        io = MetricsDataIO(inputFile, readOnly = True)
        self.metadata, self.data = io.load()

        # Create necessary objects
        #self.pp = MetricsPreProcessor(self.data)
        self.aggregator = GPUMetricsAggregator(self.metadata, self.data)
        self.plotter = GraphIO()

        # Call pre-processing functions
        if self.detectOutliers != "none":
            self.pp.removeOutliers(self.detectOutliers, self.detectionAlgorithm)

    def plotUsageMap(self):
        # For load balancing heatmap we aggregate over the time dimension.
        # This will yield a single average value for each metric for each GPU.
        data = self.aggregator.aggregateTime()
        self.plotter.plotUsageMap(data)
         
    def plotTimeSeries(self):
        # For time series we aggregate over the space dimension.
        # This will yield an average time-series for each metric over all GPUs.
        data = self.aggregator.aggregateSpace()
        self.plotter.plotTimeSeries(data)
    
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

    def showMetadata(self):
        print("Metadata")
        print(self.metadata)

        print("Data")
        print(self.data.keys())