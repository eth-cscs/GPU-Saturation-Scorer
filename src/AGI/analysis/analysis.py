# External imports
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Needed for plotting
import matplotlib.pyplot as plt

# Needed for timestamp
from datetime import datetime

# AGI imports
from AGI.io.MetricsDataIO import MetricsDataIO
from AGI.io.format import formatDataFrame

class GPUMetricsAnalyzer:
    def __init__(self, inputFile: str, verbose: bool = False):
        self.input_file = inputFile
        self.metrics = None
        self.verbose = verbose
        self.data = MetricsDataIO(inputFile, readOnly = True).load()
        
    def clusterOutlierSamples(self, X, y):
        # Check that X and y have the same length
        assert X.shape == y.shape == (len(y), 1)

        # Compute two clusters
        labels = KMeans(n_clusters=2).fit_predict(X, y)

        # Get mask for each cluster
        mask = labels == 0
        
        # Compute average GPU utilization for each cluster
        avgUtil = np.array([y[mask].mean() , y[~mask].mean()])

        # Get cluster with lower average GPU utilization
        outlierCluster = np.argmin(avgUtil)
        samplesCluster = np.argmax(avgUtil)

        # Force at least 15% difference in average GPU utilization between clusters, otherwise
        # we consider that there are no outlier samples. This is necessary to avoid false positives
        # when the workload has a low GPU utilization or when few samples are collected.
        if (avgUtil[samplesCluster] - avgUtil[outlierCluster]) / avgUtil[samplesCluster] < 0.15:
            return np.zeros(len(y), dtype=bool)
        
        return labels == outlierCluster
    
    # Detect outlier points with simple heuristic based on GPU utilization
    # Idea: separate samples into two clusters using KMeans clustering and
    #       drop all samples in the cluster with the lower average GPU utilization.
    #       This should work well for workloads with high GPU utilization (e.g., training a neural network).
    def detectOutlierSamples(self, detectionMode):
        # Iterate over all GPUs
        for gpu, df in self.data.items():
            # Compute average GPU utilization for each sample
            y = df['DEV_GPU_UTIL'].to_numpy().reshape(-1, 1) # Extract GPU utilization as y-values
            # Compute x-values (sample index)
            X = np.arange(1., len(y)+1., dtype=np.float64).reshape(-1, 1)
            # We apply a log feature transformation to the x-values to make the clustering more robust
            X = np.log(X)

            # Mark all samples as non-outlier
            outlierSamples = np.zeros(len(y), dtype=bool)

            # Detect leading outlier samples
            if detectionMode in ["leading", "all"]:
                # Cluster samples
                outlierSamples |= self.clusterOutlierSamples(X, y)

            # Detect trailing outlier samples
            if detectionMode in ["trailing", "all"]:
                # Cluster samples
                # The idea is to reverse the order of the samples and cluster them again
                # using the same log feature transformation for the x-values. Then, we reverse
                # the order of the resulting mask to get the original order of the samples.
                # This is a simple way to detect trailing outlier samples without increasing
                # the number of clusters to 3 as this does not seem to work well in practice.
                outlierSamples |= self.clusterOutlierSamples(X, y[::-1])[::-1]

            # Drop all samples in the cluster with lower average GPU utilization)
            self.data[gpu].drop(self.data[gpu].index[outlierSamples], inplace=True)

    # Plot time series of a dataframe
    def plotDataFrame(self, df, fname, title, ymax = None):
        x = np.arange(1, len(df)+1)
        fig, ax = plt.subplots()

        for col in df.columns:
            ax.plot(x, df[col], label=col)
   
        ax.legend(ncols=1, bbox_to_anchor=(1, 1),
                  loc='upper left')
    
        if ymax is not None:
            ax.set_ylim(0, ymax)
        
        ax.grid(0.8)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(fname)

    # Generate time series plot for each metric
    def plotTimeSeries(self):
        # Get length of longest dataframe
        n = max([len(df) for df in self.data.values()])

        # Compute average samples over all GPUs
        # Each df is converted to a numpy array and then
        # padded with NaNs to the length of the longest df
        df = np.array([np.pad(df.to_numpy(), ((0, n - len(df)), (0, 0)), 
                              mode='constant', constant_values=np.nan) for df in self.data.values()])

        # Compute mean over all GPUs
        # This will ignore NaNs
        df = np.nanmean(df, axis=0)

        # Convert back to pandas dataframe
        df = pd.DataFrame(df, columns=self.data[list(self.data.keys())[0]].columns)
        
        # Plot time series for different categories of metrics
        gpu_activity = ["DEV_GPU_UTIL", "SM_ACTIVE", "SM_OCCUPANCY", "DRAM_ACTIVE"]
        flop_activity = ["PIPE_TENSOR_CORE_ACTIVE", "PIPE_FP64_ACTIVE", "PIPE_FP32_ACTIVE", "PIPE_FP16_ACTIVE"]
        memory_activity = ["PCIE_TX_BYTES", "PCIE_RX_BYTES"]
        
        # Get timestamp for unique file names
        tstamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Plot GPU activity
        self.plotDataFrame(df[gpu_activity],
                           f"gpu_activity_{tstamp}.png",
                           "GPU Activity",
                            ymax=1.1
                           )

        # Plot flop activity
        self.plotDataFrame(df[flop_activity],
                            f"flop_activity_{tstamp}.png",
                            "Floating Point Activity",
                             ymax=1.1
                            )

        # Plot memory activity
        self.plotDataFrame(df[memory_activity],
                           f"memory_activity_{tstamp}.png",
                            "Memory Activity"
                           )
    
    def summary(self, verbose):
        # If verbosity is enable, print metric for each GPU/EPOCH
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