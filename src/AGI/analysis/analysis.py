# External imports
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# AGI imports
from AGI.io.MetricsDataIO import MetricsDataIO
from AGI.io.format import formatDataFrame

class GPUMetricsAnalyzer:
    def __init__(self, inputFile: str, verbose: bool = False, detectOutliers: str = "leading"):
        self.input_file = inputFile
        self.metrics = None
        self.verbose = verbose
        self.detectOutliers = detectOutliers
        self.data = MetricsDataIO(inputFile, readOnly = True).load()
        
    # Detect outlier points with simple heuristic based on GPU utilization
    # Idea: separate samples into two clusters using KMeans clustering and
    #       drop all samples in the cluster with the lower average GPU utilization.
    #       This should work well for workloads with high GPU utilization (e.g., training a neural network).
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
    
    def detectOutlierSamples(self):
        print("Heuristically detecting outlier samples...", end='\n\n')
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
            if self.detectOutliers in ["leading", "all"]:
                # Cluster samples
                outlierSamples |= self.clusterOutlierSamples(X, y)

            # Detect trailing outlier samples
            if self.detectOutliers in ["trailing", "all"]:
                # Cluster samples
                # The idea is to reverse the order of the samples and cluster them again
                # using the same log feature transformation for the x-values. Then, we reverse
                # the order of the resulting mask to get the original order of the samples.
                # This is a simple way to detect trailing outlier samples without increasing
                # the number of clusters to 3 as this does not seem to work well in practice.
                outlierSamples |= self.clusterOutlierSamples(X, y[::-1])[::-1]

            # Drop all samples in the cluster with lower average GPU utilization)
            self.data[gpu].drop(self.data[gpu].index[outlierSamples], inplace=True)

            if self.verbose:
                print(f"GPU: {gpu}")
                print(f"Number of samples: {len(y)}")
                print(f"Number of samples discarded: {len(y) - len(self.data[gpu])}")
                print(f"Number of samples remaining: {len(self.data[gpu])}")
                print(f"Discarded percentage: {(len(y) - len(self.data[gpu])) / len(y) * 100:.2f}%", end="\n\n")

                # Important to store a copy of the dataframe here
                # Otherwise, the dataframe will be modified in-place
                # and the summary will fail
                df = self.data[gpu].agg(['median', 'min', 'max'])

                # Format df in human-readable format (this modifies df in-place)
                formatDataFrame(df)

                # Print summary
                print("Metrics (average over all time steps)")
                print(df.transpose().to_string(), end="\n\n")

    def summary(self):
        # Detect outlier samples if needed
        if self.detectOutliers != "none":
            self.detectOutlierSamples()

        # Concatenate all dataframes
        df = pd.concat(self.data.values(), ignore_index=True, axis=0)

        # Compute aggregate metrics for each column
        df = df.agg(['mean', 'median', 'min', 'max'])
        
        # Format df in human-readable format (this modifies df in-place)
        formatDataFrame(df)

        # Print summary
        print("Summary of GPU metrics (average over all GPUs and all time steps)")
        print(df.transpose().to_string())