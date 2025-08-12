# External imports
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import ruptures as rpt
from math import log


class MetricsPreProcessor:
    def __init__(self, data: dict):
        self.data = data

    # Implements the KMeans cluster-based outlier detection method
    def clusterSamples(self, X, y):
        # Check that X and y have the same length
        assert X.shape == y.shape == (len(y), 1)

        # Compute two clusters
        labels = KMeans(n_clusters=2, n_init='auto').fit_predict(X, y)

        # Get mask for each cluster
        mask = labels == 0

        # Compute average GPU utilization for each cluster
        avgUtil = np.array([y[mask].mean(), y[~mask].mean()])

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
    def removeOuliersKMeans(self, detectionMode):
        # Iterate over all GPUs
        for gpu, df in self.data.items():
            # Compute average GPU utilization for each sample
            # Extract GPU utilization as y-values
            y = df['DEV_GPU_UTIL'].to_numpy().reshape(-1, 1)
            # Compute x-values (sample index)
            X = np.arange(1., len(y)+1., dtype=np.float64).reshape(-1, 1)
            # We apply a log feature transformation to the x-values to make the clustering more robust
            X = np.log(X)

            # Mark all samples as non-outlier
            outlierSamples = np.zeros(len(y), dtype=bool)

            # Detect leading outlier samples
            if detectionMode in ["leading", "all"]:
                # Cluster samples
                outlierSamples |= self.clusterSamples(X, y)

            # Detect trailing outlier samples
            if detectionMode in ["trailing", "all"]:
                # Cluster samples
                # The idea is to reverse the order of the samples and cluster them again
                # using the same log feature transformation for the x-values. Then, we reverse
                # the order of the resulting mask to get the original order of the samples.
                # This is a simple way to detect trailing outlier samples without increasing
                # the number of clusters to 3 as this does not seem to work well in practice.
                outlierSamples |= self.clusterSamples(X, y[::-1])[::-1]

            # Drop all samples in the cluster with lower average GPU utilization)
            self.data[gpu].drop(
                self.data[gpu].index[outlierSamples], inplace=True)

    # Function that implements the change point detection (CPD) method
    def detectBreakPoints(self, y):

        # Use 5% window size
        N = len(y)
        window_size = int(0.05 * N)

        # Use Window based method to detect change points
        algo = rpt.Window(model="l2", width=window_size)

        # Fit model
        algo.fit(y)

        # Predict change points
        # Fix number of change points to 2 for now
        result = algo.predict(n_bkps=2)

        return result

    # Detect outlier points with simple heuristic based on GPU utilization
    # Idea: use change point detection (CPD) to detect breaks in the time series of GPU utilization.
    #       This should be more robust than KMeans and should be the default method for detecting outliers.
    def removeOutliersCPD(self, detectionMode):
        # Iterate over all GPUs
        for gpu, df in self.data.items():
            # Compute average GPU utilization for each sample
            y = df['DEV_GPU_UTIL'].to_numpy().reshape(-1, 1)

            N = len(y)

            # Detect change points
            bkpts = self.detectBreakPoints(y)
            bkpts.pop()  # Remove last change point as it is always the last sample

            # No change points detected
            # Either CPD failed to detect any change points or there are no significant change points in the data.
            if not bkpts:
                return

            if detectionMode in ["leading", "all"]:
                # Drop all samples before the first change point
                i = 0
                while i < len(bkpts) and bkpts[i]/N <= 0.30:
                    i += 1

                # Check if we are dropping more than 30% of the samples
                if i > 0:
                    i -= 1

                bkpt = bkpts[i]

                # Check if we are dropping more than 30% of the samples
                # This is a simple heuristic to check if the break point is leading or trailing.
                if bkpt/N <= 0.30:
                    self.data[gpu].drop(
                        self.data[gpu].index[:bkpt], inplace=True)

            if detectionMode in ["trailing", "all"]:
                # Drop all samples after the last change point
                i = -1
                while i >= -len(bkpts) and 1. - bkpts[i]/N <= 0.30:
                    i -= 1

                # Check if we are dropping more than 30% of the samples
                if i < -1:
                    i += 1

                bkpt = bkpts[i]

                # Drop all samples after the last change point.
                # Check if we are dropping more than 30% of the samples.
                # This is a simple heuristic to check if the break point is leading or trailing.
                if 1. - bkpt/N <= 0.30:
                    self.data[gpu].drop(
                        self.data[gpu].index[bkpt:], inplace=True)

    # Interface function to remove outliers from the data
    def removeOutliers(self, detectionMode, detectionAlgorithm):
        # Detect outliers
        if detectionAlgorithm == "CPD":
            self.removeOutliersCPD(detectionMode)
        elif detectionAlgorithm == "KMeans":
            self.removeOuliersKMeans(detectionMode)
        else:
            raise ValueError(f"Unknown detection method: {detectionAlgorithm}")
