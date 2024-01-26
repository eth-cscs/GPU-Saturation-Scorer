# External imports
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# AGI imports
from AGI.io.MetricsDataIO import MetricsDataIO
from AGI.io.format import formatDataFrame

class GPUMetricsAnalyzer:
    def __init__(self, inputFile: str, verbose: bool = False, detectWarmup: bool = True):
        self.input_file = inputFile
        self.metrics = None
        self.verbose = verbose
        self.detectWarmup = detectWarmup
        self.data = MetricsDataIO(inputFile, readOnly = True).load()
        
    # Detect warmup point with simple heuristic based on GPU utilization
    # Idea: separate samples into two clusters using agglomerative clustering and
    #       drop all samples in the cluster with the lower average GPU utilization.
    #       This should work well for workload with high GPU utilization (e.g., training a neural network).
    def detectWarmupSamples(self):
        print("Heuristically detecting warmup samples...", end='\n\n')
        # Iterate over all GPUs
        for gpu, df in self.data.items():
            # Compute average GPU utilization for each sample
            y = df['DEV_GPU_UTIL'].to_numpy().reshape(-1, 1) # Extract GPU utilization as y-values
            X = np.arange(len(y), dtype=np.float64).reshape(-1, 1) # Create x-values (sample index)

            # Check that X and y have the same length
            assert X.shape == y.shape == (len(y), 1)

            # Compute two clusters
            labels = AgglomerativeClustering(n_clusters=2).fit_predict(X, y)

            mask = labels == 0
            
            # Compute average GPU utilization for each cluster
            avgUtil = np.array([y[mask].mean() , y[~mask].mean()])

            # Get cluster with lower average GPU utilization
            cluster = np.argmin(avgUtil)

            # Drop all samples in the cluster with lower average GPU utilization
            self.data[gpu].drop(self.data[gpu].index[labels == cluster], inplace=True)

            if self.verbose:
                print(f"GPU: {gpu}")
                print(f"Number of samples: {len(y)}")
                print(f"Number of samples discarded: {len(y) - len(self.data[gpu])}")
                print(f"Number of samples remaining: {len(self.data[gpu])}")
                print(f"Discarded percentage: {(len(y) - len(self.data[gpu])) / len(y) * 100:.2f}%", end="\n\n")

                # Important to store a copy of the dataframe here
                # Otherwise, the dataframe will be modified in-place
                # and the summary will fail
                df = self.data[gpu].agg(['mean', 'min', 'max'])

                # Format df in human-readable format (this modifies df in-place)
                formatDataFrame(df)

                # Print summary
                print("Metrics (average over all time steps)")
                print(df.transpose().to_string(), end="\n\n")

    def summary(self):
        # Detect warmup samples if needed
        if self.detectWarmup:
            self.detectWarmupSamples()

        # Concatenate all dataframes
        df = pd.concat(self.data.values(), ignore_index=True, axis=0)

        # Compute aggregate metrics for each column
        df = df.agg(['mean', 'min', 'max'])
        
        # Format df in human-readable format (this modifies df in-place)
        formatDataFrame(df)

        # Print summary
        print("Summary of GPU metrics (average over all GPUs and all time steps)")
        print(df.transpose().to_string())



        

            






