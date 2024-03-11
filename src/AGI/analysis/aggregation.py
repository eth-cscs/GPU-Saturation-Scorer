# External imports
import numpy as np
import pandas as pd

# AGI imports
from AGI.io.format import formatDataFrame

class GPUMetricsAggregator:
    def __init__(self, data):
        self.data = data
        self.timeAggregate = None
        self.spaceAggregate = None

    def aggregateTime(self) -> pd.DataFrame:
        # If space aggregation has already been computed, return it
        if self.timeAggregate is not None:
            return self.timeAggregate

        # For each GPU, compute the time-series average of the metrics
        self.timeAggregate = pd.DataFrame({ gpu: df.agg(["mean"]) for gpu, df in self.data.items() })
        
        return self.timeAggregate

    # Function that implements space (GPU) aggregation of GPU metrics in
    # order to check the average time-series of the metrics over all GPUs.
    # Used as input for the plotTimeSeries function.
    def aggregateSpace(self) -> pd.DataFrame:
        # If time aggregation has already been computed, return it
        if self.spaceAggregate is not None:
            return self.spaceAggregate
        
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
        self.spaceAggregate = pd.DataFrame(df, columns=self.data[list(self.data.keys())[0]].columns)

        # Return space aggregate
        return self.spaceAggregate
    
