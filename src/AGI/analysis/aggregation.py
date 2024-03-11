# External imports
import numpy as np
import pandas as pd

# AGI imports
from AGI.io.format import formatDataFrame

class GPUMetricsAggregator:
    def __init__(self, metadata, data):
        self.metadata = metadata
        self.data = data
        self.timeAggregate = None
        self.spaceAggregate = None

    def aggregateTime(self) -> pd.DataFrame:
        # If space aggregation has already been computed, return it
        if self.timeAggregate is not None:
            return self.timeAggregate
       
        # Init self.timeAggregate
        self.timeAggregate = {}

        # Get table names for each slurm_job_id
        slurmJobIds = self.metadata['slurm_job_id'].unique()
        
        # For each job, aggregate the metrics over time
        for jobId in slurmJobIds:
            tnames = self.metadata[self.metadata['slurm_job_id'] == jobId]['tname'].unique()
            hosts = self.metadata[self.metadata['slurm_job_id'] == jobId]['host'].unique()

            print(tnames, hosts)

            # For each GPU, compute the time-series average of the metrics
            df = { t: self.data[t].agg(["mean"]) for t in tnames }
            
            # Get label for the job
            label = self.metadata[self.metadata['tname'] == tnames[0]]['label'].values[0]
            # Create key for the timeAggregate dictionary
            key = f"{label}_{jobId}"
            # Store the time-series average of the metrics
            self.timeAggregate[key] = df

        return self.timeAggregate

    # Function that implements space (GPU) aggregation of GPU metrics in
    # order to check the average time-series of the metrics over all GPUs.
    # Used as input for the plotTimeSeries function.
    def aggregateSpace(self) -> pd.DataFrame:
        # If time aggregation has already been computed, return it
        if self.spaceAggregate is not None:
            return self.spaceAggregate
        
        # Init self.timeAggregate
        self.spaceAggregate = {}

        # Get table names for each slurm_job_id
        slurmJobIds = self.metadata['slurm_job_id'].unique()
        
        # For each job, aggregate the metrics over time
        for jobId in slurmJobIds:
            tnames = self.metadata[self.metadata['slurm_job_id'] == jobId]['tname'].unique()

            # Get length of longest dataframe
            n = max([len(self.data[t]) for t in tnames])

            # Compute average samples over all GPUs
            # Each df is converted to a numpy array and then
            # padded with NaNs to the length of the longest df
            df = np.array([np.pad(self.data[t].to_numpy(), ((0, n - len(self.data[t])), (0, 0)), 
                                mode='constant', constant_values=np.nan) for t in tnames])

            # Compute mean over all GPUs
            # This will ignore NaNs
            df = np.nanmean(df, axis=0)

            # Get label for the job
            label = self.metadata[self.metadata['tname'] == tnames[0]]['label'].values[0]
            # Create key for the timeAggregate dictionary
            key = f"{label}_{jobId}"
            # Convert back to pandas dataframe
            self.spaceAggregate[key] = pd.DataFrame(df, columns=self.data[tnames[0]].columns)

        # Return space aggregate
        return self.spaceAggregate
    
