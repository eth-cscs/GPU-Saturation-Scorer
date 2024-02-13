# Needed for timestamp
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class GraphIO:
    def __init__(self, data) -> None:
        self.data = data

    # Plot time series of a dataframe
    def plotDataFrameTimeSeries(self, df, fname, title, ymax = None):
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

    # Plot time series for different categories of metrics
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
        self.plotDataFrameTimeSeries(df[gpu_activity],
                           f"gpu_activity_{tstamp}.png",
                           "GPU Activity",
                            ymax=1.1
                           )

        # Plot flop activity
        self.plotDataFrameTimeSeries(df[flop_activity],
                            f"flop_activity_{tstamp}.png",
                            "Floating Point Activity",
                             ymax=1.1
                            )

        # Plot memory activity
        self.plotDataFrameTimeSeries(df[memory_activity],
                           f"memory_activity_{tstamp}.png",
                            "Memory Activity"
                           )
    