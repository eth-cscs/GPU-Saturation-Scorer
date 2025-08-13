###############################################################
# Project: GPU Saturation Scorer
#
# File Name: GraphIO.py
#
# Description:
# This file contains the implementation of the Grapher class, which
# contains basic plotting functions for the collected metrics.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil, sqrt
import seaborn as sns

class Grapher:
    """
    Description:
    This class contains basic plotting functions for the collected metrics.

    Attributes:
    - None

    Methods:
    - br(self, s, n=15): Break string into multiple lines to avoid overlapping.
    - plotMetricUsageMap(self, data, fname, title): Plot a heatmap of the GPU usage.

    """
    # def br(self, s, n=15):
    #     # Break string into multiple lines to avoid overlapping
    #     return "\n".join([s[i:i+n] for i in range(0, len(s), n)])

    # # Plot usage map
    # def plotMetricUsageMap(self, data, fname, title):
    #     # Ensure data is 1d numpy array
    #     values = data.to_numpy().flatten()

    #     # Read labels as numpy array of strings from index of dataframe
    #     labels = data.index.to_numpy().astype(str)
    #     labels = np.array(
    #         [f"{self.br(label)}\n{round(value, 2)}" for value, label in zip(values, labels)])

    #     # We want to plot a square heatmap so we need to compute the size of the square
    #     n = ceil(sqrt(len(values)))

    #     # Pad with NaNs to make it square
    #     values = np.pad(values, (0, n**2 - len(values)),
    #                     mode='constant', constant_values=np.nan)
    #     labels = np.pad(labels, (0, n**2 - len(labels)),
    #                     mode='constant', constant_values='x')

    #     # Reshape values into a square matrix
    #     values = values.reshape(n, n)
    #     labels = labels.reshape(n, n)

    #     fig, ax = plt.subplots()

    #     # Create heatmap
    #     ax = sns.heatmap(values,
    #                      cmap="RdBu",
    #                      annot=labels,
    #                      fmt="s",
    #                      clip_on=False,
    #                      linecolor='black',
    #                      linewidth=1.5,
    #                      ax=ax,
    #                      vmin=0,
    #                      vmax=1)

    #     # Set title
    #     ax.set_title(title)

    #     # Remove ticks
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    #     # Save figure
    #     plt.tight_layout()
    #     plt.savefig(fname, format='pdf')

    # # Plot usage map
    # def plotUsageMaps(self, data, metrics=all_metrics):
    #     # Plot data for each job
    #     for tname, df in data.items():
    #         # Plot usage maps for each metric
    #         for metric in (gpu_activity_metrics + flop_activity_metrics):
    #             self.plotMetricUsageMap(df[[metric]],
    #                                     f"{tname}_{metric}_load_balancing_{self.tstamp}.pdf",
    #                                     f"{tname} {metric} Load Balancing")

    # Function that implements time-series plotting of the metrics.
    def plot_time_series(self, df, fname, title, ymax=None, ymin = 0.0):
        # Set ticks theme
        # sns.set_theme("whitegrid")

        # Downsample the DataFrame to avoid cluttering the plot if necessary.
        #df = self.downsample_df(df, 1000)

        # Create a long-form DataFrame for Seaborn's lineplot function.
        # This involves melting the DataFrame so each row represents a single observation for a given variable.
        df_long = df.melt('time', var_name='Columns', value_name='Values')
        
        # Initialize the matplotlib figure for size, if desired.
        plt.figure(figsize=(10, 6))

        # Create a line plot with Seaborn. x='index' for the x-axis, and y='Values' for the y-axis.
        # The 'hue' parameter categorizes data points by the 'Columns' to differentiate lines.
        ax = sns.lineplot(data=df_long, x='time', y='Values', hue='Columns')

        # Set the plot title and labels
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')

        # If a maximum y-axis value is specified, adjust the y-axis limit.
        if ymax is not None:
            plt.ylim(ymin, ymax)

        # Adjust the legend to not distort plot, if necessary.
        ax.legend(title='Metrics', bbox_to_anchor=(1, 1), loc='upper left')
        plt.grid(0.8)

        # Seaborn handles tight_layout internally, but we should call since we've manually adjusted the plot.
        plt.tight_layout()

        # Save the plot to a file.
        plt.savefig(fname, format='pdf')

    
    def downsample_df(self, x: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Description:
        This function downsamples a pandas DataFrame by segmenting the rows
        into groups of n and averaging the values in each group.

        Parameters:
        - x: The pandas DataFrame to downsample.
        - n: The number of rows to skip.

        Returns:
        - The downsampled pandas DataFrame.
        """
        if len(x) < n:
            return x
        else:
             return x.groupby(np.arange(len(x))//n).mean()
