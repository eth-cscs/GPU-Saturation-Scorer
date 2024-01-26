# External imports
import pandas as pd

# AGI imports
from AGI.profiler.metrics import demangledMetricNames

# Format pandas DataFrames to human-readable format
def formatDataFrame(df: pd.DataFrame) -> None:
    for metric in demangledMetricNames.values():
        if metric in df.columns:
            df[metric] = df[metric].apply(metricNames2Formats[metric])

# Format floating point number to percentage with 2 decimal places
def formatPercent(value):
    return f"{value * 100.0:.2f}%"

# Format byte values to human-readable format
def formatBytes(value):
    if value < 1024:
        return f"{value} B"
    elif value < 1024 ** 2:
        return f"{value / 1024:.2f} KB"
    elif value < 1024 ** 3:
        return f"{value / 1024 ** 2:.2f} MB"
    else:
        return f"{value / 1024 ** 3:.2f} GB"

metricNames2Formats = {
    "DEV_GPU_UTIL": formatPercent,
    "SM_ACTIVE": formatPercent,
    "SM_OCCUPANCY": formatPercent,
    "PIPE_TENSOR_CORE_ACTIVE": formatPercent,
    "PIPE_FP64_ACTIVE": formatPercent,
    "PIPE_FP32_ACTIVE": formatPercent,
    "PIPE_FP16_ACTIVE": formatPercent,
    "DRAM_ACTIVE": formatPercent,
    "PCIE_TX_BYTES": formatBytes,
    "PCIE_RX_BYTES": formatBytes
}