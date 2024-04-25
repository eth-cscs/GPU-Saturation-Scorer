##############################################################
# Project: Alps GPU Insight
#
# File Name: format.py
#
# Description:
# This file contains convenience functions to format pandas DataFrames
# to human-readable format.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################

import pandas as pd

def format_DataFrame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
    This function formats a pandas DataFrame to human-readable format.

    Parameters:
    - df: The pandas DataFrame to format.

    Returns:
    - df_out: The formatted pandas DataFrame.

    Notes:
    - The function uses a dictionary to map metric names to formatting functions.
      For metrics not in the dictionary, a generic formatting function is used.
    """
    df_out = pd.DataFrame()
    
    for metric in df.columns:
        # Check if column contains numeric data
        if pd.api.types.is_numeric_dtype(df[metric]):
            format_func = metric_names2formats.get(metric, format_generic)
            df_out[metric] = df[metric].apply(format_func)
        else: # Skip potential non-numeric columns
            df_out[metric] = df[metric]

    return df_out

def format_percent(value: float) -> str:
    """
    Description:
    This function formats a value as a percentage.

    Parameters:
    - value: The value to format.

    Returns:
    - The formatted value as a percentage in string format.
    """
    return f"{value * 100.0:.2f}%"

def format_byte_rate(value: float) -> str:
    """
    Description:
    This function formats a value as a byte rate.

    Parameters:
    - value: The value to format.

    Returns:
    - The formatted value as a byte rate in string format.
    """
    if value < 1024:
        return f"{value} B/s"
    elif value < 1024 ** 2:
        return f"{value / 1024:.2f} KB/s"
    elif value < 1024 ** 3:
        return f"{value / 1024 ** 2:.2f} MB/s"
    else:
        return f"{value / 1024 ** 3:.2f} GB/s"

def format_generic(value: float) -> str:
    """
    Description:
    This function formats a generic value using generic K, M, G suffixes.

    Parameters:
    - value: The value to format.

    Returns:
    - The formatted value in string format.
    """
    if value < 1e3:
        return f"{value:.2f}"
    elif value < 1e6:
        return f"{value / 1e3:.2f} K"
    elif value < 1e9:
        return f"{value / 1e6:.2f} M"
    else:
        return f"{value / 1e9:.2f} G"

# Format utilization metric which is in the range [0, 100]
def format_utilization(value):
    """
    Description:
    This function formats a utilization value expressed as an integer percentage.

    Parameters:
    - value: The value to format.

    Returns:
    - The formatted value as a percentage in string format.

    Notes:
    - The value is divided by 100 to convert it to a percentage. This is
      Necessary as some percentage metrics are expressed as integers.
    """
    return format_percent(value/100.0)

metric_names2formats = {
"gpu_utilization": format_utilization,
"sm_active": format_percent,
"sm_occupancy": format_percent,
"tensor_active": format_percent,
"fp64_active": format_percent,
"fp32_active": format_percent,
"fp16_active": format_percent,
"dram_active": format_percent,
"pcie_tx_bytes": format_byte_rate,
"pcie_rx_bytes": format_byte_rate
}