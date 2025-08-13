##############################################################
# Project: GPU saturation scorer
#
# File Name: format.py
#
# Description:
# This file contains convenience functions to format CLI output
# to human-readable format.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################

import pandas as pd
from tabulate import tabulate
from rich import print as rprint

def print_summary(job, data):
    # Print summary information about the job
    # This is done via tabulate to format the output
    metadata = [
    [f"Job ID: {job['job_id']}"],
    [f"Step ID: {job['step_id']}"],
    [f"Label: {job['label']}"],
    [f"Command: \"{job['cmd']}\""],
    [f"No. hosts: {job['n_hosts']}"],
    [f"No. processes: {job['n_procs']}"],
    [f"No. GPUs: {job['n_gpus']}"],
    [f"Median elapsed time: {job['median_elapsed']:.2f}s"],
    ]

    # Print metadata using tabulate
    print(tabulate(metadata, tablefmt='psql', headers=['Job Metadata'], maxcolwidths=80))
    print() # Add a newline for better readability
    
    # Print the metrics for the job
    print_title("Global Summary of Metrics:", color="red")
    print_df(data, show_index=True)  # Show index to display metrics

def print_metrics(data: pd.DataFrame) -> None:
    """
    Description:
    This function prints the metrics for each job to the console.

    Parameters:
    - data: The pandas DataFrame containing the metrics data.

    Returns:
    - None

    Notes:
    - The DataFrame should have the following columns:
      * job_id: The job ID.
      * metrics: The collected metrics.
      * cmd: The command that was executed.
    """

    # Print metrics for each job
    for job_id, metrics, label in data.values:
        # print(f"Job ID: {job_id}")
        # print(f"Command: \"{cmd}\"")
        # print("Collected Metrics:")
        header = (f'Job ID: {job_id}\n'
                  f'Label: {label}\n'
                  'Collected Metrics:')
        
        # This is a workaround to be able to use
        # tabulate with a single column
        m = [[m] for m in metrics.split(",")]

        # Print metrics using tabulate
        print(tabulate(m, tablefmt='psql', headers=[header]))

    print()

def wrap_text(text: str, n: int = 20) -> str:
    """
    Description:
    This function wraps text to a maximum length of n characters.

    Parameters:
    - text: The text to wrap.
    - n: The maximum length to wrap to.

    Returns:
    - The wrapped text.
    """
    return "\n".join([text[i:i+n] for i in range(0, len(text), n)])

def trim_df(data: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Description:
    This function trims the data in a pandas DataFrame to a maximum length of n characters.

    Parameters:
    - data: The pandas DataFrame to trim.
    - n: The maximum length to trim to.
    """
    for col in data.columns:
        data[col] = data[col].apply(lambda x: x[:n-3] + "..." if len(x) > n else x)
    
    return data

def print_title(title: str, color: str = "green") -> None:
    """
    Description:
    This function prints a title to the console.

    Parameters:
    - title: The title to print.
    - color: The color of the title.

    Returns:
    - None
    """
    rprint(f"[bold][{color}]{title}[/]")

def print_df(df: pd.DataFrame, show_index: bool = False, maxcolwidths=20) -> None:
    """
    Description:
    This function prints a pandas DataFrame to the console.

    Parameters:
    - df: The pandas DataFrame to print.
    - show_index: Flag to show the index.

    Returns:
    - None
    """
    print(tabulate(df,
                   headers='keys',
                   tablefmt='psql',
                   maxcolwidths=maxcolwidths,
                   showindex=show_index))
    print()

def format_df(df: pd.DataFrame) -> pd.DataFrame:
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
    if value < 1e3:
        return f"{value} B/s"
    elif value < 1e6:
        return f"{value / 1e3:.2f} KB/s"
    elif value < 1e9:
        return f"{value / 1e6:.2f} MB/s"
    else:
        return f"{value / 1e9:.2f} GB/s"
    
def format_byte(value: float) -> str:
    """
    Description:
    This function formats a value as a byte.

    Parameters:
    - value: The value to format.

    Returns:
    - The formatted value as a byte in string format.
    """
    if value < 1e3:
        return f"{value} B"
    elif value < 1e6:
        return f"{value / 1e3:.2f} KB"
    elif value < 1e9:
        return f"{value / 1e6:.2f} MB"
    else:
        return f"{value / 1e9:.2f} GB"

def format_generic(value) -> str:
    """
    Description:
    This function formats a generic value using generic K, M, G suffixes.

    Parameters:
    - value: The value to format.

    Returns:
    - The formatted value in string format.
    """ 
    # Check if value is float
    if pd.api.types.is_float(value):
        if value < 1e3:
            return f"{value:.2f}"
        elif value < 1e6:
            return f"{value / 1e3:.2f} K"
        elif value < 1e9:
            return f"{value / 1e6:.2f} M"
        else:
            return f"{value / 1e9:.2f} G"
    
    # Else, leave value as is
    return value

def format_fb(value) -> str:
    return format_byte(value * 1e6) # Convert from MB to B

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
"fb_free": format_fb,
"fb_used": format_fb,
"fb_total": format_fb,
"fb_resv": format_fb,
"sm_active": format_percent,
"sm_occupancy": format_percent,
"tensor_active": format_percent,
"fp64_active": format_percent,
"fp32_active": format_percent,
"fp16_active": format_percent,
"total_flop_activity": format_percent,
"dram_active": format_percent,
"pcie_tx_bytes": format_byte_rate,
"pcie_rx_bytes": format_byte_rate,
"nvlink_tx_bytes": format_byte_rate,
"nvlink_rx_bytes": format_byte_rate
}
