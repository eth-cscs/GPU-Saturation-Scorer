###############################################################
# Project: GPU saturation scorer
#
# File Name: report.py
#
# Description:
# This file contains the report class that is used to generate
# the PDF report of the analysis. This class includes
# all necessary plotting and formatting functions for the report.
#
# Authors:
# Marcel Ferrari (CSCS)
# Cerlane Leong (CSCS)
#
###############################################################

import numpy as np
import pandas as pd
import fpdf as PDF
import uuid
import os
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.spatial.qhull import QhullError
from GSS.io.format import *

from pathlib import Path

# Disable stupid fpdf2 warnings
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("fpdf").setLevel(logging.CRITICAL)

DIR = Path(__file__).parent
FONT_DIR = DIR / ".."  / "fonts"

class PDFReport:
    def __init__(self, db, job, outfile, tmp_dir):
        self.db = db
        self.job = job
        self.outfile = outfile
        self.tmp_dir = tmp_dir
        self.GRAY = 220
       

    def downsample(self, arrays, nmax = 100):
        # Downsample data to a maximum of nmax points by 
        # taking the mean nmax segments of the data
        rax = []
        for x in arrays:
            n = len(x)
            if n > nmax:
                x = np.array_split(x, nmax)
                x = np.array([np.mean(i) for i in x])
            rax.append(x)
        return tuple(rax)

    def body(self):
        #self.pdf.set_font("Helvetica", "", 12)
        self.pdf.set_font("DejaVuSans", "", size=10)

    def title(self):
        #self.pdf.set_font("Helvetica", "B", 18)
        self.pdf.set_font("DejaVuSans", "B", size=16)


    def subtitle(self):
        #self.pdf.set_font("Helvetica", "I", 14)
        self.pdf.set_font("DejaVuSans", "I", 12)
    
    def bold_title(self):
        #self.pdf.set_font("Helvetica", "B", 16)
        self.pdf.set_font("DejaVuSans", "B", 14)

    def newpage(self):
        self.pdf.add_page()

    def draw_dataframe(self, df):
        """
        Function to draw a dataframe as a table in the PDF report.
        """  
        COLUMNS = [list(df)]  # Get list of dataframe columns
        ROWS = df.values.tolist()  # Get list of dataframe rows
        DATA = COLUMNS + ROWS  # Combine columns and rows in one list
        with self.pdf.table(
            borders_layout="MINIMAL",
            cell_fill_color=self.GRAY,
            cell_fill_mode="ROWS",
            width=(self.pdf.w - self.pdf.l_margin - self.pdf.r_margin),
            line_height=self.pdf.font_size*2.,
            text_align="LEFT",
        ) as table:
            for data_row in DATA:
                row = table.row()
                for datum in data_row:
                    row.cell(datum)

    def draw_title(self):
        # Title
        self.title()
        self.pdf.set_xy(0, 15)
        self.pdf.cell(297, 10, f"GPU Metrics Report - {self.job['label']}", 0, 1, "C")
        # Subtitle
        self.subtitle()
        self.pdf.cell(297, 10, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")

    def draw_metadata(self):
        # Reduce length of cmd if it is too long. 
        if len(self.job['cmd'])>500:
            self.job['cmd'] = f"{self.job['cmd'][0:500]}" + "..."
        # Add metadata table to the report
        metadata = {
            "Job Metadata Entry": ["Job ID", 
                      "Step ID", 
                      "Label", 
                      "Command", 
                      "No. hosts", 
                      "No. processes",
                      "No. GPUs",
                      "Median elapsed time"],
            "Value": [
                self.job['job_id'],
                self.job['step_id'],
                self.job['label'],
                self.job['cmd'],
                self.job['n_hosts'],
                self.job['n_procs'],
                self.job['n_gpus'],
                f"{self.job['median_elapsed']:.2f}s"
            ]
        }

        metadata = pd.DataFrame(metadata).astype(str)

        self.pdf.ln(5)
        self.bold_title()
        self.pdf.cell(text="Job Metadata", ln=True)
        self.body()

        self.draw_dataframe(metadata)
    
    def draw_summary(self):
        # Print aggregate data over all gpus for all metrics
        # Get the raw data for the job
        data = self.db.query(f"SELECT {self.job['metrics']} FROM data WHERE job_id={self.job['job_id']} AND step_id={self.job['step_id']}")
        
        # Aggregate data
        agg = format_df(data.agg(['median', 'mean', 'min', 'max'])).T.astype(str) # Transpose to get metrics as rows
        agg.reset_index(inplace=True)
        agg.rename(columns={"index": "Metric"}, inplace=True)
        
        # Add table to the report
        self.bold_title()
        self.pdf.cell(text="Summary of performance metrics", ln=True)
        self.body()
        self.pdf.ln(2)
        self.pdf.cell(text="Aggregate data over all GPUs and over the full workload time.", ln=True)
        
        # Draw the table
        self.draw_dataframe(agg)
    
    def draw_gpu_metrics(self):
   
        # Query average performance metrics per each GPU
        data = self.db.query(f"""
                            SELECT
                                *
                            FROM
                                data
                            WHERE
                                job_id={self.job['job_id']} AND step_id={self.job['step_id']}
                            ORDER BY
                                proc_id, gpu_id ASC
                            """)
    
        # Drop sample_index and time
        data.drop(columns=["job_id", "step_id", "sample_index", "time"], inplace=True)
        
        # Aggregate all metrics grouping by proc_id and gpu_id
        data = data.groupby(["proc_id", "gpu_id"]).mean()

        # Reset index to turn proc_id and gpu_id back into columns
        data.reset_index(inplace=True)

        # Rename columns
        data = format_df(data).astype(str)

        # Add table to the report
        self.bold_title()
        self.pdf.cell(text="GPU-Specific Averages", ln=True)
        self.body()
        self.pdf.ln(2)
        self.pdf.cell(text="Aggregate data over the full workload time.", ln=True)
        self.body()

        # Draw the table 8 metrics at a time
        metrics = self.job["metrics"].split(",")

        step = 6
        for i in range(0, len(metrics), step):
            self.draw_dataframe(data[["proc_id", "gpu_id"] + metrics[i:i+step]])
            # Create new page if there are more metrics to display
            if i + step < len(metrics):
                self.newpage()
     
    def draw_warnings(self, warnmsg):
        self.pdf.set_font("DejaVuSans", style="", size=10)
        self.pdf.ln()
        self.pdf.set_text_color(220, 50, 50)
        self.pdf.write(text="[WARNING] ")
        self.pdf.set_text_color(50, 50, 50)
        self.pdf.write(text=warnmsg)
        self.pdf.ln()

    def plot_time_series(self, x, y_avg, min_y, max_y, metric):
        # Downsample data to a maximum of 1000 points
        x, y_avg, min_y, max_y = self.downsample((x, y_avg, min_y, max_y))

        # Set the figure size to fit an A4 page (11 inches width, 3 inches height)
        fig = plt.figure(figsize=(11, 2.0))
        
        # Create the grid for the plots with adjusted width ratios
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
        
        # Create the time series plot (left)
        ax1 = fig.add_subplot(gs[0])
        # Add the shaded area between min_y and max_y
        ax1.fill_between(x, min_y, max_y, color='lightblue', alpha=0.5, label='Range')
        
        # Plot the actual y line over the shaded area
        ax1.plot(x, y_avg, label=metric, color='b', linewidth=2.0)
        ax1.set_xlabel('Time')
        ax1.set_ylabel(metric, labelpad=10)
        miny, maxy = min(min_y), max(max_y)
        ax1.set_ylim(miny - 0.1 * abs(miny), maxy + 0.1 * abs(maxy))
        ax1.grid(alpha=0.8)

        # Set 10 ticks on the y-axis
        yticks = np.linspace(miny, maxy, 8)
        # Set yticklabels to scientific notation with 1 decimal place            
        y_avg_max = max(y_avg)
        if 1.0 >= y_avg_max >= 0.1: # Percentages
            yticklabels = [f"{y:.2f}" for y in yticks]
        else:
            yticklabels = [f"{y:.1e}" for y in yticks]
        ax1.set_yticks(yticks, yticklabels)
        
        # Create the distribution plot (right)
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        n_bins = 16
        bins = np.linspace(miny, maxy, n_bins)
        ax2.hist(y_avg, orientation='horizontal', alpha=0.8, color='red',
                    weights=np.full_like(y_avg, 1./len(y_avg)), bins=bins)
        ax2.set_xlabel('Percentage')
        
        # Set x-axis limits to 0-1 to represent percentages
        _,pmax = ax2.get_xlim()
        pmax = round(pmax, 1) # Round pmax to the nearest 0.1
        ax2.set_xlim(0, pmax)
        xticks = np.linspace(0, pmax, int(pmax*5) + 1)
        xtick_labels = [f"{int(x*100)}%" for x in xticks]
        ax2.set_xticks(xticks, xtick_labels, rotation=45)
        
        # Disable y-axis tick labels for the distribution plot (right side) without hiding them on the left side
        ax2.tick_params(axis='y', which='both', labelleft=False)

        # Ensure both plots have the same y-axis limits
        ax2.set_ylim(ax1.get_ylim())
        
        # Generate a unique filename for the plot
        filename = f"{uuid.uuid4()}.svg"
        figpath = os.path.join(self.tmp_dir, filename)
        plt.savefig(figpath, format='svg')
        
        # Close the figure to free up memory
        plt.close(fig)
        return figpath
    
    def draw_time_series(self):
        # Add title
        self.bold_title()
        self.pdf.cell(text="Time-Series of Performance Metrics", ln=True)
        self.body()
        self.pdf.ln(2)
        self.pdf.cell(text="Aggregate data over the all GPUs.", ln=True)
        self.pdf.ln(5)

        # Collect average metrics:
        metrics = self.job["metrics"].split(",")
        
        # Aggregate data over all processes and all GPUs
        # Use sample_id, time to group data by time and sample
        # as we dont want to deal with floating point time values
        data = self.db.query(f"""
                            SELECT
                                sample_index,time,{','.join([f'AVG({m}) AS {m}, MIN({m}) AS min_{m}, MAX({m}) AS max_{m}' for m in metrics])}
                            FROM
                                data
                            WHERE
                                job_id={self.job['job_id']} AND step_id={self.job['step_id']}
                            GROUP BY
                                sample_index
                            ORDER BY
                                time ASC
                            """)
        
        t = data["time"].to_numpy()
        
        print("Generating time series plots...")
        for metric in tqdm(metrics):
            y_avg = data[metric].to_numpy()
            min_y = data[f"min_{metric}"].to_numpy()
            max_y = data[f"max_{metric}"].to_numpy()
            figpath = self.plot_time_series(t, y_avg, min_y, max_y, metric)
            self.pdf.image(figpath, x=self.pdf.l_margin, w=self.pdf.w - self.pdf.l_margin - self.pdf.r_margin)
            self.pdf.ln(5)

    def plot_load_balancing(self, y_avg, min_y, max_y, metric):
        # Create custom indexing
        x = np.array(list(range(len(y_avg))))
        
        # Set the figure size to fit an A4 page (11 inches width, 3 inches height)
        fig, ax = plt.subplots(figsize=(11, 2.0))

        # Create the load-balancing plot
        ax.fill_between(x, min_y, max_y, color='lightblue', alpha=0.5, label='Range')
        ax.plot(x, y_avg, label=metric, color='b', linewidth=2.0)
        ax.set_xlabel('Global GPU index')
        ax.set_ylabel(metric, labelpad=10)
        miny, maxy = min(min_y), max(max_y)
        ax.set_ylim(miny - 0.1 * abs(miny), maxy + 0.1 * abs(maxy))
        ax.grid(alpha=0.8)

        # Generate a unique filename for the plot
        filename = f"{uuid.uuid4()}.svg"
        figpath = os.path.join(self.tmp_dir, filename)
        plt.savefig(figpath, format='svg')

        # Close the figure to free up memory
        plt.close(fig)

        return figpath
    
    def draw_load_balancing(self):
        # Add title
        self.bold_title()
        self.pdf.cell(text="Load Balancing", ln=True)
        self.body()
        self.pdf.ln(2)
        self.pdf.cell(text="Aggregate data over time.", ln=False)
        self.pdf.cell(text="The shaded area shows min/max values.", ln=False)
        self.pdf.cell(text="A horizontal line represents ideal load-balancing.", ln=True)
        self.pdf.ln(5)

        # Collect average metrics:
        metrics = self.job["metrics"].split(",")
        
        # Aggregate data over time for each GPU
        data = self.db.query(f"""
                            SELECT
                               proc_id, gpu_id, {','.join([f'AVG({m}) AS {m}, MIN({m}) AS min_{m}, MAX({m}) AS max_{m}' for m in metrics])}
                            FROM
                                data
                            WHERE
                                job_id={self.job['job_id']} AND step_id={self.job['step_id']}
                            GROUP BY
                                proc_id, gpu_id
                            ORDER BY
                                proc_id, gpu_id ASC
                            """)
        
        print("Generating load-balancing plots...")
        for metric in tqdm(metrics):
            y_avg = data[metric].to_numpy()
            min_y = data[f"min_{metric}"].to_numpy()
            max_y = data[f"max_{metric}"].to_numpy()
            figpath = self.plot_load_balancing(y_avg, min_y, max_y, metric)
            self.pdf.image(figpath, x=self.pdf.l_margin, w=self.pdf.w - self.pdf.l_margin - self.pdf.r_margin)
            self.pdf.ln(5)

    def plot_heatmap(self, X, T, Y, metric):
        # Proceed with plotting
        fig, ax = plt.subplots(figsize=(7, 5.25))
        contour = ax.contourf(T, X, Y, levels=20, cmap='jet')

        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('GPU ID (unique)')
        ax.set_title(metric)
        plt.colorbar(contour, ax=ax, label="Metric Value")

        # Generate a unique filename for the plot
        filename = f"{uuid.uuid4()}.svg"
        figpath = os.path.join(self.tmp_dir, filename)
        plt.savefig(figpath, format='svg')

        # Close the figure to free up memory
        plt.close(fig)

        return figpath
    
    def draw_heatmaps(self):
        # Add title
        self.bold_title()
        self.pdf.cell(text="Metric Heatmaps", ln=True)
        self.body()
        self.pdf.ln(2)
        self.pdf.cell(text="Evolution of performance metrics over time.", ln=False)
        self.pdf.ln(5)

        # Collect average metrics:
        metrics = self.job["metrics"].split(",")
        
        # Check how many unique GPUs we have
        n_gpus = self.db.query(f"""
                                SELECT COUNT(*) AS n_gpus
                                FROM (
                                    SELECT DISTINCT proc_id, gpu_id
                                    FROM data
                                    WHERE job_id={self.job['job_id']} AND step_id={self.job['step_id']}
                                )""")['n_gpus'][0]
        
        # Query the data
        data = self.db.query(f"""
                            SELECT
                               proc_id, gpu_id, sample_index, {','.join([f"{m}" for m in metrics])}
                            FROM
                                data
                            WHERE
                                job_id={self.job['job_id']} AND step_id={self.job['step_id']}
                            ORDER BY
                                proc_id, gpu_id, sample_index ASC
                            """)
        
        # Create unique ids
        data["unique_id"] = data["proc_id"].astype(str) + "_" + data["gpu_id"].astype(str)

        # Factorize the unique ids to get unique integer values
        data["unique_id"] = pd.factorize(data["unique_id"])[0]

        # Read sampling time from job metadata
        sampling_time = self.job['sampling_time']
        
        # x and t data
        x = data["unique_id"].to_numpy()
        t = data["sample_index"].to_numpy() * sampling_time/1000
        
        # Grid coordinates
        x_grid = np.arange(n_gpus)
        t_grid = data["sample_index"].unique() * sampling_time/1000  # Convert from ms to seconds

        # Create meshgrid
        T, X = np.meshgrid(t_grid, x_grid)        

        print("Generating heatmaps...")
        for metric in tqdm(metrics):

            # Extract the metric values for the current metric
            y = data[metric].to_numpy()
       
            #if np.abs(y).max() > 1e-3:
            try: 
                # Interpolate (x, t, y) to (X, T, Y)
                Y = griddata((x, t), y, (X, T), method='linear')

                # Plot heatmap for the current metric
                figpath = self.plot_heatmap(X, T, Y, metric)

                self.pdf.image(figpath, x=self.pdf.l_margin, h=(self.pdf.h - self.pdf.b_margin - self.pdf.t_margin)/2.2)
                self.pdf.ln(5)
            except QhullError:
                    emsg = "Data is 2 dimension. Heatmap ("+metric+") cannot be generated. This is likely because only one gpu is used."
                    self.draw_warnings(emsg)
                    print("[WARN] " + emsg)
            #else:
            #     emsg = "Metric is too small for heatmap ("+metric+") to be plotted."
            #     self.draw_warnings(emsg)
            #     print("[WARN] " + emsg)

    def write(self):        
        # Create new PDF file
        self.pdf = PDF.FPDF(orientation='L')
        self.pdf.add_font(fname=FONT_DIR / "DejaVuSans.ttf")
        self.pdf.add_font("DejaVuSans", style="B", fname=FONT_DIR / "DejaVuSans-Bold.ttf")
        self.pdf.add_font("DejaVuSans", style="I", fname=FONT_DIR / "DejaVuSans-Oblique.ttf")
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()


        
        # Draw the title and metadata
        self.draw_title()
        self.draw_metadata()
        self.newpage()

        # Draw summary
        self.draw_summary()
        self.newpage()

        # Draw GPU metrics
        self.draw_gpu_metrics()
        self.newpage()

        # Draw time series
        self.draw_time_series()
        self.newpage()

        # Draw load balancing
        self.draw_load_balancing()
        
        # Add vertical page
        self.pdf.add_page(orientation='P')

        # Draw heatmaps
        self.draw_heatmaps()
        
        # Save the PDF
        self.pdf.output(self.outfile)
        
        
        
