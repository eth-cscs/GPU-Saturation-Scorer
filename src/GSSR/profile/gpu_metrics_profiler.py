###############################################################
# Project: GPU saturation scorer 
#
# File Name: gpu_metrics_profiler.py
#
# Description:
# This file implements the GPUMetricsProfiler class, which is used to
# profile GPU metrics. The GPUMetricsProfiler class is responsible for
# collecting GPU metrics data using DCGM.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################

# DCGM imports
from DcgmReader import DcgmReader

# GSS imports
from GSS.io.json_io import JSONDataIO
from GSS.io.binary_io import BinaryDataIO
from GSS.utils.slurm_handler import SlurmJob
from GSS.profile.metrics import metric_ids

# Other imports
import time
import uuid
import subprocess
import socket
import sys
import datetime
import json
import os

class GPUMetricsProfiler:
    """
    Description:
    This class is used to profile GPU metrics. This is done via the DCGM bindings,
    specifically the DcgmReader class. This means that this class is only
    compatible with NVIDIA GPUs.

    Attributes:
    - job: The SlurmJob object that represents the job being profiled.
    - sampling_time: The sampling time in milliseconds.
    - max_runtime: The maximum runtime in seconds. If set to <= 0, the profiler will run indefinitely.
    - metadata: A dictionary containing metadata about the profiling run.
    - data: A dictionary containing the collected metrics data.
    - field_group_name: A unique identifier for the GPU group.
    - file_path: The path to the output file where the data will be stored.
    - io: The data IO handler.
    - dr: The DcgmReader object used to collect the metrics data.

    Methods:
    - __init__(self, job, sampling_time, max_runtime, force_overwrite, output_format): Constructor method.
    - run(self, command): Run the profiler.
    - truncate_data(self): Truncate the collected data to the smallest number of samples.
    - get_collected_data(self): Get the collected metadata and data.

    Notes:
    - The profiler supports two output formats: JSON and binary. The default is JSON as it is human-readable,
      but binary is recommended for large datasets as it is more efficient.
    - Currently, this is the main profiling class in GSS, however in the future support for CPU profiling
      as well as MPI/NCCL profiling will be added.

    """
    def __init__(self, job: SlurmJob, sampling_time: int = 500, max_runtime: int = 600, force_overwrite: bool = False, output_format: str = "json") -> None:
        """
        Description:
        Constructor method for the GPUMetricsProfiler class.

        Parameters:
        - job: The SlurmJob object that represents the job being profiled.
        - sampling_time: The sampling time in milliseconds. Must be >= 20ms.
        - max_runtime: The maximum runtime in seconds. If set to <= 0, the profiler will run indefinitely.
        - force_overwrite: If True, the profiler will overwrite the output file if it already exists.
        - output_format: The output format for the data. Can be either "json" or "binary".

        Returns:
        - None

        Notes:
        - The default sampling time is 500ms.
        - The default maximum runtime is 600s.
        - The default output format is JSON.
        - The default value for force_overwrite is False.
        """
        # Check if sampling time is too low
        if sampling_time < 20:
            print("Warning: sampling time is too low. Defaulting to 20ms.")
            sampling_time = 20

        # Store options
        self.job = job
        self.sampling_time = sampling_time
        self.max_runtime = max_runtime
        self.metadata = {}
        self.data = {}

        # Generate GPU group UUID
        self.field_group_name = str(uuid.uuid4())

        # Generate file path
        self.file_path = os.path.join(
            self.job.output_folder, self.job.output_file)

        # Initialize IO handler (JSON only for now)
        # if output_format == "json":
            # Add .json extension
        self.file_path += ".json"
        self.io = JSONDataIO(self.file_path, force_overwrite=force_overwrite)
        # else: # Binary format
        #     # Add .bin extension
        #     self.file_path += ".bin"
        #     self.io = BinaryDataIO(self.file_path, force_overwrite=force_overwrite)

        self.io.check_overwrite()  # Check if file exists and fail if necessary

        # Initialize DCGM reader
        self.dr = DcgmReader(fieldIds=metric_ids, gpuIds=self.job.gpu_ids, fieldGroupName=self.field_group_name,
                             updateFrequency=int(self.sampling_time * 1000))  # Convert from milliseconds to microseconds

    def run(self, command: str) -> None:
        """
        Description:
        Run the profiler.

        Parameters:
        - command: The command to profile.

        Returns:
        - None

        Notes:
        - The profiler will run indefinitely if max_runtime is set to <= 0.
        Else, it will run for max_runtime seconds before killing the process.
        """
        # Record start time
        start_time = time.time()

        # Flush stdout and stderr before opening the process
        sys.stdout.flush()

        # Redirect stdout
        process = subprocess.Popen(command, shell=True)

        # Throw away first data point
        self.dr.GetLatestGpuValuesAsFieldNameDict()

        # Profiling loop with timeout check
        while self.max_runtime <= 0 or time.time() - start_time < self.max_runtime:
            # Query DCGM for latest samples
            # Note: theoretically, it is possible to query data without such a loop using GetAllGpuValuesAsFieldIdDictSinceLastCall()
            # however the results seem to be inconsistent and not as accurate as using a loop -> use a loop for now
            samples = self.dr.GetLatestGpuValuesAsFieldNameDict()

            # Fuse data in metrics dictionary
            for gpu_id in samples:
                # Initialize dictionary for GPU if it does not exist
                if gpu_id not in self.data:
                    self.data[gpu_id] = {}

                # Store new samples
                for metric in samples[gpu_id]:
                    # Check if metric has been seen before and if not add it to the dictionary
                    if metric not in self.data[gpu_id]:
                        self.data[gpu_id][metric] = []

                    # Append new sample
                    self.data[gpu_id][metric].append(samples[gpu_id][metric])

            # Sleep for sampling frequency
            # Convert from milliseconds to seconds
            time.sleep(self.sampling_time / 1e3)

            # Check if the process has completed
            if process.poll() is not None:
                if process.returncode != 0:
                    print("WARNING: Process exited with non-zero return code. Dumping data to file.")
                break

        # Check if the loop exited due to timeout
        if process.poll() is None:
            sleep(3.0) # Sleep for 3 seconds to try and avoid killing it before it has a chance to exit cleanly.
            print("""WARNING: Killing process due to profiling timeout. Dumping data to file.
                              This may result in some processes returning non-zero exit codes.""")
            
            # Kill the process
            process.kill()
            

        # Compute timestamps
        end_time = time.time()
        elapsed = end_time - start_time

        # Truncate the metrics to the smallest number of samples
        n_samples = self.truncate_data()

        # Assemble the metadata
        self.metadata = {
            "job_id": self.job.job_id,
            "step_id": self.job.step_id,
            "label": self.job.label,
            "hostname": self.job.hostname,
            "proc_id": self.job.proc_id,
            "n_gpus": len(self.job.gpu_ids),
            "gpu_ids": self.job.gpu_ids,
            "start_time": start_time,
            "end_time": end_time,
            "elapsed": elapsed,
            "sampling_time": self.sampling_time,
            "n_samples": n_samples,
            "cmd": command.pop()
        }

        # Dump data to file
        self.io.dump(self.metadata, self.data)

    def truncate_data(self) -> int:
        """
        Description:
        Truncate the collected data to the smallest number of samples.

        Parameters:
        - None

        Returns:
        - n_samples: The smallest number of samples collected.

        Notes:
        - This method is used to ensure that all metrics have the same number of samples.
          Sometimes, due to the timing of the profiler, some metrics may have more samples than others
          and it is necessary to truncate in order to ensure that there are no missing values in the data.
        - Truncation is done by discarding the first samples as they are usually not representative of the workload.
        """
        # Get smallest number of samples
        n_samples = min([len(self.data[gpu_id][metric])
                        for gpu_id in self.data for metric in self.data[gpu_id]])

        # Truncate metrics to the smallest number of samples
        for gpu_id in self.data:
            for metric in self.data[gpu_id]:
                self.data[gpu_id][metric] = self.data[gpu_id][metric][-n_samples:]

        return n_samples

    def get_collected_data(self) -> list:
        return self.metadata, self.data
