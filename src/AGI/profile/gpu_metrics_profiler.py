# DCGM imports
from DcgmReader import DcgmReader

# AGI imports
from AGI.io.json_io import JSONDataIO
from AGI.io.binary_io import BinaryDataIO
from AGI.utils.slurm_handler import SlurmJob
from AGI.profile.metrics import metric_ids

# Other imports
import time
import uuid
import subprocess
import socket
import sys
import datetime
import json
import os

# Main class used to run AGI


class GPUMetricsProfiler:
    def __init__(self, job: SlurmJob, sampling_time: int = 500, max_runtime: int = 600, force_overwrite: bool = False, output_format: str = "json") -> None:
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

        # Initialize IO handler
        if output_format == "json":
            # Add .json extension
            self.file_path += ".json"
            self.io = JSONDataIO(self.file_path, force_overwrite=force_overwrite)
        else: # Binary format
            # Add .bin extension
            self.file_path += ".bin"
            self.io = BinaryDataIO(self.file_path, force_overwrite=force_overwrite)

        self.io.check_overwrite()  # Check if file exists and fail if necessary

        # Initialize DCGM reader
        self.dr = DcgmReader(fieldIds=metric_ids, gpuIds=self.job.gpu_ids, fieldGroupName=self.field_group_name,
                             updateFrequency=int(self.sampling_time * 1000))  # Convert from milliseconds to microseconds

    def run(self, command: str) -> None:
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
                    raise Exception(
                        "The profiled command returned a non-zero exit code.")
                break

        # Check if the loop exited due to timeout
        if process.poll() is None:
            # Kill the process
            process.kill()
            print("Process killed due to timeout.")

        # Compute timestamps
        end_time = time.time()
        elapsed = end_time - start_time

        # Truncate the metrics to the smallest number of samples
        n_samples = self.truncate_data()

        # Assemble the metadata
        self.metadata = {
            "job_id": self.job.job_id,
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
