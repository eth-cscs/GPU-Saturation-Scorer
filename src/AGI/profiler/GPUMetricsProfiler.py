# Import DCGM modules
from DcgmReader import DcgmReader

# Import AGI modules
from AGI.io.json_io import JSONDataIO
from AGI.utils.slurm_handler import SlurmJob
from AGI.profiler.metrics import metricIds

# Import other modules
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
    def __init__(self, job: SlurmJob, samplingTime: int = 500, maxRuntime: int = 600, forceOverwrite: bool = False) -> None:
        
        # Check if sampling time is too low
        if samplingTime < 20:
            print("Warning: sampling time is too low. Defaulting to 20ms.")
            samplingTime = 20

        # Store options
        self.job = job
        self.samplingTime = samplingTime
        self.maxRuntime = maxRuntime
        self.metadata = {}
        self.data = {}

        # Generate GPU group UUID
        self.fieldGroupName = str(uuid.uuid4())
        
        # Store reference to IOHandler
        self.file_path = os.path.join(self.job.output_folder, self.job.output_file)
        self.IO = JSONDataIO(self.file_path, forceOverwrite = forceOverwrite)
        self.IO.checkOverwrite() # Check if file exists and fail if necessary

        # Initialize DCGM reader
        self.dr = DcgmReader(fieldIds=metricIds, gpuIds=self.job.gpuIds, fieldGroupName=self.fieldGroupName, updateFrequency=int(self.samplingTime*1000)) # Convert from milliseconds to microseconds)
    
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
        while self.maxRuntime <= 0 or time.time() - start_time < self.maxRuntime:
            # Query DCGM for latest samples
            # Note: theoretically, it is possible to query data without such a loop using GetAllGpuValuesAsFieldIdDictSinceLastCall()
            # however the results seem to be inconsistent and not as accurate as using a loop -> use a loop for now
            samples = self.dr.GetLatestGpuValuesAsFieldNameDict()
        
            # Fuse data in metrics dictionary
            for gpuId in samples:
                # Initialize dictionary for GPU if it does not exist
                if gpuId not in self.data:
                    self.data[gpuId] = {}
                
                # Store new samples
                for metric in samples[gpuId]:
                    # Check if metric has been seen before and if not add it to the dictionary
                    if metric not in self.data[gpuId]:
                        self.data[gpuId][metric] = []

                    # Append new sample
                    self.data[gpuId][metric].append(samples[gpuId][metric])

            # Sleep for sampling frequency
            time.sleep(self.samplingTime/1e3) # Convert from milliseconds to seconds

            # Check if the process has completed
            if process.poll() is not None:
                if process.returncode != 0:
                    raise Exception("The profiled command returned a non-zero exit code.")
                break

        # Check if the loop exited due to timeout
        # We check for None because poll() returns None if the process is still running, otherwise it returns the exit code
        if process.poll() is None: # 
            # Kill the process
            process.kill()
            print("Process killed due to timeout.")
        
        # Compute timestamps
        end_time = time.time()
        duration = end_time - start_time
        start_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y/%m/%d-%H:%M:%S')
        end_time = datetime.datetime.fromtimestamp(end_time).strftime('%Y/%m/%d-%H:%M:%S')
        
        # We need to truncate the metrics to the smallest number of samples across all GPUs
        # This is because on occasion, some metrics may have a few more samples than others (in the order of 1-5 samples)
        n_samples = self.truncateData()

        # Assemble the metadata
        self.metadata = {
            "SLURM_JOB_ID": self.job.jobId,
            "label": self.job.label,
            "hostname": self.job.hostname,
            "procid": self.job.procId,
            "n_gpus": len(self.job.gpuIds),
            "gpu_ids": self.job.gpuIds,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "sampling_time": self.samplingTime,
            "n_samples": n_samples,
            "cmd": command[0]
        }

        # Dump data to file
        self.IO.dump(self.metadata, self.data)

    # This function truncates the metrics to the smallest number of samples across all GPUs
    # The return value is the number of samples.
    def truncateData(self) -> int:
        # Get smallest number of samples
        n_samples = min([len(self.data[gpuId][metric]) for gpuId in self.data for metric in self.data[gpuId]])

        # Truncate metrics to the smallest number of samples
        for gpuId in self.data:
            for metric in self.data[gpuId]:
                self.data[gpuId][metric] = self.data[gpuId][metric][-n_samples:] 
        
        return n_samples

    def getCollectedData(self) -> list:
        return (self.metadata, self.metrics)