# Import DCGM modules
from DcgmReader import DcgmReader

# Import AGI modules
from .metrics import metricIds, demangledMetricNames
from AGI.io import MetricsDataIO
from AGI.utils.utils import readEnvVar

# Import other modules
import time
import uuid
import subprocess
import socket
import sys
import datetime

# Main class used to run AGI
class GPUMetricsProfiler:
    def __init__(self, gpuIds: list, label: str = None, samplingTime: int = 500, maxRuntime: int = 600) -> None:
        
        # Check if sampling time is too low
        if samplingTime < 100:
            print("Warning: sampling time is too low. Defaulting to 100ms.")
            samplingTime = 100

        # Store options
        self.gpuIds = gpuIds
        self.samplingTime = samplingTime
        self.maxRuntime = maxRuntime
        self.metadata = {}
        self.metrics = {}

        # Generate GPU group UUID
        self.fieldGroupName = str(uuid.uuid4())
        
        # Get metadata
        self.hostname = socket.gethostname()
        self.procid = str(readEnvVar("SLURM_PROCID", throw=False))
        self.jobid = str(readEnvVar("SLURM_JOB_ID", throw=False))
        self.label = label if label else readEnvVar("SLURM_JOB_NAME", throw=False)

        # If label is still None, generate a default label
        if self.label is None:
            self.label = f"unlabeled_job_{self.jobid}"

        # Make sure label is a valid table name with no spaces
        self.label = self.label.replace(" ", "_")
    
        # Initialize DCGM reader
        self.dr = DcgmReader(fieldIds=metricIds, gpuIds=self.gpuIds, fieldGroupName=self.fieldGroupName, updateFrequency=int(self.samplingTime*1000)) # Convert from milliseconds to microseconds)
            
    def run(self, command: str) -> None:
        # Record start time
        start_time = time.time()

        # Flush stdout and stderr before opening the process
        sys.stdout.flush()

        # Redirect stdout and stderr to output file if specified
        process = subprocess.Popen(command, shell=True)
        
        # Throw away first data point
        data = self.dr.GetLatestGpuValuesAsFieldIdDict()
        
        # Profiling loop with timeout check
        while self.maxRuntime <= 0 or time.time() - start_time < self.maxRuntime:
            
            # Query DCGM for latest values
            # Note: theoretically, it is possible to query data without such a loop using GetAllGpuValuesAsFieldIdDictSinceLastCall()
            # however the results seem to be inconsistent and not as accurate as using a loop -> use a loop for now
            
            # Note 2: we could use GetAllGpuValuesAsFieldNameDict() instead of GetLatestGpuValuesAsFieldIdDict() to avoid having to demangle the field names,
            # however the former yields long string names that are annoying to parse -> use the latter for now
            data = self.dr.GetLatestGpuValuesAsFieldIdDict()
        
            # Fuse data in metrics dictionary
            for gpuId in data:
                gpuName = self.getGPUName(gpuId)
                
                if gpuName not in self.metrics:
                    self.metrics[gpuName] = {}
                
                for metricId in data[gpuId]:
    
                    metricName = demangledMetricNames[metricId]
                    
                    if metricName not in self.metrics[gpuName]:
                        self.metrics[gpuName][metricName] = []

                    self.metrics[gpuName][metricName].append(data[gpuId][metricId])

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
        
        end_time = time.time()
        
        # Compute number of samples
        n_samples = len(next(iter(self.metrics.values()))["DEV_GPU_UTIL"])

        # Compute timestamps
        duration = end_time - start_time
        start_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y/%m/%d-%H:%M:%S')
        end_time = datetime.datetime.fromtimestamp(end_time).strftime('%Y/%m/%d-%H:%M:%S')

        # Generate metadata
        self.metadata = {
            "slurm_job_id": self.jobid,
            "label": self.label,
            "hostname": self.hostname,
            "procid": self.procid,
            "n_gpus": len(self.gpuIds),
            "gpu_ids": ", ".join([str(id) for id in self.gpuIds]),
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "tname": ", ".join([self.getGPUName(gpuId) for gpuId in self.gpuIds]),
            "sampling_time": self.samplingTime,
            "n_samples": n_samples
        }

    def getCollectedData(self) -> list:
        return (self.metadata, self.metrics)
    
    def getGPUName(self, gpuId):
        return f"{self.label}/{self.hostname}/gpu{gpuId}"