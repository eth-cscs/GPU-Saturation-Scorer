# Import DCGM modules
from DcgmReader import DcgmReader

# Import AGI modules
from .metrics import metricIds, demangledMetricNames
from AGI.io import MetricsDataIO

# Import other modules
import time
import uuid
import subprocess
import socket
import sys

# Main class used to run AGI
class GPUMetricsProfiler:
    def __init__(self, gpuIds: list, samplingTime: int, maxRuntime: int) -> None:
        
        # Check if sampling time is too low
        if samplingTime < 100:
            print("Warning: sampling time is too low. Defaulting to 100ms.")
            samplingTime = 100

        # Store options
        self.gpuIds = gpuIds
        self.samplingTime = samplingTime
        self.maxRuntime = maxRuntime
        self.metrics = []
        
        # Get hostname
        self.hostname = socket.gethostname()

        # Generate GPU group UUID
        self.fieldGroupName = str(uuid.uuid4())
    
        # Initialize DCGM reader
        self.dr = DcgmReader(fieldIds=metricIds, gpuIds=self.gpuIds, fieldGroupName=self.fieldGroupName, updateFrequency=int(self.samplingTime*1000)) # Convert from milliseconds to microseconds
            
    def run(self, command: str) -> None:
        # Record start time
        start_time = time.time()

        # Flush stdout and stderr before opening the process
        sys.stdout.flush()

        # Redirect stdout and stderr to output file if specified
        process = subprocess.Popen(command, shell=True)

        # Initialize metrics dictionary
        metrics = {}
        
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
                gpuName = self.get_gpu_name(gpuId)
                
                if gpuName not in metrics:
                    metrics[gpuName] = {}
                
                for metricId in data[gpuId]:
                    
                    metricName = demangledMetricNames[metricId]
                    
                    if metricName not in metrics[gpuName]:
                        metrics[gpuName][metricName] = []

                    metrics[gpuName][metricName].append(data[gpuId][metricId])

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
            
        # Append collected profiling metrics
        self.metrics.append(metrics)

    def getCollectedData(self) -> list:
        return self.metrics
    
    def get_gpu_name(self, gpuId):
        return f"{self.hostname}_gpu:{gpuId}"