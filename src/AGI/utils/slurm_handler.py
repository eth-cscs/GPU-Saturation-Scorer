# System imports
import socket
import os

# Class used to interface with the Slurm environment variables
class SlurmJob:
    def __init__(self, label: str = None, output_folder: str = None):
        self.procId = None
        self.jobId = None
        self.hostname = None
        self.gpuIds = None
        self.label = label
        self.output_folder = output_folder
        self.output_file = None

        # Read the environment variables
        self.readEnvironment()

    # Read the environment variables from the Slurm job and store them in the object
    def readEnvironment(self):
        # Function used to read environment variables
        
        # Read job ID and process ID - throw exception if not found
        self.jobId = int(self.readEnvVar("SLURM_JOB_ID", throw=True))
        self.procId = int(self.readEnvVar("SLURM_PROCID", throw=True))
        
        # If no label has been set explicitly, use the job ID
        if self.label is None:
            self.label = f"unlabeled_job_{self.jobId}"
        
        # Read GPU IDs - do not throw exception if not found
        # If not found, use SLURM_procId mod 4 to determine GPU ID
        # This is a workaround for when --gpus-per-task=1 is not set or when SLURM is not working properly
        # It assumes that there is exactly one GPU per rank and 4 gpus per node.
        # Also there is no guarantee that each rank will be assigned to the correct GPU.
        errMsg = "SLURM_STEP_GPUS not found: try setting the --gpus-per-task flag. Using SLURM_PROCID mod 4 to determine GPU ID."
        self.gpuIds = self.readEnvVar("SLURM_STEP_GPUS", throw=False, errMsg=errMsg)
        
        if self.gpuIds:
            self.gpuIds = [int(gpu) for gpu in self.gpuIds.strip().split(',')]
        else:
            self.gpuIds = [self.procId%4]

        # Get hostname - this is done via the socket module and should always work regardless of the Slurm environment
        self.hostname = socket.gethostname()

        # Set output folder and file
        if not self.output_folder:
            self.output_folder = f"AGI_JOB_{self.jobId}"

        # Set output file
        self.output_file = f"{self.label}_proc_{self.procId}.json"
    
    def readEnvVar(self, varName: str, throw: bool = True, errMsg = None) -> str:
        try:
            return os.environ[varName]
        except KeyError:
            if not errMsg:
                errMsg = f"Environment variable {varName} not found. Check that you are launching this tool in a Slurm job."
            
            if throw:
                raise Exception(errMsg)
            else:
                print(f"WARNING: {errMsg}")
                return None


