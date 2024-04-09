import socket
import os

# Class used to interface with the Slurm environment variables


class SlurmJob:
    def __init__(self, label: str = None, output_folder: str = None):
        self.proc_id = None
        self.job_id = None
        self.hostname = None
        self.gpu_ids = None
        self.label = label
        self.output_folder = output_folder
        self.output_file = None

        # Read the environment variables
        self.read_environment()

    # Read the environment variables from the Slurm job and store them in the object
    def read_environment(self):
        # Function used to read environment variables

        # Read job ID and process ID - throw exception if not found
        self.job_id = int(self.read_env_var("SLURM_JOB_ID", throw=True))
        self.proc_id = int(self.read_env_var("SLURM_PROCID", throw=True))

        # If no label has been set explicitly, use the job ID
        if self.label is None:
            self.label = f"unlabeled_job_{self.job_id}"

        # Read GPU IDs - do not throw exception if not found
        error_msg = "SLURM_STEP_GPUS not found: try setting the --gpus-per-task flag. Using SLURM_PROCID mod 4 to determine GPU ID."
        self.gpu_ids = self.read_env_var(
            "SLURM_STEP_GPUS", throw=False, error_msg=error_msg)

        if self.gpu_ids:
            self.gpu_ids = [int(gpu)
                            for gpu in self.gpu_ids.strip().split(',')]
        else:
            self.gpu_ids = [self.proc_id % 4]

        # Get hostname - this is done via the socket module and should always work regardless of the Slurm environment
        self.hostname = socket.gethostname()

        # Set output folder and file
        if not self.output_folder:
            self.output_folder = f"AGI_JOB_{self.job_id}"

        # Set output file
        self.output_file = f"{self.label}_proc_{self.proc_id}.json"

    def read_env_var(self, var_name: str, throw: bool = True, error_msg=None) -> str:
        try:
            return os.environ[var_name]
        except KeyError:
            if not error_msg:
                error_msg = f"Environment variable {var_name} not found. Check that you are launching this tool in a Slurm job."

            if throw:
                raise Exception(error_msg)
            else:
                print(f"WARNING: {error_msg}")
                return None
