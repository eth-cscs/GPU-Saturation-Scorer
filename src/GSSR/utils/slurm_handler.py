###############################################################
# Project: GPU saturation scorer
#
# File Name: slurm_handler.py
#
# Description:
# This file implements the SlurmJob class, which is used to
# interface with the Slurm environment variables. This class
# is used to read the environment variables and store them in
# the object for easy access.
#
# Authors:
# Marcel Ferrari (CSCS)
# Cerlane Leong (CSCS)
#
###############################################################

import socket
import os
import sys

class SlurmJob:
    """
    Description:
    This class is used to interface with the Slurm environment variables.
    It reads the environment variables and stores them in the object for easy access.

    Attributes:
    - proc_id: The process ID of the current job.
    - job_id: The job ID of the current job.
    - hostname: The hostname of the current node.
    - gpu_ids: The GPU IDs assigned to the current job.
    - label: The label of the current job.
    - output_folder: The output folder of the current job. 
    - output_file: The output file of the current job.

    Methods:
    - __init__(self, label: str = None, output_folder: str = None): Constructor method.
    - read_environment(self): Read the environment variables from the Slurm job and store them in the object.
    - read_env_var(self, var_name: str, throw: bool = True, error_msg=None) -> str: Read an environment variable and return its value.

    Notes:
    - None

    """
    def __init__(self, label: str = None, output_folder: str = None) -> None:
        """
        Description:
        Constructor method.

        Parameters:
        - label: The label of the current job. Used to identify which workload is being profiled.
        - output_folder: The output folder of the current job. Used to dump the output profiling data.

        Returns:
        - None

        Notes:
        - None
        """
        self.proc_id = None
        self.job_id = None
        self.hostname = None
        self.gpu_ids = None
        self.label = label
        self.output_folder = output_folder
        self.output_file = None

        # Read the environment variables
        self.read_environment()

    
    def read_environment(self) -> None:
        """
        Description:
        This method reads the environment variables from the Slurm job and stores them in the object.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - GSS uses the following environment variables: SLURM_JOB_ID, SLURM_PROCID, SLURM_STEP_GPUS.
        - If SLURM_STEP_GPUS is not found, the method will use SLURM_PROCID mod 4 to determine the GPU ID.
          This is only a workaround and may not work in all cases.
        - The method will throw an error if SLURM_JOB_ID or SLURM_PROCID are not found.
        """
        # Function used to read environment variables

        # Read job ID and process ID - throw exception if not found
        self.job_id = int(self.read_env_var("SLURM_JOB_ID", throw=True))
        self.step_id = int(self.read_env_var("SLURM_STEP_ID", throw=True))
        self.proc_id = int(self.read_env_var("SLURM_PROCID", throw=True))

        # If no label has been set explicitly, use the job ID
        if self.label is None:
            self.label = f"unlabeled_job_{self.job_id}"
        else:
            self.label = self.label.replace(" ", "_") # Replace spaces with underscores

        # Append job ID and step ID to the label
        self.label = f"{self.label}_job_{self.job_id}_step_{self.step_id}"

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

        # if output_folder is the default 'profile_out', then append job_id to it
        if self.output_folder=='profile_out':
            self.output_folder += f"_JobID_{self.job_id}"

        # Set output directory for specific job
        self.output_folder = os.path.join(self.output_folder, self.label)

        # Set output file
        self.output_file = f"{self.label}_proc_{self.proc_id}"

    def read_env_var(self, var_name: str, throw: bool = True, error_msg: str = None) -> str:
        """
        Description:
        This method reads an environment variable and returns its value.

        Parameters:
        - var_name: The name of the environment variable to read.
        - throw: If True, the method will throw an error if the environment variable is not found.
        - error_msg: The error message to display if the environment variable is not found.

        Returns:
        - The value of the environment variable in string format.

        Notes:
        - If the environment variable is not found, the method will throw an error if throw is True.
        """
        try:
            return os.environ[var_name]
        except KeyError:
            if not error_msg:
                error_msg = f"Environment variable {var_name} not found. Check that you are launching this tool in a Slurm job."

            if throw:
                sys.exit(error_msg)
            else:
                print(f"WARNING: {error_msg}")
                return None
