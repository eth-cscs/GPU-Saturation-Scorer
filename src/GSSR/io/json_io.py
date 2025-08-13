###############################################################
# Project: GPU Saturation Scorer
#
# File Name: json_io.py
#
# Description:
# This file contains the JSONDataIO class, which is used to
# handle JSON data input/output.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################

import json
import os
from GSSR.io.base_io import BaseIO

class JSONDataIO(BaseIO):
    """
    Description:
    This class is used to handle JSON data input/output.

    Attributes:
    - file (str): Path to the JSON file.
    - force_overwrite (bool): Flag to force overwrite of existing file.

    Methods:
    - dump(self, metadata: dict, data: dict) -> None: Dump metadata and data to a JSON file.
    - load(self) -> tuple: Load metadata and data from a JSON file.

    Notes:
    - This class uses the json module to serialize and deserialize data.
    - JSON is the default format for data output as it is human-readable.
      If you need to save space, consider using the BinaryDataIO class.
    """

    def __init__(self, file: str, force_overwrite: bool = False) -> None:
        """
        Description:
        Constructor method.

        Parameters:
        - file (str): Path to the JSON file.
        - force_overwrite (bool): Flag to force overwrite of existing file.

        Returns:
        - None

        Notes:
        - The force_overwrite flag can be used to overwrite the file without checking if it exists.
        """
        # Call parent constructor
        super().__init__(file, force_overwrite)

    # This function dumps the data to a JSON file
    def dump(self, metadata: dict, data: dict) -> None:
        """
        Description:
        This method dumps metadata and data to a JSON file.

        Parameters:
        - metadata (dict): A dictionary containing metadata.
        - data (dict): A dictionary containing data.

        Returns:
        - None

        Notes:
        - The data dictionary should contain a mapping from GPU ID to a dictionary containing metric: values pairs.
          This is important as the GPU IDs are NOT stored explicitly in the metadata.
        """
        # Create directory if necessary
        dirname = os.path.dirname(self.file)
        print(dirname)
        os.makedirs(dirname, exist_ok=True)

        # Combine metadata and data and write to file
        with open(self.file, 'w+') as f:
            json.dump({'metadata': metadata, 'data': data}, f)

    # This function loads the data from a JSON file
    def load(self) -> tuple:
        """
        Description:
        This method loads metadata and data from a JSON file.

        Parameters:
        - None

        Returns:
        - tuple: A tuple containing metadata and data.
          This might be changed in the future when CPU/MPI/NCCL support is added.
        """
        # Read data from file
        with open(self.file, 'r') as f:
            data = json.load(f)

        # Try to return metadata and data
        try:
            return data['metadata'], data['data']
        except KeyError:
            print(f"WARN: file {self.file} appears to be corrupted. Ignoring.")
            return None, None
