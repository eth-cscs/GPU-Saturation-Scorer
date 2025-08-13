##############################################################
# Project: GPU Saturation Scorer
#
# File Name: binary_io.py
#
# Description:
# This file contains the class to handle binary data input/output.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################

import pickle
import os
from GSSR.io.base_io import BaseIO

# This class is used to handle JSON data input/output
class BinaryDataIO(BaseIO):
    """
    Description:
    This class provides a context manager interface to read and write binary data.
    It inherits from the BaseIO class.

    Attributes:
    - file: The file to read/write data to/from.
    - force_overwrite: A flag to force overwrite of the file.

    Methods:
    - dump(self, metadata: dict, data: dict): Dump metadata and data to a binary file.
    - load(self) -> tuple: Load metadata and data from a binary file.

    Notes:
    - Under the hood, this class uses the pickle module to serialize and deserialize data.
    """
    def __init__(self, file: str, force_overwrite: bool = False) -> None:
        """
        Description:
        Constructor method.

        Parameters:
        - file: The file to read/write data to/from.
        - force_overwrite: A flag to force overwrite of the file.

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
        This method dumps metadata and data to a binary file.

        Parameters:
        - metadata: A dictionary containing metadata.
        - data: A dictionary containing data.

        Returns:
        - None

        Notes:
        - This method writes metadata and data to a binary file using the pickle module.
        - The data dictionary should contain a mapping from GPU ID to a dictionary containing metric: values pairs.
          This is important as the GPU IDs are NOT stored explicitly in the metadata.
        """
        # Create directory if necessary
        dirname = os.path.dirname(self.file)
        os.makedirs(dirname, exist_ok=True)

        # Combine metadata and data and write to file
        with open(self.file, 'wb+') as f:
            pickle.dump({'metadata': metadata, 'data': data}, f)

    # This function loads the data from a JSON file
    def load(self) -> tuple:
        """
        Description:
        This method loads metadata and data from a binary file.

        Parameters:
        - None

        Returns:
        - A tuple containing metadata and data.
        """

        # Read data from file
        with open(self.file, 'rb') as f:
            data = pickle.load(f)

        # Try to return metadata and data
        try:
            return data['metadata'], data['data']
        except KeyError:
            print(f"WARN: file {self.file} appears to be corrupted. Ignoring.")
            return None, None
