###############################################################
# Project: GPU Saturation Scorer
#
# File Name: base_io.py
#
# Description:
# This file contains the base class for IO operations.
# It provides a context manager interface to read and write data.
# It also provides a check_overwrite function to check if a file exists before writing.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################

import os
import sys

class BaseIO:
    """
    Description:
    This class provides a base class for IO operations. It provides a context manager interface
    to read and write data. It also provides a check_overwrite function to check if a file exists
    before writing.

    Attributes:
    - file: The file to read/write data to/from.
    - force_overwrite: A flag to force overwrite of the file.

    Methods:
    - check_overwrite(self): Check if the file exists and raise an exception if it does.
    - __enter__(self): Context manager enter function.
    - __exit__(self): Context manager exit function.

    Notes:
    - This class should be inherited by other IO classes.
    - The check_overwrite function should be called before any expensive operation to fail fast.
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
        self.force_overwrite = force_overwrite
        self.file = file

    def check_overwrite(self) -> None:
        """
        Description:
        This function checks if the file exists and raises an exception if it does. If the force_overwrite
        flag is enabled, this function is skipped.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - This function should be called before any expensive operation to fail fast.
        """
        if self.force_overwrite:
            return  # Skip check if force overwrite is enabled

        if os.path.exists(self.file):
            sys.exit(f"File {self.file} already exists! Use --force to force overwrite.")

    # Context manager functions
    # These functions allow the class to be used as a context manager
    # Example:
    # with BaseIO() as io:
    #     io.write(data)
    #     data = io.read()

    def __enter__(self):
        """
        Description:
        Context manager enter function.

        Parameters:
        - None

        Returns:
        - self

        Notes:
        - This function is called when the context manager is entered.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Description:
        Context manager exit function.

        Parameters:
        - exc_type: The exception type.
        - exc_value: The exception value.
        - traceback: The traceback object.

        Returns:
        - None

        Notes:
        - This function is called when the context manager is exited.
        """
        pass
