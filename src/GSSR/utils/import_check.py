###############################################################
# Project: GPU Saturation Scorer
#
# File Name: import_check.py
#
# Description:
# This file implements functions to check if all required
# libraries are installed. This is useful for ensuring that
# all dependencies are met before running the application.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################


import sys
import os

def load_dcgm() -> None:
    """
    Description:
    This function is used to check if the DCGM library is installed
    and if the python bindings are available.

    Parameters:
    - None

    Returns:
    - None

    Notes:
    - This function will throw an error if the DCGM library is not found.

    """
     
    # Set-up DCGM library path
    try:
        # Check if DCGM is already in the path
        import pydcgm
        import DcgmReader
        import dcgm_fields
        import dcgm_structs
        import pydcgm
        import dcgm_structs
        import dcgm_fields
        import dcgm_agent
        import dcgmvalue

    except ImportError:
        # Look for DCGM_HOME variable
        if 'DCGM_HOME' in os.environ:
            dcgm_bindings = os.path.join(
                os.environ['DCGM_HOME'], 'bindings', 'python3')
        # Look for DCGM_HOME in /usr/local
        elif os.path.exists('/usr/local/dcgm/bindings/python3'):
            dcgm_bindings = '/usr/local/dcgm/bindings/python3'
        # Throw error
        else:
            sys.exit(
                'Unable to find DCGM_HOME. Please set DCGM_HOME environment variable to the location of the DCGM installation.')

        sys.path.append(dcgm_bindings)



def test_import(module_name: str, errmsg: str = None) -> None:
    """
    Description:
    This function is used to check if a module is installed.

    Parameters:
    - module_name: Name of the module to check
    - errmsg: Error message to display if the module is not found

    Returns:
    - None

    Notes:
    - This function will throw an error if the module is not found.

    """
    try:
        __import__(module_name)
    except ImportError:
        sys.exit(f"Error: Module {module_name} not found. \
                  Please make sure that all requirements are installed.")

def check_import_requirements():
    """
    Description:
    This function is used to check if all requirements are installed.

    Parameters:
    - None

    Returns:
    - None

    Notes:
    - test_import will throw an error if a module is not found.
    """
    test_import('pandas')
    test_import('matplotlib')
    test_import('numpy')
    test_import('seaborn')
    test_import('scipy')
