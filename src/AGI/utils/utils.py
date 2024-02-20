import sys
import os

# Function used to read environment variables
def readEnvVar(varName: str, throw: bool = True) -> str:
    try:
        return os.environ[varName]
    except KeyError:
        errMsg = f"Environment variable {varName} not found."
        if throw:
            raise Exception(errMsg)
        else:
            print(f"WARNING: {errMsg}")
            return None

# Function used to check if DCGM is installed and python bindings are available
def checkDCGMImports() -> None:
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
            dcgm_bingings = os.path.join(os.environ['DCGM_HOME'], 'bindings', 'python3')
        # Look for DCGM_HOME in /usr/local
        elif os.path.exists('/usr/local/dcgm/bindings/python3'):
            dcgm_bindings = '/usr/local/dcgm/bindings/python3'
        # Throw error
        else:
            raise Exception('Unable to find DCGM_HOME. Please set DCGM_HOME environment variable to the location of the DCGM installation.')
        
        sys.path.append(dcgm_bindings)