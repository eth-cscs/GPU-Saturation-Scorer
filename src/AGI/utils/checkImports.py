import sys
import os

def checkDCGMImports():
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