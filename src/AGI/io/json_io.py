# This class is used to handle JSON data input/output

# System imports
import json
import os

# AGI imports
from AGI.io.base_io import BaseIO

class JSONDataIO(BaseIO):
    def __init__(self, file: str, forceOverwrite: bool = False) -> None:
        # Call parent constructor
        super().__init__(file, forceOverwrite)

    # This function dumps the data to a JSON file
    def dump(self, metadata: dict, data: dict) -> None:
        # Create directory if necessary
        dirname = os.path.dirname(self.file)
        if not os.path.exists(dirname):
            # Create directory
            os.makedirs(dirname)

        # Combine metadata and data and write to file
        with open(self.file, 'w+') as f:
            json.dump({'metadata': metadata, 'data': data}, f)

    # This function loads the data from a JSON file
    def load(self) -> tuple:
        # Read data from file
        with open(self.file, 'r') as f:
            data = json.load(f)

        # Try to return metadata and data
        try:
            return data['metadata'], data['data']
        except KeyError:
            print(f"WARN: file {self.file} appears to be corrupted. Ignoring.")
            return None, None