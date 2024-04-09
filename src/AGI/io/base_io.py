import os

# This class is the base class for all input/output classes
# It implements the basic functionality that all input/output classes should have


class BaseIO:
    def __init__(self, file: str, force_overwrite: bool = False) -> None:
        self.force_overwrite = force_overwrite
        self.file = file

    # This function checks if a file exists and if it does, it raises an exception
    # This should be called before any expensive operation as we want to fail fast!
    def check_overwrite(self) -> None:
        if self.force_overwrite:
            return  # Skip check if force overwrite is enabled

        if os.path.exists(self.file):
            raise FileExistsError(
                f"File {self.file} already exists! Use --force to force overwrite.")

    # Context manager functions
    # These functions allow the class to be used as a context manager
    # Example:
    # with BaseIO() as io:
    #     io.write(data)
    #     data = io.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
