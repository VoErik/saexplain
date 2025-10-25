import time
import logging
from datetime import timedelta

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ExecTimer:
    """
    A context manager to log the execution time of a block of code.
    """
    def __init__(self, name="Execution Block"):
        self.name = name

    def __enter__(self):
        logging.info(f"Starting '{self.name}'...")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        elapsed_time = str(timedelta(seconds=duration))
        logging.info(f"Finished '{self.name}' in {elapsed_time}")

        return False