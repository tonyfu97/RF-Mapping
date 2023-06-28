import logging
import datetime
import os
import sys

sys.path.append('../..')
import src.rf_mapping.constants as c

all = ['log_file']

def get_logger(file_path, caller_file):
    """
    Example usage
    -------------
    from my_logging_module import get_logger
    logger = get_logger('/path/to/mylogfile.log', __file__)
    logger.info('This is an info message.')
    """
    script_name = os.path.basename(caller_file)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.basicConfig(filename=file_path, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info(f'{script_name} was run at {current_time}')
    return logging
