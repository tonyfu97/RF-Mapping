"""
Functions for handling file IOs.

Tony Fu, June 29, 2022
"""
import os


#######################################.#######################################
#                                                                             #
#                            DELETE_ALL_NPY_FILES                             #
#                                                                             #
###############################################################################
def delete_all_npy_files(dir):
    """Removes all numpy files in the directory."""
    for f in os.listdir(dir):
        if f.endswith('.npy'):
            os.remove(os.path.join(dir, f))


#######################################.#######################################
#                                                                             #
#                               CHECK_EXTENSION                               #
#                                                                             #
###############################################################################
def check_extension(file_name, extension):
    """
    Checks if file_name has the extension. If not, adds the extension.
    
    Parameters
    ----------
    file_name : str
        The name of the file.
    extension : str
        Extensions like ".pdf".
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    if not file_name.endswith(extension):
        file_name = file_name + extension
    
    return file_name
