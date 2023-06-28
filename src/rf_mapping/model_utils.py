"""
Utilities for working with pre-trained models.

Example usage
-------------
>>> model_info = ModelInfo()
>>> model_info.get_layer_names('alexnet')
['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
>>> model_info.get_layer_index('alexnet', 'conv3')
6
>>> model_info.get_num_units('alexnet', 'conv3')
384
>>> model_info.get_rf_size('alexnet', 'conv3')
99
>>> model_info.get_xn('alexnet', 'conv3')
127

Tony Fu, Bair Lab, Feb 2023
"""

import os
import sys

import pandas as pd

sys.path.append('../../..')
import src.rf_mapping.constants as c

__all__ = ['ModelInfo']

MODEL_INFO_FILE_PATH = os.path.join(c.REPO_DIR, "data", "model_info.txt")


class ModelInfo:
    """
    A class for loading and accessing model information from a CSV file.

    Attributes:
        model_info (pandas.DataFrame): The model information, loaded from the
        CSV file.

    """
    def __init__(self):
        """
        Initializes a new instance of the ModelInfo class.

        """
        self.model_info = self._load_model_info(MODEL_INFO_FILE_PATH)

    def _load_model_info(self, model_info_file_path: str) -> pd.DataFrame:
        """
        Loads the model info from the specified CSV file.

        Args:
            model_info_file_path (str): The path to the CSV file containing the
            model information.

        Returns:
            pandas.DataFrame: The model information, loaded from the CSV file.

        """
        model_info = pd.read_csv(model_info_file_path, delim_whitespace=True)
        model_info.columns = ["model", "layer", "layer_index", "rf_size", "xn", "num_units"]
        return model_info

    def get_layer_names(self, model_name: str) -> int:
        """
        Returns the names of all conv layers of the specified model.

        Args:
            model_name (str): The name of the model.

        Returns:
            list of string: The names of layers in the model.

        """
        return self.model_info.loc[(self.model_info['model'] == model_name)
                                   , 'layer']

    def get_layer_index(self, model_name: str, layer_name: str) -> int:
        """
        Returns the index of the specified layer.

        Args:
            model_name (str): The name of the model.
            layer_name (str): The name of the layer.

        Returns:
            int: The index of the specified layer in the model.

        """
        return self.model_info.loc[(self.model_info['model'] == model_name) &
                                   (self.model_info['layer'] == layer_name), 'layer_index'].iloc[0]

    def get_num_units(self, model_name: str, layer_name: str) -> int:
        """
        Returns the number of units in the specified layer.

        Args:
            model_name (str): The name of the model.
            layer_name (str): The name of the layer.

        Returns:
            int: The number of units in the specified layer.

        """
        return self.model_info.loc[(self.model_info['model'] == model_name) &
                                   (self.model_info['layer'] == layer_name), 'num_units'].iloc[0]

    def get_rf_size(self, model_name: str, layer_name: str) -> int:
        """
        Returns the receptive field size (without additional padding to ensure
        the center unit's RF is indeed centered) of the specified layer.

        Args:
            model_name (str): The name of the model.
            layer_name (str): The name of the layer.

        Returns:
            int: The receptive field size of the specified layer.

        """
        return self.model_info.loc[(self.model_info['model'] == model_name) &
                                   (self.model_info['layer'] == layer_name), 'rf_size'].iloc[0]

    def get_xn(self, model_name: str, layer_name: str) -> int:
        """
        Returns the receptive field size (plus additional padding to ensure the
        center unit's RF is indeed centered) of the specified layer.

        Args:
            model_name (str): The name of the model.
            layer_name (str): The name of the layer.

        Returns:
            int: The receptive field size of the specified layer.

        """
        return self.model_info.loc[(self.model_info['model'] == model_name) &
                                   (self.model_info['layer'] == layer_name), 'xn'].iloc[0]
