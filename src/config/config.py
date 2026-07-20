"""
Module
-------
Configuration Loader

Author
------
Amit Pimpalkar

Organization
------------
RBU, Nagpur

Year
----
2026

Purpose
-------
Loads project configuration from YAML.

Usage
-----
Configuration is loaded once during application startup and
shared across all modules.

"""

from pathlib import Path

from typing import Dict

import yaml


class ConfigurationLoader:
    """
    Loads YAML configuration files.
    """

    def __init__(self, configuration_file: Path):

        self.configuration_file = configuration_file

    def load(self) -> Dict:
        """
        Reads configuration file.

        Returns
        -------
        Dictionary containing project configuration.
        """

        if not self.configuration_file.exists():

            raise FileNotFoundError(

                f"Configuration not found: {self.configuration_file}"

            )

        with open(

            self.configuration_file,

            "r",

            encoding="utf-8"

        ) as stream:

            configuration = yaml.safe_load(stream)

        return configuration