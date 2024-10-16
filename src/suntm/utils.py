"""
Collection of shared utility functions for the suntm package.

"""

from __future__ import annotations

import logging


def setup_logging(self):
    self._logger = logging.getLogger("SunTopic")
    # Add console handler and set level to DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    # Add formatter to handler
    ch.setFormatter(formatter)
    # Add handler to logger
    self._logger.addHandler(ch)
