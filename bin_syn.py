#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:55:00 2019

@author: ibladmin
"""

"""
Script to get alf files from bin files from terminal

"""

from ibllib.io import extractors
from pathlib import Path
from brainbox.core import Bunch


def run_alf_extractors(session_path):
    """
    Extract camera timestamps from the sync matrix
    :param session_path: path to ap.bin file from  
    :return: no return command, alf files are created
    """
    
    extractors.ephys_fpga._get_main_probe_sync(session_path)
    
    
if __name__ == "__main__":
    # Map command line arguments to function arguments.
    run_alf_extractors(*sys.argv[1:])
