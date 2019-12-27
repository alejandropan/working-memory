#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 18:02:08 2019

@author: alex
"""

import numpy as np
from scipy.signal import butter, lfilter, freqz, bessel, filtfilt
import matplotlib.pyplot as plt


def clean_noise(ss):
    """
    clean square digital signals with a bessel filter,
    ss is the raw data from ss = sr.read_sync(sl)
    """
    b, a = bessel(1, 0.2, 'low', analog=False)
    output_signal = filtfilt(b, a, ss, axis = 0)

    return output_signal