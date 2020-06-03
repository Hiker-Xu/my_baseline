#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

def unNormalizeData(normalized_data, data_mean, data_std):
    T = normalized_data.shape[0]  # batch size
    D = data_mean.shape[0] 

    orig_data = np.hstack((np.zeros([T, 3]), normalized_data))

    #orig_data = np.zeros((T,D), dtype=np.float32)

    #orig_data[:, ] = 

    # Multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data
  
