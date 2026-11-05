# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 15:15:37 2025

@author: Iffy_z
"""

import numpy as np
from functools import wraps

def vectorize_integral(method):
    @wraps(method)
    def wrapper(self, a):
        if np.isscalar(a):
            return method(self,a)
        else:
            results = []
            for a_i in a:
                results.append(method(self,a_i))
            return np.array(results)
    return wrapper


