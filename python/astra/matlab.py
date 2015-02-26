#-----------------------------------------------------------------------
#Copyright 2013 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/pyastratoolbox/
#
#
#This file is part of the Python interface to the
#All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").
#
#The Python interface to the ASTRA Toolbox is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#The Python interface to the ASTRA Toolbox is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with the Python interface to the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------
"""This module implements a MATLAB-like interface to the ASTRA Toolbox.

Note that all functions are called with a :class:`string` as the first
argument, specifying the operation to perform. This un-pythonic way
is used to make transitioning from MATLAB code to Python code easier, as
the MATLAB interface uses the same type of method calling.

After an initial ``import astra``, these functions can be accessed in the
``astra.m`` module.

"""

from . import astra_c
from . import data2d_c
from . import data3d_c
from . import projector_c
from . import algorithm_c
from . import matrix_c
import numpy as np


def astra(command, *args):
    """MATLAB-like interface to the :mod:`astra.astra` module
    
    For example:
    
    ``astra.m.astra('use_cuda')`` -- Check if CUDA is enabled.
    
    """
    return getattr(astra_c, command)(*args)


def data2d(command, *args):
    """MATLAB-like interface to the :mod:`astra.data2d` module
    
    For example:
    
    ``astra.m.data2d('create',type,geometry,data)`` -- Create a 2D object.
    
    """
    return getattr(data2d_c, command)(*args)


def data3d(command, *args):
    """MATLAB-like interface to the :mod:`astra.data3d` module
    
    For example:
    
    ``astra.m.data3d('get',i)`` -- Get 3D object data.
    
    """
    return getattr(data3d_c, command)(*args)


def projector(command, *args):
    """MATLAB-like interface to the :mod:`astra.projector` module
    
    For example:
    
    ``astra.m.projector('volume_geometry',i)`` -- Get volume geometry.
    
    """
    return getattr(projector_c, command)(*args)


def matrix(command, *args):
    """MATLAB-like interface to the :mod:`astra.matrix` module
    
    For example:
    
    ``astra.m.matrix('delete',i)`` -- Delete a matrix.
    
    """
    return getattr(matrix_c, command)(*args)


def algorithm(command, *args):
    """MATLAB-like interface to the :mod:`astra.algorithm` module
    
    For example:
    
    ``astra.m.algorithm('run',i,1000)`` -- Run an algorithm with 1000 iterations.
    
    """
    if command == 'iterate':
        command = 'run'
    return getattr(algorithm_c, command)(*args)
