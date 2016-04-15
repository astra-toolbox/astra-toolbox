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

from . import astra_c as a

def credits():
    """Print credits of the ASTRA Toolbox."""
    return a.credits()


def use_cuda():
    """Test if CUDA is enabled.
    
    :returns: :class:`bool` -- ``True`` if CUDA is enabled.
    """
    return a.use_cuda()


def version(printToScreen=False):
    """Check version of the ASTRA Toolbox.
    
    :param printToScreen: If ``True``, print version string. If ``False``, return version integer.
    :type printToScreen: :class:`bool`
    :returns: :class:`string` or :class:`int` -- The version string or integer.
    
    """
    return a.version(printToScreen)

def set_gpu_index(idx, memory=0):
    """Set default GPU index to use.
    
    :param idx: GPU index
    :type idx: :class:`int`
    """
    a.set_gpu_index(idx, memory)

def delete(ids):
    """Delete an astra object.
    
    :param ids: ID or list of ID's to delete.
    :type ids: :class:`int` or :class:`list`
    
    """
    return a.delete(ids)

def info(ids):
    """Print info about an astra object.
    
    :param ids: ID or list of ID's to show.
    :type ids: :class:`int` or :class:`list`
    
    """
    return a.info(ids)


