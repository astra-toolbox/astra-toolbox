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

from . import algorithm_c as a

def create(config):
    """Create algorithm object.
    
    :param config: Algorithm options.
    :type config: :class:`dict`
    :returns: :class:`int` -- the ID of the constructed object.
    
    """
    return a.create(config)

def run(i, iterations=1):
    """Run an algorithm.
    
    :param i: ID of object.
    :type i: :class:`int`
    :param iterations: Number of iterations to run.
    :type iterations: :class:`int`
    
    """
    return a.run(i,iterations)

def get_res_norm(i):
    """Get residual norm of algorithm.
    
    :param i: ID of object.
    :type i: :class:`int`
    :returns: :class:`float` -- The residual norm.
    
    """
    
    return a.get_res_norm(i)
    
def delete(ids):
    """Delete a matrix object.
    
    :param ids: ID or list of ID's to delete.
    :type ids: :class:`int` or :class:`list`
    
    """
    return a.delete(ids)

def clear():
    """Clear all matrix objects."""
    return a.clear()

def info():
    """Print info on matrix objects in memory."""
    return a.info()
