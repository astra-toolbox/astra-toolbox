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
from . import matrix_c as m

def delete(ids):
    """Delete a matrix object.
    
    :param ids: ID or list of ID's to delete.
    :type ids: :class:`int` or :class:`list`
    
    """
    return m.delete(ids)

def clear():
    """Clear all matrix objects."""
    return m.clear()

def create(data):
    """Create matrix object with data.
    
    :param data: Data to fill the created object with.
    :type data: :class:`scipy.sparse.csr_matrix`
    :returns: :class:`int` -- the ID of the constructed object.
    
    """
    return m.create(data)

    
def store(i,data):
    """Fill existing matrix object with data.
    
    :param i: ID of object to fill.
    :type i: :class:`int`
    :param data: Data to fill the object with.
    :type data: :class:`scipy.sparse.csr_matrix`
    
    """
    return m.store(i,data)

def get_size(i):
    """Get matrix dimensions.
    
    :param i: ID of object.
    :type i: :class:`int`
    :returns: :class:`tuple` -- matrix dimensions.
    """
    return m.get_size(i)
    
def get(i):
    """Get a matrix object.
    
    :param i: ID of object to get.
    :type i: :class:`int`
    :returns: :class:`scipy.sparse.csr_matrix` --  The object data.
    
    """
    return m.get(i)

def info():
    """Print info on matrix objects in memory."""
    return m.info()
    
        
