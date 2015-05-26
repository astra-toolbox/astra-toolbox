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
from . import projector_c as p

def create(config):
    """Create projector object.

    :param config: Projector options.
    :type config: :class:`dict`
    :returns: :class:`int` -- the ID of the constructed object.

    """
    return p.create(config)


def delete(ids):
    """Delete a projector object.

    :param ids: ID or list of ID's to delete.
    :type ids: :class:`int` or :class:`list`

    """
    return p.delete(ids)


def clear():
    """Clear all projector objects."""
    return p.clear()


def info():
    """Print info on projector objects in memory."""
    return p.info()

def projection_geometry(i):
    """Get projection geometry of a projector.

    :param i: ID of projector.
    :type i: :class:`int`
    :returns: :class:`dict` -- projection geometry

    """
    return p.projection_geometry(i)


def volume_geometry(i):
    """Get volume geometry of a projector.

    :param i: ID of projector.
    :type i: :class:`int`
    :returns: :class:`dict` -- volume geometry

    """
    return p.volume_geometry(i)


def weights_single_ray(i, projection_index, detector_index):
    return p.weights_single_ray(i, projection_index, detector_index)


def weights_projection(i, projection_index):
    return p.weights_projection(i, projection_index)


def splat(i, row, col):
    return p.splat(i, row, col)

def is_cuda(i):
    """Check whether a projector is a CUDA projector.

    :param i: ID of projector.
    :type i: :class:`int`
    :returns: :class:`bool` -- True if the projector is a CUDA projector.

    """
    return p.is_cuda(i)


def matrix(i):
    """Get sparse matrix of a projector.

    :param i: ID of projector.
    :type i: :class:`int`
    :returns: :class:`int` -- ID of sparse matrix.

    """
    return p.matrix(i)
