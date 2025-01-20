# -----------------------------------------------------------------------
# Copyright: 2010-2022, imec Vision Lab, University of Antwerp
#            2013-2022, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------

from . import data2d_c as d
from .pythonutils import checkArrayForLink


def clear():
    """Clear all 2D data objects."""
    return d.clear()

def delete(ids):
    """Delete a 2D object.

    :param ids: ID or list of ID's to delete.
    :type ids: :class:`int` or :class:`list`

    """
    return d.delete(ids)

def create(datatype, geometry, data=None):
    """Create a 2D object.

    :param datatype: Data object type, '-vol' or '-sino'.
    :type datatype: :class:`string`
    :param geometry: Volume or projection geometry.
    :type geometry: :class:`dict`
    :param data: Data to fill the constructed object with, either a scalar or array.
    :type data: :class:`float` or :class:`numpy.ndarray`
    :returns: :class:`int` -- the ID of the constructed object.

    """
    return d.create(datatype,geometry,data)

def link(datatype, geometry, data):
    """Link a 2D numpy array with the toolbox.

    :param datatype: Data object type, '-vol' or '-sino'.
    :type datatype: :class:`string`
    :param geometry: Volume or projection geometry.
    :type geometry: :class:`dict`
    :param data: Numpy array to link
    :type data: :class:`numpy.ndarray`
    :returns: :class:`int` -- the ID of the constructed object.

    """
    checkArrayForLink(data)
    return d.create(datatype,geometry,data,True)

def store(i, data):
    """Fill existing 2D object with data.

    :param i: ID of object to fill.
    :type i: :class:`int`
    :param data: Data to fill the object with, either a scalar or array.
    :type data: :class:`float` or :class:`numpy.ndarray`

    """
    return d.store(i, data)

def get_geometry(i):
    """Get the geometry of a 2D object.

    :param i: ID of object.
    :type i: :class:`int`
    :returns: :class:`dict` -- The geometry of object with ID ``i``.

    """
    return d.get_geometry(i)

def change_geometry(i, geom):
    """Change the geometry of a 2D object.

    :param i: ID of object.
    :type i: :class:`int`
    :param geom: new geometry.
    :type geom: :class:`dict`

    """
    return d.change_geometry(i, geom)

def get(i):
    """Get a 2D object.

    :param i: ID of object to get.
    :type i: :class:`int`
    :returns: :class:`numpy.ndarray` -- The object data.

    """
    return d.get(i)

def get_shared(i):
    """Get a 2D object with memory shared between the ASTRA toolbox and numpy array.

    :param i: ID of object to get.
    :type i: :class:`int`
    :returns: :class:`numpy.ndarray` -- The object data.

    """
    return d.get_shared(i)


def get_single(i):
    """Get a 2D object in single precision.

    :param i: ID of object to get.
    :type i: :class:`int`
    :returns: :class:`numpy.ndarray` -- The object data.

    """
    return d.get_single(i)

def shepp_logan(geometry, modified=True, returnData=True):
    """Create a 2D data object with a Shepp-Logan phantom.

    :param geometry: Volume geometry
    :param modified: If False, generate the original Shepp-Logan phantom
    :param returnData: If False, only return the ID of the new data object
    :returns: :class:`int` or (:class:`int`, :class`numpy.ndarray`)

    """
    i = create('-vol', geometry)
    d.shepp_logan(i, modified)
    if returnData:
      return i, get(i)
    else:
      return i

def info():
    """Print info on 2D objects in memory."""
    return d.info()
