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

from . import astra_c as a

def credits():
    """Print credits of the ASTRA Toolbox."""
    return a.credits()


def use_cuda():
    """Test if CUDA is enabled.
    
    :returns: :class:`bool` -- ``True`` if CUDA is enabled.
    """
    return a.use_cuda()

def set_gpu_index(idx, memory=0):
    """Set default GPU index to use.
    
    :param idx: GPU index
    :type idx: :class:`int`
    """
    a.set_gpu_index(idx, memory)

def get_gpu_info(idx=-1):
    """Get GPU info.
    
    :param idx: GPU index, or -1 for current device
    :type idx: :class:`int`
    :returns: :class:`str` -- GPU info
    """
    return a.get_gpu_info(idx)

def has_feature(feature):
    """Check a feature flag.

    These are used to check if certain functionality has been
    enabled at compile time, if new functionality is present, or if
    a backward-incompatible change is present.

    See include/astra/Features.h for a list.

    :param feature: The name of the feature
    :type feature: :class:`str`
    :returns: :class:`bool` -- The presence of the feature
    """
    return a.has_feature(feature)

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


