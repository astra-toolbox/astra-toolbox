# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
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

from libcpp.string cimport string

from .PyIncludes cimport *

cdef extern from "astra/AstraObjectManager.h" namespace "astra":
    cdef cppclass CMatrixManager:
        string info()
        void clear()
        void remove(int i)
        int store(CSparseMatrix *)
        CSparseMatrix * get(int i)

cdef extern from "astra/AstraObjectManager.h" namespace "astra::CMatrixManager":
    cdef CMatrixManager* getSingletonPtr()
