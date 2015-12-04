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
from libcpp.string cimport string

from .PyIncludes cimport *

cdef extern from "astra/AstraObjectManager.h" namespace "astra":
    cdef cppclass CProjector3DManager:
        string info()
        void clear()
        void remove(int i)
        int store(CProjector3D *)
        int store(CProjector3D *, int idx)
        CProjector3D * get(int i)
        int getIndex(CProjector3D *)


cdef extern from "astra/AstraObjectManager.h" namespace "astra::CProjector3DManager":
    cdef CProjector3DManager* getSingletonPtr()
