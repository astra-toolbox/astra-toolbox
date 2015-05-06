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
# distutils: language = c++
# distutils: libraries = astra

from libcpp.string cimport string
from libcpp cimport bool
from libcpp.list cimport list
from libcpp.vector cimport vector

cdef extern from "astra/Globals.h" namespace "astra":
    ctypedef float float32
    ctypedef double float64
    ctypedef unsigned short int uint16
    ctypedef signed short int sint16
    ctypedef unsigned char uchar8
    ctypedef signed char schar8
    ctypedef int int32
    ctypedef short int int16

cdef extern from "astra/XMLNode.h" namespace "astra":
    cdef cppclass XMLNode:
        string getName()
        XMLNode addChildNode(string name)
        XMLNode addChildNode(string, string)
        void addAttribute(string, string)
        void addAttribute(string, float32)
        void addOption(string, string)
        bool hasOption(string)
        string getAttribute(string)
        list[XMLNode] getNodes()
        vector[float32] getContentNumericalArray()
        void setContent(double*, int, int, bool)
        void setContent(double*, int)
        string getContent()
        bool hasAttribute(string)

cdef extern from "astra/XMLDocument.h" namespace "astra":
    cdef cppclass XMLDocument:
        void saveToFile(string sFilename)
        XMLNode getRootNode()
        
cdef extern from "astra/XMLDocument.h" namespace "astra::XMLDocument":
    cdef XMLDocument *createDocument(string rootname)
