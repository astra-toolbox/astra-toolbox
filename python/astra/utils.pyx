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

import numpy as np
import six
from libcpp.string cimport string
from libcpp.list cimport list
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from cpython.version cimport PY_MAJOR_VERSION

cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument
from .PyXMLDocument cimport XMLNode
from .PyIncludes cimport *


cdef XMLDocument * dict2XML(string rootname, dc):
    cdef XMLDocument * doc = PyXMLDocument.createDocument(rootname)
    cdef XMLNode * node = doc.getRootNode()
    try:
        readDict(node, dc)
    except:
        six.print_('Error reading XML')
        del doc
        doc = NULL
    finally:
        del node
    return doc

def convert_item(item):
    if isinstance(item, six.string_types):
        return item.encode('ascii')

    if type(item) is not dict:
        return item

    out_dict = {}
    for k in item:
        out_dict[convert_item(k)] = convert_item(item[k])
    return out_dict


def wrap_to_bytes(value):
    if isinstance(value, six.binary_type):
        return value
    s = str(value)
    if PY_MAJOR_VERSION == 3:
        s = s.encode('ascii')
    return s


def wrap_from_bytes(value):
    s = value
    if PY_MAJOR_VERSION == 3:
        s = s.decode('ascii')
    return s


cdef void readDict(XMLNode * root, _dc):
    cdef XMLNode * listbase
    cdef XMLNode * itm
    cdef int i
    cdef int j

    dc = convert_item(_dc)
    for item in dc:
        val = dc[item]
        if isinstance(val, np.ndarray):
            if val.size == 0:
                break
            listbase = root.addChildNode(item)
            listbase.addAttribute(< string > six.b('listsize'), < float32 > val.size)
            index = 0
            if val.ndim == 2:
                for i in range(val.shape[0]):
                    for j in range(val.shape[1]):
                        itm = listbase.addChildNode(six.b('ListItem'))
                        itm.addAttribute(< string > six.b('index'), < float32 > index)
                        itm.addAttribute( < string > six.b('value'), < float32 > val[i, j])
                        index += 1
                        del itm
            elif val.ndim == 1:
                for i in range(val.shape[0]):
                    itm = listbase.addChildNode(six.b('ListItem'))
                    itm.addAttribute(< string > six.b('index'), < float32 > index)
                    itm.addAttribute(< string > six.b('value'), < float32 > val[i])
                    index += 1
                    del itm
            else:
                raise Exception("Only 1 or 2 dimensions are allowed")
            del listbase
        elif isinstance(val, dict):
            if item == six.b('option') or item == six.b('options') or item == six.b('Option') or item == six.b('Options'):
                readOptions(root, val)
            else:
                itm = root.addChildNode(item)
                readDict(itm, val)
                del itm
        else:
            if item == six.b('type'):
                root.addAttribute(< string > six.b('type'), <string> wrap_to_bytes(val))
            else:
                itm = root.addChildNode(item, wrap_to_bytes(val))
                del itm

cdef void readOptions(XMLNode * node, dc):
    cdef XMLNode * listbase
    cdef XMLNode * itm
    cdef int i
    cdef int j
    for item in dc:
        val = dc[item]
        if node.hasOption(item):
            raise Exception('Duplicate Option: %s' % item)
        if isinstance(val, np.ndarray):
            if val.size == 0:
                break
            listbase = node.addChildNode(six.b('Option'))
            listbase.addAttribute(< string > six.b('key'), < string > item)
            listbase.addAttribute(< string > six.b('listsize'), < float32 > val.size)
            index = 0
            if val.ndim == 2:
                for i in range(val.shape[0]):
                    for j in range(val.shape[1]):
                        itm = listbase.addChildNode(six.b('ListItem'))
                        itm.addAttribute(< string > six.b('index'), < float32 > index)
                        itm.addAttribute( < string > six.b('value'), < float32 > val[i, j])
                        index += 1
                        del itm
            elif val.ndim == 1:
                for i in range(val.shape[0]):
                    itm = listbase.addChildNode(six.b('ListItem'))
                    itm.addAttribute(< string > six.b('index'), < float32 > index)
                    itm.addAttribute(< string > six.b('value'), < float32 > val[i])
                    index += 1
                    del itm
            else:
                raise Exception("Only 1 or 2 dimensions are allowed")
            del listbase
        else:
            node.addOption(item, wrap_to_bytes(val))

cdef vectorToNumpy(vector[float32] inp):
    cdef int i
    cdef int sz = inp.size()
    ret = np.empty(sz)
    for i in range(sz):
        ret[i] = inp[i]
    return ret

cdef XMLNode2dict(XMLNode * node):
    cdef XMLNode * subnode
    cdef list[XMLNode * ] nodes
    cdef list[XMLNode * ].iterator it
    dct = {}
    if node.hasAttribute(six.b('type')):
        dct['type'] = node.getAttribute(six.b('type'))
    nodes = node.getNodes()
    it = nodes.begin()
    while it != nodes.end():
        subnode = deref(it)
        if subnode.hasAttribute(six.b('listsize')):
            dct[subnode.getName(
                )] = vectorToNumpy(subnode.getContentNumericalArray())
        else:
            dct[subnode.getName()] = subnode.getContent()
        del subnode
    return dct

cdef XML2dict(XMLDocument * xml):
    cdef XMLNode * node = xml.getRootNode()
    dct = XMLNode2dict(node)
    del node;
    return dct;

cdef createProjectionGeometryStruct(CProjectionGeometry2D * geom):
    cdef int i
    cdef CFanFlatVecProjectionGeometry2D * fanvecGeom
    # cdef SFanProjection* p
    dct = {}
    dct['DetectorCount'] = geom.getDetectorCount()
    if not geom.isOfType(< string > six.b('fanflat_vec')):
        dct['DetectorWidth'] = geom.getDetectorWidth()
        angles = np.empty(geom.getProjectionAngleCount())
        for i in range(geom.getProjectionAngleCount()):
            angles[i] = geom.getProjectionAngle(i)
        dct['ProjectionAngles'] = angles
    else:
        raise Exception("Not yet implemented")
        # fanvecGeom = <CFanFlatVecProjectionGeometry2D*> geom
        # vecs = np.empty(fanvecGeom.getProjectionAngleCount()*6)
        # iDetCount = pVecGeom.getDetectorCount()
        # for i in range(fanvecGeom.getProjectionAngleCount()):
        #	p = &fanvecGeom.getProjectionVectors()[i];
        #	out[6*i + 0] = p.fSrcX
        #	out[6*i + 1] = p.fSrcY
        #	out[6*i + 2] = p.fDetSX + 0.5f*iDetCount*p.fDetUX
        #	out[6*i + 3] = p.fDetSY + 0.5f*iDetCount*p.fDetUY
        #	out[6*i + 4] = p.fDetUX
        #	out[6*i + 5] = p.fDetUY
        # dct['Vectors'] = vecs
    if (geom.isOfType(< string > six.b('parallel'))):
        dct["type"] = "parallel"
    elif (geom.isOfType(< string > six.b('fanflat'))):
        raise Exception("Not yet implemented")
        # astra::CFanFlatProjectionGeometry2D* pFanFlatGeom = dynamic_cast<astra::CFanFlatProjectionGeometry2D*>(_pProjGeom)
        # mGeometryInfo["DistanceOriginSource"] = mxCreateDoubleScalar(pFanFlatGeom->getOriginSourceDistance())
        # mGeometryInfo["DistanceOriginDetector"] =
        # mxCreateDoubleScalar(pFanFlatGeom->getOriginDetectorDistance())
        dct["type"] = "fanflat"
    elif (geom.isOfType(< string > six.b('sparse_matrix'))):
        raise Exception("Not yet implemented")
        # astra::CSparseMatrixProjectionGeometry2D* pSparseMatrixGeom =
        # dynamic_cast<astra::CSparseMatrixProjectionGeometry2D*>(_pProjGeom);
        dct["type"] = "sparse_matrix"
        # dct["MatrixID"] =
        # mxCreateDoubleScalar(CMatrixManager::getSingleton().getIndex(pSparseMatrixGeom->getMatrix()))
    elif(geom.isOfType(< string > six.b('fanflat_vec'))):
        dct["type"] = "fanflat_vec"
    return dct

cdef createVolumeGeometryStruct(CVolumeGeometry2D * geom):
    mGeometryInfo = {}
    mGeometryInfo["GridColCount"] = geom.getGridColCount()
    mGeometryInfo["GridRowCount"] = geom.getGridRowCount()

    mGeometryOptions = {}
    mGeometryOptions["WindowMinX"] = geom.getWindowMinX()
    mGeometryOptions["WindowMaxX"] = geom.getWindowMaxX()
    mGeometryOptions["WindowMinY"] = geom.getWindowMinY()
    mGeometryOptions["WindowMaxY"] = geom.getWindowMaxY()

    mGeometryInfo["option"] = mGeometryOptions
    return mGeometryInfo
