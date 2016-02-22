# -----------------------------------------------------------------------
# Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
#            2013-2016, CWI, Amsterdam
#
# Contact: astra@uantwerpen.be
# Website: http://sf.net/projects/astra-toolbox
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
#
# distutils: language = c++
# distutils: libraries = astra

import six
try:
    from six.moves import range
except ImportError:
    # six 1.3.0
    from six.moves import xrange as range

import numpy as np
import scipy.sparse as ss

from libcpp cimport bool

cimport PyMatrixManager
from .PyMatrixManager cimport CMatrixManager
from .PyIncludes cimport *
from .utils import wrap_from_bytes

cdef CMatrixManager * manM = <CMatrixManager * >PyMatrixManager.getSingletonPtr()


def delete(ids):
    try:
        for i in ids:
            manM.remove(i)
    except TypeError:
        manM.remove(ids)

def clear():
    manM.clear()

cdef int csr_matrix_to_astra(data,CSparseMatrix *mat) except -1:
    if isinstance(data,ss.csr_matrix):
        csrD = data
    else:
        csrD = data.tocsr()
    if not mat.isInitialized():
        raise Exception("Couldn't initialize data object.")
    if csrD.nnz > mat.m_lSize or csrD.shape[0] > mat.m_iHeight:
        raise Exception("Matrix too large to store in this object.")
    for i in range(len(csrD.indptr)):
        mat.m_plRowStarts[i] = csrD.indptr[i]
    for i in range(csrD.nnz):
        mat.m_piColIndices[i] = csrD.indices[i]
        mat.m_pfValues[i] = csrD.data[i]

cdef astra_to_csr_matrix(CSparseMatrix *mat):
    indptr = np.zeros(mat.m_iHeight+1,dtype=np.int)
    indices = np.zeros(mat.m_plRowStarts[mat.m_iHeight],dtype=np.int)
    data = np.zeros(mat.m_plRowStarts[mat.m_iHeight])
    for i in range(mat.m_iHeight+1):
        indptr[i] = mat.m_plRowStarts[i]
    for i in range(mat.m_plRowStarts[mat.m_iHeight]):
        indices[i] = mat.m_piColIndices[i]
        data[i] = mat.m_pfValues[i]
    return ss.csr_matrix((data,indices,indptr),shape=(mat.m_iHeight,mat.m_iWidth))

def create(data):
    cdef CSparseMatrix* pMatrix
    pMatrix = new CSparseMatrix(data.shape[0], data.shape[1], data.nnz)
    if not pMatrix.isInitialized():
        del pMatrix
        raise Exception("Couldn't initialize data object.")
    try:
        csr_matrix_to_astra(data,pMatrix)
    except:
        del pMatrix
        raise Exception("Failed to create data object.")

    return manM.store(pMatrix)

cdef CSparseMatrix * getObject(i) except NULL:
    cdef CSparseMatrix * pDataObject = manM.get(i)
    if pDataObject == NULL:
        raise Exception("Data object not found")
    if not pDataObject.isInitialized():
        raise Exception("Data object not initialized properly.")
    return pDataObject


def store(i,data):
    cdef CSparseMatrix * pDataObject = getObject(i)
    csr_matrix_to_astra(data,pDataObject)

def get_size(i):
    cdef CSparseMatrix * pDataObject = getObject(i)
    return (pDataObject.m_iHeight,pDataObject.m_iWidth)

def get(i):
    cdef CSparseMatrix * pDataObject = getObject(i)
    return astra_to_csr_matrix(pDataObject)

def info():
    six.print_(wrap_from_bytes(manM.info()))
