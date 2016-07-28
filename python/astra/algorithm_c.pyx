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
from .PyIncludes cimport *

cimport PyAlgorithmManager
from .PyAlgorithmManager cimport CAlgorithmManager

cimport PyAlgorithmFactory
from .PyAlgorithmFactory cimport CAlgorithmFactory

cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument

cimport utils
from .utils import wrap_from_bytes

cdef CAlgorithmManager * manAlg = <CAlgorithmManager * >PyAlgorithmManager.getSingletonPtr()

cdef extern from *:
    CReconstructionAlgorithm2D * dynamic_cast_recAlg "dynamic_cast<CReconstructionAlgorithm2D*>" (CAlgorithm * ) except NULL


def create(config):
    cdef Config * cfg = utils.dictToConfig(six.b('Algorithm'), config)
    cdef CAlgorithm * alg
    alg = PyAlgorithmFactory.getSingletonPtr().create(cfg.self.getAttribute(six.b('type')))
    if alg == NULL:
        del cfg
        raise Exception("Unknown algorithm.")
    if not alg.initialize(cfg[0]):
        del cfg
        del alg
        raise Exception("Algorithm not initialized.")
    del cfg
    return manAlg.store(alg)

cdef CAlgorithm * getAlg(i) except NULL:
    cdef CAlgorithm * alg = manAlg.get(i)
    if alg == NULL:
        raise Exception("Unknown algorithm.")
    if not alg.isInitialized():
        raise Exception("Algorithm not initialized.")
    return alg


def run(i, iterations=0):
    cdef CAlgorithm * alg = getAlg(i)
    cdef int its = iterations
    with nogil:
        alg.run(its)


def get_res_norm(i):
    cdef CReconstructionAlgorithm2D * pAlg2D
    cdef CAlgorithm * alg = getAlg(i)
    cdef float32 res = 0.0
    pAlg2D = dynamic_cast_recAlg(alg)
    if pAlg2D == NULL:
        raise Exception("Operation not supported.")
    if not pAlg2D.getResidualNorm(res):
        raise Exception("Operation not supported.")
    return res


def delete(ids):
    try:
        for i in ids:
            manAlg.remove(i)
    except TypeError:
        manAlg.remove(ids)


def clear():
    manAlg.clear()


def info():
    six.print_(wrap_from_bytes(manAlg.info()))
