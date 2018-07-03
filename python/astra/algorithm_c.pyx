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
    CReconstructionAlgorithm2D * dynamic_cast_recAlg2D "dynamic_cast<astra::CReconstructionAlgorithm2D*>" (CAlgorithm * )
    CReconstructionAlgorithm3D * dynamic_cast_recAlg3D "dynamic_cast<astra::CReconstructionAlgorithm3D*>" (CAlgorithm * )

cdef extern from "src/PythonPluginAlgorithm.h" namespace "astra":
    cdef cppclass CPluginAlgorithm:
        object getInstance()

cdef extern from *:
    CPluginAlgorithm * dynamic_cast_PluginAlg "dynamic_cast<astra::CPluginAlgorithm*>" (CAlgorithm * )


def create(config):
    cdef Config * cfg = utils.dictToConfig(six.b('Algorithm'), config)
    cdef CAlgorithm * alg
    alg = PyAlgorithmFactory.getSingletonPtr().create(cfg.self.getAttribute(six.b('type')))
    if alg == NULL:
        del cfg
        raise Exception("Unknown Algorithm.")
    if not alg.initialize(cfg[0]):
        del cfg
        del alg
        raise Exception("Unable to initialize Algorithm.")
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
    cdef CReconstructionAlgorithm3D * pAlg3D
    cdef CAlgorithm * alg = getAlg(i)
    cdef float32 res = 0.0
    pAlg2D = dynamic_cast_recAlg2D(alg)
    pAlg3D = dynamic_cast_recAlg3D(alg)
    if pAlg2D != NULL:
        if not pAlg2D.getResidualNorm(res):
            raise Exception("Operation not supported.")
    elif pAlg3D != NULL:
        if not pAlg3D.getResidualNorm(res):
            raise Exception("Operation not supported.")
    else:
        raise Exception("Operation not supported.")
    return res


def delete(ids):
    try:
        for i in ids:
            manAlg.remove(i)
    except TypeError:
        manAlg.remove(ids)


def get_plugin_object(algorithm_id):
    cdef CAlgorithm *alg
    cdef CPluginAlgorithm *pluginAlg
    alg = getAlg(algorithm_id)
    pluginAlg = dynamic_cast_PluginAlg(alg)
    if not pluginAlg:
        raise Exception("Not a plugin algorithm")
    return pluginAlg.getInstance()


def clear():
    manAlg.clear()


def info():
    six.print_(wrap_from_bytes(manAlg.info()))
