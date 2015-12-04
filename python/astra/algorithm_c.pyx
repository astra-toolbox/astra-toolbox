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



import log as logging
from mpi4py import MPI

cdef CAlgorithmManager * manAlg = <CAlgorithmManager * >PyAlgorithmManager.getSingletonPtr()

cdef extern from *:
    CReconstructionAlgorithm2D * dynamic_cast_recAlg "dynamic_cast<CReconstructionAlgorithm2D*>" (CAlgorithm * ) except NULL


def create(config):
    #MPI code
    comm  = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank  = comm.Get_rank()
    #END MPI code

   
    if rank > 0 and config == None:
        config = comm.bcast(config, root = 0)
    
    logging.debug("Create algorithm rank: " + str(rank) + " " + str(config))

    
    cdef Config * cfg = utils.dictToConfig(six.b('Algorithm'), config)
    cdef CAlgorithm * alg
    alg = PyAlgorithmFactory.getSingletonPtr().create(cfg.self.getAttribute(six.b('type')))

    if alg == NULL:
        del cfg
        raise Exception("Unknown algorithm.")

    #if the algorithm supports MPI, bcast the creation to the clients
    if alg.isMPICapable() and rank == 0 and size > 1:
        comm.bcast(201,    root = 0)
        config = comm.bcast(config, root = 0)

    if not alg.initialize(cfg[0]):
        del cfg
        del alg
        raise Exception("Algorithm not initialized.")

    #Ensure that both the master and client use the same object IDs
    idx = -1

    if alg.isMPICapable():
        if rank == 0:
            idx = manAlg.store(alg)
            idx = comm.bcast(idx,  root = 0)
        else:
            idx = comm.bcast(None, root = 0)
            idx = manAlg.store(alg, idx)
    else:
        idx = manAlg.store(alg)


    del cfg
    return idx

cdef CAlgorithm * getAlg(i) except NULL:
    cdef CAlgorithm * alg = manAlg.get(i)
    if alg == NULL:
        raise Exception("Unknown algorithm.")
    if not alg.isInitialized():
        raise Exception("Algorithm not initialized.")
    return alg


def run(i, iterations=0):
    #MPI code
    comm  = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank  = comm.Get_rank()
    #END MPI code

    if rank > 0: #Note the > 0 the root does the send below
        i, iterations = comm.bcast([i, iterations], root = 0)

    
    cdef CAlgorithm * alg = getAlg(i)
    
    if alg.isMPICapable() and rank == 0 and size > 1:
        comm.bcast(202,    root = 0)
        comm.bcast([i, iterations], root = 0)
        #i, iterations = comm.bcast([i, iterations], root = 0)

    cdef int its = iterations
    with nogil:
        alg.run(its)


def get_res_norm(i):
    print("="*78)
    print("TODO, algorithm get_res_norm ")
    print("="*78)
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
    #MPI code
    comm  = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank  = comm.Get_rank()
    if rank == 0 and size > 1:
        comm.bcast(203,    root = 0)
    ids = comm.bcast(ids, root = 0)
    #END MPI code

    try:
        for i in ids:
            manAlg.remove(i)
    except TypeError:
        manAlg.remove(ids)


def clear():
    #MPI code
    comm  = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank  = comm.Get_rank()
    if rank == 0 and size > 1:
        comm.bcast(204,    root = 0)
    manAlg.clear()


def info():
    six.print_(wrap_from_bytes(manAlg.info()))
