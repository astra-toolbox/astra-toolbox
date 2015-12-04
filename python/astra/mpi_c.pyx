
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

from mpi4py import MPI

cimport PyProjector3DManager
from .PyProjector3DManager cimport CProjector3DManager

#We use dill for pickling as it supports (lambda) functions
import dill

import inspect
import collections


cimport PyData3DManager
from .PyData3DManager cimport CData3DManager

cdef CProjector3DManager * manmpi3d = <CProjector3DManager * >PyProjector3DManager.getSingletonPtr()

cdef CData3DManager * man3d = <CData3DManager * >PyData3DManager.getSingletonPtr()

cdef extern from *:
    CFloat32Data3DMemory * dynamic_cast_mem "dynamic_cast<astra::CFloat32Data3DMemory*>" (CFloat32Data3D * ) except NULL

"""
    Utility function that checks if the ASTRA library was built with MPI Support. 
    If not we return False. 
    If with we return True

    For the multi-process execution it is required that ASTRA is compiled with
    the MPI support.
"""
def isBuiltWithMPI():
    cdef CMPIProjector3D * mpiPrj = NULL
    mpiPrj = new CMPIProjector3D()
    
    builtWithMPI = False
    #if ASTRA is not built with MPI support we cannot continue
    if mpiPrj.isBuiltWithMPI() :
        builtWithMPI = True
    del mpiPrj

    return builtWithMPI

"""
    create
    A call to this function setups the domain distribution. it requires two
    required arguments and 3 optional ones:
    Required:
        - prj_geom , the projection/detector geometry, a modified copy is returned
        - vol_geom , the volume geometry, a modified copy is returned
    Optional:
        - nGhostcellsVolume , the number of extra (overlapping) slices in the
          distributed sub-volumes
        - nGhostcellsProjection , the number of extra (overlapping) slices in
          the distributed sub-projection volumes
        - GPUList , a list of GPU-ids to be used. This list can be equal to the
          total number of GPUs or less. It will be accessed as follows:
            listIdx = procId % len(GPUList) 
            gpuIdx  = GPUList[listIdx] 
           
    Returns:
        - A modified prj_geom that still has the original dimensions but an extra 'MPI' property. 
        - A modified vol_geom that still has the original dimensions but an extra 'MPI' property. 


"""
def create(prj_geom, vol_geom, nGhostcellsVolume = 0, nGhostcellsProjection = 0,
        GPUList = None):
    geom = {}
    geom['ProjectionGeometry'] = prj_geom
    geom['VolumeGeometry']     = vol_geom
    #geom['nGhostcells']        = nGhostcells, gives unused warnings
    if GPUList:
        geom['GPUList'] = GPUList

    nGhostcells = [nGhostcellsVolume, nGhostcellsProjection]

    #MPI Setup
    comm  = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank  = comm.Get_rank()
    
    if size == 1:
        return prj_geom, vol_geom          #We are not running in a distributed setup
    if rank == 0:
        comm.bcast(401,    root = 0)
    geom        = comm.bcast(geom,        root = 0)
    nGhostcells = comm.bcast(nGhostcells, root = 0)
    #End setup

    cdef Config *cfg = NULL
    cfg = utils.dictToConfig(six.b('MPIProjector3D'), geom)
    cdef CMPIProjector3D * mpiPrj

    mpiPrj = new CMPIProjector3D()

    #if ASTRA is not built with MPI support we cannot continue
    if not mpiPrj.isBuiltWithMPI() :
        del cfg
        del mpiPrj
        return


    res = mpiPrj.initialize(cfg[0], nGhostcells[0], nGhostcells[1])
    #prjGlobal = mpiPrj.getProjectionGlobal()
    #prjLocal  = mpiPrj.getProjectionGlobal()
    #volGlobal = mpiPrj.getVolumeGlobal()
    #volLocal  = mpiPrj.getVolumeLocal()
    idx = -1
    if rank == 0:
        idx = manmpi3d.store(<CProjector3D*>mpiPrj)
        idx = comm.bcast(idx,  root = 0)
    else:
        idx = comm.bcast(None, root = 0)
        idx = manmpi3d.store(<CProjector3D*>mpiPrj, idx)

    del cfg

    if rank == 0:
        prj_geom2 = prj_geom.copy()
        vol_geom2 = vol_geom.copy()
        prj_geom2['MPI'] = idx
        vol_geom2['MPI'] = idx
        
        return prj_geom2, vol_geom2


"""
    This function executes a python command that is encoded using dill. 
    This is an internal function and should not be called by user code.
"""
def _runInternal(pyCmd, args):
    #MPI Setup
    comm  = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank  = comm.Get_rank()

    
    if rank == 0: comm.bcast(402,    root = 0)

    pyCmd, args = comm.bcast([pyCmd, args], root = 0)
    pyCmd       = dill.loads(pyCmd)
    args        = dill.loads(args)

    res = None

    if callable(pyCmd):
        res =  pyCmd(**args)  #This code is callable, so call it!
    else:
        exec(pyCmd, args)     #Execute this string using 'args' as environment

    if rank != 0:
        comm.send(res, dest = 0, tag = 4020)
    else:
        return res

"""
    run
    Run a python command issued by the root process on all processes using 
    the supplied arguments. This function is called by user code.
    Parameters:
        - code , a piece of code or a function that has to be executed
        - args , the arguments to be passed to the supplied function in 'code'
    returns
        - A list with per process the output of the supplied code. 
"""
def run(code, args = None):
    comm  = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank  = comm.Get_rank()

    argDict  = collections.OrderedDict()

    #if inspect.isfunction(code):
    if hasattr(code, "__call__"):
        #Put the function arguments in an (ordered) dictionary
        #also set the default arguments
        funcdefs = inspect.getargspec(code).defaults
        funcargs = inspect.getargspec(code).args
        nArgs    = len(funcargs)
        nDef     = 0 if funcdefs == None else len(funcdefs)

        for idx in range(nDef):
            argIdx = nArgs-nDef+idx
            argDict[funcargs[argIdx]] = funcdefs[idx]

        #Fill the required arguments with values
        if args is not None:
            #Use the supplied arguments to fill the arguments
            #Test that all required arguments have a value
            if ((nArgs-nDef) > len(args)):
                raise Exception(
                     "The number of supplied arguments is lower than the number of function arguments.")
            idx = 0
            for idx in range(nArgs):
                if idx < len(args):
                    argDict[funcargs[idx]] = args[idx]               #Set supplied
                else:
                    argDict[funcargs[idx]] = funcdefs[idx-len(args)] #Set default
    else:
        #It is not a function, make sure that the arguments are encoded as a dictionary
        if isinstance(args, dict):
            argDict = args.copy()

    pickleData  = dill.dumps(code)
    pickleData2 = dill.dumps(argDict)
    res = _runInternal(pickleData, pickleData2)

    result = [res]
    for i in range(1, size):
        temp = comm.recv(source = i, tag = 4020)
        result.append(temp)

    return result


# API functions to retrieve info about (distributed) objects

"""
    Utility function to retrieve an astra3d object 
    given its unique identifier.
"""
cdef CFloat32Data3D * getObject(i) except NULL:
    cdef CFloat32Data3D * pDataObject = man3d.get(i)
    if pDataObject == NULL:
        raise Exception("Data object not found")
    if not pDataObject.isInitialized():
        raise Exception("Data object not initialized properly.")
    return pDataObject



"""
    Returns the global slice informatino of a specified  astra3d object
    
    Returns a tuple of 3 values:
    0 : The start slice in global coordinates. So what the actual place is 
        of slice 0 in the full data-object
    1 : The number of slices in this object
    2 : The total number of slices in the global object
"""
def getObjectSliceInfo(objID):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(objID))
    cdef CMPIProjector3D            * mpiPrj

    rank  = MPI.COMM_WORLD.Get_rank()

    if pDataObject.hasMPIProjector3D():
        mpiPrj   = pDataObject.getMPIProjector3D()
        dataType = 0 if pDataObject.getType() == THREEVOLUME else 1
        nSlices  = mpiPrj.getNumberOfSlices(rank, dataType)
        startIdx = mpiPrj.getStartSlice    (rank, dataType)
        total    = mpiPrj.getGlobalNumberOfSlices(rank, dataType)
        return (startIdx, nSlices, total)
    else:
        return None

"""
    Returns the responsible region information as 
    a tuple of 4 integers. 

    Input:
    - objID , the astra3d object for which we require the information 
    - idx , the process for which we require this information. This is 
            the current process by default.
    
    The tuple contains:
    0 : the startSliceIndex >= 0
    1 : the endSliceIndex  < nSlices
    2 : the global startIndex, this is the global-slice index of the
    startSliceIndex
    3 : the total global number of slices

"""
def getObjectResponsibleSliceInfo(objID, idx = None):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(objID))
    cdef CMPIProjector3D            * mpiPrj

    rank  = MPI.COMM_WORLD.Get_rank()
    
    if idx == None:  idx = rank


    if pDataObject.hasMPIProjector3D():
        mpiPrj   = pDataObject.getMPIProjector3D()
        dataType = 0 if pDataObject.getType() == THREEVOLUME else 1

        if pDataObject.getType() == THREEVOLUME:
            startIdx    = mpiPrj.getResponsibleVolStartIndex(idx)
            endIdx      = mpiPrj.getResponsibleVolEndIndex(idx)
            glbStartIdx = mpiPrj.getResponsibleVolStart(idx) 
            total       = mpiPrj.getGlobalNumberOfSlices(idx, dataType)
        else:
            startIdx    = mpiPrj.getResponsibleProjStartIndex(idx)
            endIdx      = mpiPrj.getResponsibleProjEndIndex(idx)
            glbStartIdx = mpiPrj.getResponsibleProjStart(idx) 
            total       = mpiPrj.getGlobalNumberOfSlices(idx, dataType)
        return (startIdx, endIdx, glbStartIdx, total)
    else:
        return None

"""
  Returns the responsible region information as a tuple 
  of slice-objects which can be used directly to access
  (numpy) array objects.

"""
def getObjectResponsibleSlices(objID, idx = None):
    info = getObjectResponsibleSliceInfo(objID, idx)
    if info == None:
        return info

    sInfoZ  = slice(info[0], info[1])
    sinfoXY = slice(None, None)
    return (sInfoZ, sinfoXY, sinfoXY)



