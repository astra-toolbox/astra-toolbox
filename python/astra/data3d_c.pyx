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

cimport cython

cimport PyData3DManager
from .PyData3DManager cimport CData3DManager

from .PyIncludes cimport *
import numpy as np

cimport numpy as np
np.import_array()

cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument

cimport utils
from .utils import wrap_from_bytes

from mpi4py import MPI

cimport PyProjector3DManager
from .PyProjector3DManager cimport CProjector3DManager

cdef CProjector3DManager * manmpi3d = <CProjector3DManager * >PyProjector3DManager.getSingletonPtr()
from .pythonutils import geom_size

import operator

import log as logging 

from six.moves import reduce

import mpi_c

cdef CData3DManager * man3d = <CData3DManager * >PyData3DManager.getSingletonPtr()

cdef extern from *:
    CFloat32Data3DMemory * dynamic_cast_mem "dynamic_cast<astra::CFloat32Data3DMemory*>" (CFloat32Data3D * ) except NULL

cdef extern from *:
    CMPIProjector3D * dynamic_cast_mpiprj "dynamic_cast<astra::CMPIProjector3D*>" (CProjector3D *) except NULL

cdef extern from "CFloat32CustomPython.h":
    cdef cppclass CFloat32CustomPython:
        CFloat32CustomPython(arrIn)

def create(datatype,geometry,data=None, link=False):
    #MPI Setup
    comm,size,rank = commSizeRank()
    pname = MPI.Get_processor_name()

    if rank == 0:
        comm.bcast(101, root = 0)

    geometry = comm.bcast(geometry, root = 0)
    datatype = comm.bcast(datatype, root = 0)

    cdef Config                     * cfg = NULL
    cdef CVolumeGeometry3D          * pGeometry
    cdef CProjectionGeometry3D      * ppGeometry
    cdef CFloat32Data3DMemory       * pDataObject3D
    cdef CConeProjectionGeometry3D  * pppGeometry
    cdef CMPIProjector3D            * mpiPrj
    cdef CFloat32CustomMemory       * pCustom

    mpiData3DIsDistributed = False #Needed for data recombination


    #Link will not work in MPI setup as the other clients don't have 
    #the numpy memory...
    if 'MPI' in geometry and link:
        print("The link feature is not (yet) supported in combination with MPI.")
        raise Exception('Geometry class not initialized.')


    if link and data.shape!=geom_size(geometry):
        raise Exception("The dimensions of the data do not match those specified in the geometry.")

    if datatype == '-vol':
        cfg = utils.dictToConfig(six.b('VolumeGeometry'), geometry)
        if 'MPI' in geometry:
           mpiPrj    = dynamic_cast_mpiprj(manmpi3d.get(geometry['MPI']))
           pGeometry = mpiPrj.getVolumeLocal()          #Geometry assigned to this process
           mpiData3DIsDistributed = True
        else:
            pGeometry = new CVolumeGeometry3D()
            if not pGeometry.initialize(cfg[0]):
                del cfg
                del pGeometry
                raise Exception('Geometry class not initialized.')
        if link:
            pCustom = <CFloat32CustomMemory*> new CFloat32CustomPython(data)
            pDataObject3D = <CFloat32Data3DMemory * > new CFloat32VolumeData3DMemory(pGeometry, pCustom)
        else:
            pDataObject3D = <CFloat32Data3DMemory * > new CFloat32VolumeData3DMemory(pGeometry)
        del cfg
        
        if not 'MPI' in geometry:
            del pGeometry
    elif datatype == '-sino' or datatype == '-proj3d':
        if 'MPI' in geometry:
            mpiPrj                 = dynamic_cast_mpiprj(manmpi3d.get(geometry['MPI']))
            ppGeometry             = mpiPrj.getProjectionLocal()
            mpiData3DIsDistributed = True
        else:
            cfg = utils.dictToConfig(six.b('ProjectionGeometry'), geometry)
            tpe = wrap_from_bytes(cfg.self.getAttribute(six.b('type')))
            if (tpe == "parallel3d"):
                ppGeometry = <CProjectionGeometry3D*> new CParallelProjectionGeometry3D();
            elif (tpe == "parallel3d_vec"):
                ppGeometry = <CProjectionGeometry3D*> new CParallelVecProjectionGeometry3D();
            elif (tpe == "cone"):
                ppGeometry = <CProjectionGeometry3D*> new CConeProjectionGeometry3D();
            elif (tpe == "cone_vec"):
                ppGeometry = <CProjectionGeometry3D*> new CConeVecProjectionGeometry3D();
            else:
                raise Exception("Invalid geometry type.")
            
            if not ppGeometry.initialize(cfg[0]):
                del cfg
                del ppGeometry
                raise Exception('Geometry class not initialized.')
        if link:
            pCustom = <CFloat32CustomMemory*> new CFloat32CustomPython(data)
            pDataObject3D = <CFloat32Data3DMemory * > new CFloat32ProjectionData3DMemory(ppGeometry, pCustom)
        else:
            pDataObject3D = <CFloat32Data3DMemory * > new CFloat32ProjectionData3DMemory(ppGeometry)
        
        if not 'MPI' in geometry:
            del ppGeometry
        del cfg
    elif datatype == "-sinocone":
        cfg = utils.dictToConfig(six.b('ProjectionGeometry'), geometry)
        pppGeometry = new CConeProjectionGeometry3D()
        if not pppGeometry.initialize(cfg[0]):
            del cfg
            del pppGeometry
            raise Exception('Geometry class not initialized.')
        if link:
            pCustom = <CFloat32CustomMemory*> new CFloat32CustomPython(data)
            pDataObject3D = <CFloat32Data3DMemory * > new CFloat32ProjectionData3DMemory(pppGeometry, pCustom)
        else:
            pDataObject3D = <CFloat32Data3DMemory * > new CFloat32ProjectionData3DMemory(pppGeometry)
    else:
        raise Exception("Invalid datatype.  Please specify '-vol' or '-proj3d'.")

    if not pDataObject3D.isInitialized():
        del pDataObject3D
        raise Exception("Couldn't initialize data object.")

    #End MPI Setup
    
    if mpiData3DIsDistributed:
        #Store a reference to the accompanying MPIProjector inside the volume object
        mpiPrj =  dynamic_cast_mpiprj(manmpi3d.get(geometry['MPI']))
        pDataObject3D.setMPIProjector3D(mpiPrj)

    if not link: fillDataObject(pDataObject3D, data)

    pDataObject3D.updateStatistics()

    #Ensure that both the master and client use the same object IDs
    idx = -1
    if rank == 0:
        idx = man3d.store(<CFloat32Data3D*>pDataObject3D)
        idx = comm.bcast(idx,  root = 0)
    else:
        idx = comm.bcast(None, root = 0)
        idx = man3d.store(<CFloat32Data3D*>pDataObject3D, idx)


    return idx

def get_geometry(i):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(i))
    cdef CFloat32ProjectionData3DMemory * pDataObject2
    cdef CFloat32VolumeData3DMemory * pDataObject3
    if pDataObject.getType() == THREEPROJECTION:
        pDataObject2 = <CFloat32ProjectionData3DMemory * >pDataObject
        geom = utils.configToDict(pDataObject2.getGeometry().getConfiguration())
    elif pDataObject.getType() == THREEVOLUME:
        pDataObject3 = <CFloat32VolumeData3DMemory * >pDataObject
        geom = utils.configToDict(pDataObject3.getGeometry().getConfiguration())
    else:
        raise Exception("Not a known data object")


    #Retrieve the index of an associated MPIProjector
    if pDataObject.hasMPIProjector3D():
        geom['MPI'] = manmpi3d.getIndex(<CProjector3D*>pDataObject.getMPIProjector3D())

    return geom

def change_geometry(i, geom):
    cdef CFloat32Data3DMemory           * pDataObject
    cdef CFloat32ProjectionData3DMemory * pDataObject2
    cdef CFloat32VolumeData3DMemory     * pDataObject3

    cdef CFloat32Data3DMemory           * pDataObject3DNew
    cdef CProjectionGeometry3D          * ppGeometryNew
    cdef CProjectionGeometry3D          * ppGeometryOri
    
    cdef CMPIProjector3D            * mpiPrjOri
    cdef CMPIProjector3D            * mpiPrjNew
    comm,size,rank = commSizeRank()


    if rank == 0: comm.bcast(109, root = 0)
    
    i    = comm.bcast(i, root = 0)
    geom = comm.bcast(geom, root = 0)

    pDataObject = dynamic_cast_mem(getObject(i))
    if pDataObject.hasMPIProjector3D() and pDataObject.getType() == THREEPROJECTION:

        if not 'MPI' in geom:
            raise Exception("The new geometry does not contain MPI information while the original geometry does. This is not supported.")

        #Retrieve geometry properties and compare that the new and old one got
        #the same global dimensions. Per process can be different.
        mpiPrjOri        = pDataObject.getMPIProjector3D()
        mpiPrjNew        = dynamic_cast_mpiprj(manmpi3d.get(geom['MPI']))
        ppGeometryOri    = mpiPrjOri.getProjectionLocal()  
        ppGeometryNew    = mpiPrjNew.getProjectionLocal()  
       
        if (ppGeometryOri.getDetectorColCount() != ppGeometryNew.getDetectorColCount() or \
            ppGeometryOri.getProjectionCount()  != ppGeometryNew.getProjectionCount() or \
            mpiPrjOri.getGlobalNumberOfSlices(rank,1) !=
            mpiPrjNew.getGlobalNumberOfSlices(rank,1) ):
            raise Exception(
                "The dimensions of the data do not match those specified in the geometry.")

        #Allocate a new memory object and add it to the data manager
        pDataObject3DNew = <CFloat32Data3DMemory * > new CFloat32ProjectionData3DMemory(ppGeometryNew)
        pDataObject3DNew.setMPIProjector3D(mpiPrjNew)
        newObjID = man3d.store(<CFloat32Data3D*>pDataObject3DNew) 

        srcData = get_shared_local(i)
        dstData = get_shared_local(newObjID)

        # Get our responsible area info which we use to send data to the new processes
        sliceInfoNew   = mpi_c.getObjectResponsibleSliceInfo(newObjID, rank)
        sliceInfo      = mpi_c.getObjectResponsibleSliceInfo(i, rank)
        nSlices        = sliceInfo[1]-sliceInfo[0]
        startSlice     = sliceInfo[2]
        startSliceIdx  = sliceInfo[0]
       
        #Lambda functions that convert global coordinates into local indices
        glb2LclOri     = (lambda x : x - sliceInfo[2]    + sliceInfo[0])
        glb2LclNew     = (lambda x : x - sliceInfoNew[2] + sliceInfoNew[0])

        sliceSends = []
        for idx in range(0, size):
            #Test for each process if the new section overlaps with our own data, 
            #if this is the case then determine which part exactly overlaps and 
            #store this. It will be send to the other processes later on.
            #Note this is all computed in global coordinates
            sliceInfoTarget  = mpi_c.getObjectResponsibleSliceInfo(newObjID, idx)
            nSlicesTarget    = sliceInfoTarget[1]-sliceInfoTarget[0]
            startSliceTarget = sliceInfoTarget[2]

            startRemote = max(startSliceTarget, startSlice)
            endRemote   = min(startSliceTarget+nSlicesTarget, startSlice + nSlices)
            overlap     = max(0, endRemote - startRemote)
            if overlap > 0 :
                sliceSends.append((startRemote, endRemote))
            else:
                sliceSends.append((None,None))

        #Send what we have and receive what we need
        combinedList  = comm.allgather(sliceSends)


        #Do the actual exchange, note we start the loop at 0 to include the local
        #data transfer. In this single loop all data is exchanged.
        for idx in range(0, size):
            src = (size + rank - idx) % size
            dst = (size + rank + idx) % size

            idxStartO = idxEndO = idxStartN = idxEndN = 0
            
            if combinedList[rank][dst][0] != None:
                idxStartO = glb2LclOri(combinedList[rank][dst][0])
                idxEndO   = glb2LclOri(combinedList[rank][dst][1])

            if combinedList[src][rank][0] != None:
                idxStartN = glb2LclNew(combinedList[src][rank][0])
                idxEndN   = glb2LclNew(combinedList[src][rank][1])

            dstData[idxStartN:idxEndN,:,:] = comm.sendrecv(srcData[idxStartO:idxEndO,:,:], 
                                             dest    = dst,   source  = src,
                                             sendtag = 54321, recvtag = 54321)
        #Delete the old data, change the index and update the ghostcells to get
        #back in a consistent state with the original objects IDs
        man3d.remove(i)
        man3d.change_index(newObjID, i)
        mpiPrjNew.pyExchangeGhostRegionsProjectionFull(pDataObject3DNew.getData())    
        return


    if pDataObject.getType() == THREEPROJECTION:
        pDataObject2 = <CFloat32ProjectionData3DMemory * >pDataObject
        # TODO: Reduce code duplication here
        cfg = utils.dictToConfig(six.b('ProjectionGeometry'), geom)
        tpe = wrap_from_bytes(cfg.self.getAttribute(six.b('type')))
        if (tpe == "parallel3d"):
            ppGeometry = <CProjectionGeometry3D*> new CParallelProjectionGeometry3D();
        elif (tpe == "parallel3d_vec"):
            ppGeometry = <CProjectionGeometry3D*> new CParallelVecProjectionGeometry3D();
        elif (tpe == "cone"):
            ppGeometry = <CProjectionGeometry3D*> new CConeProjectionGeometry3D();
        elif (tpe == "cone_vec"):
            ppGeometry = <CProjectionGeometry3D*> new CConeVecProjectionGeometry3D();
        else:
            raise Exception("Invalid geometry type.")
        if not ppGeometry.initialize(cfg[0]):
            del cfg
            del ppGeometry
            raise Exception('Geometry class not initialized.')
        del cfg
        if (ppGeometry.getDetectorColCount() != pDataObject2.getDetectorColCount() or \
            ppGeometry.getProjectionCount() != pDataObject2.getAngleCount() or \
            ppGeometry.getDetectorRowCount() != pDataObject2.getDetectorRowCount()):
            del ppGeometry
            raise Exception(
                "The dimensions of the data do not match those specified in the geometry.")
        pDataObject2.changeGeometry(ppGeometry)
        del ppGeometry

    elif pDataObject.getType() == THREEVOLUME:
        pDataObject3 = <CFloat32VolumeData3DMemory * >pDataObject
        cfg = utils.dictToConfig(six.b('VolumeGeometry'), geom)

        if 'MPI' in geom:
            mpiPrjNew =  dynamic_cast_mpiprj(manmpi3d.get(geom['MPI']))
            pGeometry =  mpiPrjNew.getVolumeLocal()
        else:
            pGeometry = new CVolumeGeometry3D()
            if not pGeometry.initialize(cfg[0]):
                del cfg
                del pGeometry
                raise Exception('Geometry class not initialized.')
            del cfg
        if (pGeometry.getGridColCount() != pDataObject3.getColCount() or \
            pGeometry.getGridRowCount() != pDataObject3.getRowCount() or \
            pGeometry.getGridSliceCount() != pDataObject3.getSliceCount()):
            del pGeometry
            raise Exception(
                "The dimensions of the data do not match those specified in the geometry.")
        pDataObject3.changeGeometry(pGeometry)
    
        if 'MPI' in geom:
            pDataObject.setMPIProjector3D(mpiPrjNew) #Replace the MPI object
        else:
            #Only delete the geometry if we are in a non MPI execution
            del pGeometry
    else:
        raise Exception("Not a known data object")


cdef fillDataObject(CFloat32Data3DMemory * obj, data2, isLocal = False):

    if obj.hasMPIProjector3D() and not isLocal:
        #MPI code
        comm,size,rank = commSizeRank()

        #Distribute this data over the MPI processes
        mpiPrj       = obj.getMPIProjector3D()
        mpiRowSizes  = []
        mpiRowStarts = []
        #Retrieve the domain distribution 
        dataType = 0 if obj.getType() == THREEVOLUME else 1
        for i in range(0, size): 
            mpiRowSizes .append(mpiPrj.getNumberOfSlices(i, dataType))
            mpiRowStarts.append(mpiPrj.getStartSlice    (i, dataType))

        data  = None
        dType = None
        
        #Array data is split. Scalar is send as is. Both using send/recv
        #dType = None -> scalar/None.  dType = 1 -> Array
        if isinstance(data2, np.ndarray): dType = 1 
        dType = comm.bcast(dType, root = 0)

        if rank == 0:
            if dType == 1:
                data = data2[mpiRowStarts[0]:mpiRowStarts[0]+mpiRowSizes[0],:,:]
            else:
                data = data2
            #Send to the other processes
            for idx in range(1,size):
                if dType == 1:
                    startR = mpiRowStarts[idx]
                    endR   = startR+mpiRowSizes[idx]
                    comm.send(data2[startR:endR,:,:], dest = idx, tag = 11)
                else:
                    comm.send(data2, dest = idx, tag = 11)
        else:
                data = comm.recv(source = 0, tag = 11)
    else:
        data = data2
    #End exchange

    if data is None:
        fillDataObjectScalar(obj, 0)
    else:
        if isinstance(data, np.ndarray):
            fillDataObjectArray(obj, np.ascontiguousarray(data,dtype=np.float32))
        else:
            fillDataObjectScalar(obj, np.float32(data))

cdef fillDataObjectScalar(CFloat32Data3DMemory * obj, float s):
    cdef int i
    for i in range(obj.getSize()):
        obj.getData()[i] = s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef fillDataObjectArray(CFloat32Data3DMemory * obj, float [:,:,::1] data):
    if (not data.shape[0] == obj.getDepth()) or (not data.shape[1] == obj.getHeight()) or (not data.shape[2] == obj.getWidth()):
        raise Exception(
            "The dimensions of the data do not match those specified in the geometry.")
    cdef float [:,:,::1] cView = <float[:data.shape[0],:data.shape[1],:data.shape[2]]> obj.getData3D()[0][0]
    cView[:] = data

cdef CFloat32Data3D * getObject(i) except NULL:
    cdef CFloat32Data3D * pDataObject = man3d.get(i)
    if pDataObject == NULL:
        raise Exception("Data object not found")
    if not pDataObject.isInitialized():
        raise Exception("Data object not initialized properly.")
    return pDataObject

@cython.boundscheck(False)
@cython.wraparound(False)
def get(i):
    #MPI code
    comm,size,rank = commSizeRank()
    if rank == 0 and size > 1:
        comm.bcast(102, root = 0)
    i = comm.bcast(i,   root = 0)
    #End MPI code
    cdef CMPIProjector3D            * mpiPrj

    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(i))
    outArr = np.empty((pDataObject.getDepth(),pDataObject.getHeight(), pDataObject.getWidth()),dtype=np.float32,order='C')
    cdef float [:,:,::1] mView = outArr
    cdef float [:,:,::1] cView = <float[:outArr.shape[0],:outArr.shape[1],:outArr.shape[2]]> pDataObject.getData3D()[0][0]
    mView[:] = cView

    
    #Combine local and remote data when MPI is used and initialized for this 
    #volume. Note: processes only send the data for which they are responsible
    #this prevents overlap and the root process can just concatenate them
    if pDataObject.hasMPIProjector3D():
      mpiPrj   = pDataObject.getMPIProjector3D()
      startIdx = 0
      endIdx   = 0

      if pDataObject.getType() == THREEVOLUME:
          startIdx = mpiPrj.getResponsibleVolStartIndex(rank)
          endIdx   = mpiPrj.getResponsibleVolEndIndex(rank)
      else:
          startIdx = mpiPrj.getResponsibleProjStartIndex(rank)
          endIdx   = mpiPrj.getResponsibleProjEndIndex(rank)

      if rank == 0:
        outArr = outArr[startIdx:endIdx]  #Production
        
        #outArr[:endIdx] = 0
        #outArr[startIdx:endIdx] = 0
        #outArr[endIdx:] = 0.5
        for sourceID in range(1,size):
            outArr2    = comm.recv(source = sourceID, tag = 12)
            outArr     = np.concatenate((outArr, outArr2), 0)
      else:
         comm.send(outArr[startIdx:endIdx], dest = 0, tag = 12)  #Production
         #outArr[endIdx:]   = 1
         #outArr[0:startIdx] = 0.75
         #outArr[startIdx:endIdx] = 0.0
         #comm.send(outArr[:], dest = 0, tag = 12) # For testing ghostcells

    return outArr

def get_shared(i):
    comm,size,rank = commSizeRank()
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(i))

    #Test if this data object is distributed or not, if it is we can not simply
    #return a piece of shared memory. So print an error message to warn the user
    if pDataObject.hasMPIProjector3D() and size > 1: 
        logging.error("You are requesting the shared pointer of a distributed data object. This does not work")
        raise Exception("Do not use get_shared in a distributed run, use get_shared_local or do a get/store combination")
    else:
        return get_shared_local(i)


def get_shared_local(i):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(i))
    cdef np.npy_intp shape[3]
    shape[0] = <np.npy_intp> pDataObject.getDepth()
    shape[1] = <np.npy_intp> pDataObject.getHeight()
    shape[2] = <np.npy_intp> pDataObject.getWidth()
    return np.PyArray_SimpleNewFromData(3,shape,np.NPY_FLOAT32,<void *>pDataObject.getData3D()[0][0])



def get_single(i):
    raise Exception("Not yet implemented")

def store(i,data):
    #MPI code
    comm,size,rank = commSizeRank()
    if rank == 0 and size > 1:
        comm.bcast(105, root = 0)
    i = comm.bcast(i,   root = 0)
    #End MPI code

    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(i))
    fillDataObject(pDataObject, data)

def dimensions(i):
    cdef CFloat32Data3D * pDataObject = getObject(i)
    return (pDataObject.getDepth(),pDataObject.getHeight(),pDataObject.getWidth())

def dimensions_global(i):
    #MPI code
    comm,size,rank = commSizeRank()
    if rank == 0 and size > 1:
        comm.bcast(107,   root = 0)
    i     = comm.bcast(i, root = 0)
    local = dimensions(i)
    res   = comm.allgather(local)
    #Combine the Z dimension over the processes
    z = 0
    for x in res:  z += x[0]
    return (z, res[0][1], res[0][2]) #Z, Y, X, like dimensions()


#Not a very useful function per se, more an example on how to get
#certain data
def dimensions_global_volume_geometry(i):
    #MPI code
    comm,size,rank = commSizeRank()
    if rank == 0 and size > 1:
        comm.bcast(106,   root = 0)
    i = comm.bcast(i, root = 0)
    #END MPI code
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(i))
    cdef CMPIProjector3D            * mpiPrj = NULL
    cdef CVolumeGeometry3D          * pGeometry = NULL 
    if pDataObject.hasMPIProjector3D():
      mpiPrj   = pDataObject.getMPIProjector3D()
      pGeometry = mpiPrj.getVolumeGlobal()
      #Same order as dimensions()
      return(
          pGeometry.getGridSliceCount(),
          pGeometry.getGridRowCount(),
          pGeometry.getGridColCount()
        )


    return dimensions(i)

def delete(ids):
    #MPI code
    comm,size,rank = commSizeRank()
    if rank == 0 and size > 1:
        comm.bcast(103,   root = 0)
    ids = comm.bcast(ids, root = 0)
    #END MPI code
    try:
        for i in ids:
            man3d.remove(i)
    except TypeError:
        man3d.remove(ids)

def clear():
    #MPI code
    comm,size,rank = commSizeRank()
    if rank == 0 and size > 1:
        comm.bcast(104,    root = 0)
    man3d.clear()

def info():
    six.print_(wrap_from_bytes(man3d.info()))


def commSizeRank():
    comm  = MPI.COMM_WORLD
    size  = comm.Get_size()
    rank  = comm.Get_rank()
    return comm,size,rank


#Exchange ghost regions of the supplied data object
def sync(i):
    #MPI code
    comm,size,rank = commSizeRank()
    if rank == 0 and size > 1:
        comm.bcast(108,   root = 0)
    i = comm.bcast(i, root = 0)
    #END MPI code

    cdef CMPIProjector3D      * mpiPrj
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(i))

    if pDataObject.hasMPIProjector3D():
        mpiPrj = pDataObject.getMPIProjector3D()
        
        if pDataObject.getType() == THREEVOLUME:
            #This is a volume data-buffer, sync the ghost regions
            mpiPrj.pyExchangeGhostRegionsVolume(pDataObject.getData())
        else:
            #Sync the Full ghost data for the projection data-buffer
            mpiPrj.pyExchangeGhostRegionsProjectionFull(pDataObject.getData())
            #mpiPrj.pyExchangeGhostRegionsProjection(pDataObject.getData())
    


#Specific local functions, these do NOT perform any MPI communication 
#or cause any sub functions to not perform the communication.

@cython.boundscheck(False)
@cython.wraparound(False)
def get_local(i):
    cdef CFloat32Data3DMemory * pDataObject = dynamic_cast_mem(getObject(i))
    outArr = np.empty((pDataObject.getDepth(),pDataObject.getHeight(), pDataObject.getWidth()),dtype=np.float32,order='C')
    cdef float [:,:,::1] mView = outArr
    cdef float [:,:,::1] cView = <float[:outArr.shape[0],:outArr.shape[1],:outArr.shape[2]]> pDataObject.getData3D()[0][0]
    mView[:] = cView
    return outArr


def store_local(i,data):
    cdef CFloat32Data3D * pDataObject = getObject(i)
    fillDataObject(dynamic_cast_mem(pDataObject), data, True)

