#
#   Collection of utility functions for the ASTRA Python layer
#   possibly with MPI support.
#
#


import astra




#Each process reads all their buffer data (including ghostcells), 
#so not only the responsible area this eliminates the requirement of an 
#extra sync step.
def readDistributedData(dataformat, filepath, filename, imSize, dst_id):
    """ Read a (distributed) data set. Each process will read only the part of
    the dataset that it has to have available for reconstruction purposes. 
    There are two ways the data can be stored. Either one file per angle or one
    file per slice. The processes read all the data they are responsible for,
    including any ghostcell data.
    If the data is stored per angle, AND the data has to be downsized than each
    process reads the full file, downsamples this image and then makes a
    selection from that image. If there is no downsampling each process will
    only read the data that is required.


    :param  dataformat: Data source storage method, 'angles' or 'slices'. 
    :type dataformat: :class:`string`
    :param filepath: The folder in which the input data is stored
    :type filepath: :class:`string`
    :param filename: Filename pattern of the input files the '%d' will be
    replaced by the angle or slice number
    :type filename: :class:`string`
    :param imSize: The 2D dimensions of the source images. 
    :type imSize: :class:`list`
    :param dst_id: The astra.data3d object in which the data is saved
    :type dst_id: :class:`int`

    """
    import astra.mpi_c as mpi
    import numpy as np
    import scipy
    import os
    from mpi4py import MPI


    print("Reading a distributed fileset from %s " % (filepath))

    dims      = astra.data3d.dimensions(dst_id)

    #Figure out start-slice and number of slices and get the dimensions
    if MPI.COMM_WORLD.Get_size() > 1:
        sliceInfo = mpi.getObjectSliceInfo(dst_id)
    else:
        sliceInfo = [0, dims[0], dims[0]]  #Note local and full are the same


    if sliceInfo == None:
         astra.log.error("Unable to retrieve the distributed slice configuration for the supplied data object")
         raise Exception("Unable to retrieve the distributed slice configuration")

    startSlice  = sliceInfo[0]
    nSlicesLoc  = sliceInfo[1]
    nSlicesFull = sliceInfo[2]
    nAngles     = dims[1]
    P           = astra.data3d.get_shared_local(dst_id)

    #Compute read offset in bytes (number of slices to skip)
    offset = imSize[0]*startSlice*np.dtype(np.float32).itemsize
    items  = imSize[0]*nSlicesLoc


    if dataformat == 'angles':
        for angle in range(nAngles):
            fileName = (filepath+filename) % (angle,)
        
            if imSize[1] != nSlicesFull:
                #The data has to be downSamples to do this we have to read the fullset,
                #reduce it and then select our slices.
                img = np.fromfile(fileName, dtype=np.float32).reshape(imSize[1],imSize[0])
                img = scipy.misc.imresize(img, (nSlicesFull,dims[2]))
                P[:, angle, :] = img[startSlice:startSlice+nSlicesLoc, :]
            else:
                #No sampling required, jump the slices we do not need and only read our data
                f = open(fileName, "rb")
                f.seek(offset, os.SEEK_SET)        
                P[:, angle, :] = np.fromfile(f, dtype=np.float32, count=items).reshape(nSlicesLoc, imSize[0])
                f.close()
    elif dataformat == 'slices':
        for idx in range(startSlice, startSlice+nSlicesLoc):
            fileName = (filepath+filename) % (idx)
            P[idx-startSlice,:,:] = np.fromfile(fileName, dtype=np.float32).reshape(nAngles,imSize[1])
    else:
        raise("Unknown data-storage format specified")


def writeDistributedData(dataformat, filepath, filename, src_id):
    """ Write a (distributed) data set. Each process will write only the part of
    the dataset for which it is responsible. 
    There are two ways the data can be stored. Either one file per angle or one
    file per slice.
    If the data is to be stored per angle, then each process sends it angle 
    data to the root process (rank 0). This process then writes the data to
    file. When writing Volume data and the 'angles' option is specified then
    the data will be written per Y-slice.
    If the data is stored per slice, each process only has to write its 
    own data.


    :param  dataformat: Data source storage method, 'angles' or 'slices'. 
    :type dataformat: :class:`string`
    :param filepath: The folder in which the output data is to be stored
    :type filepath: :class:`string`
    :param filename: Filename pattern of the output files the '%d' will be
    replaced by the angle or slice number
    :type filename: :class:`string`
    :param src_id: The astra.data3d object to be stored on disk
    :type src_id: :class:`int`

    """
    import astra.mpi_c as mpi
    import numpy as np
    import scipy
    import os
    from mpi4py import MPI

    comm  = MPI.COMM_WORLD     
    size  = comm.Get_size() 
    rank  = comm.Get_rank()

    print("Writing a distributed fileset to %s " % (filepath))

        
    dims      = astra.data3d.dimensions(src_id)
    #Figure out start-slice and number of slices and get the dimensions
    if size > 1:
        sliceInfo = mpi.getObjectResponsibleSliceInfo(src_id)
    else:
        sliceInfo = [0, dims[0], dims[0]]


    if sliceInfo == None:
         astra.log.error("Unable to retrieve the distributed slice configuration for the supplied data object")
         raise Exception("Unable to retrieve the distributed slice configuration")

    startSlice  = sliceInfo[0]
    endSlice    = sliceInfo[1]
    startOffset = sliceInfo[2]
    nAngles     = dims[1]
    P           = astra.data3d.get_shared_local(src_id)
    
    
    if dataformat == 'angles':        
        #Gather data on process 0 and then write. Since we can not allow multiple
        #processes to write to the same file. 

        for angle in range(nAngles):
            if rank == 0:
                fileName = (filepath+filename) % (angle)
                outArr   = P[startSlice:endSlice, angle, :]
                for sourceID in range(1,size):
                    outArr2    = comm.recv(source = sourceID, tag = 212) 
                    outArr     = np.concatenate((outArr, outArr2), 0)   
                outArr[:,:].tofile(fileName)
            else:
                comm.send(P[startSlice:endSlice, angle, :], dest = 0, tag = 212)
    elif dataformat == 'slices':
        for idx in range(startSlice, endSlice):
            fileName = (filepath+filename) % (idx + (startOffset-startSlice))
            P[idx,:,:].tofile(fileName)
    else:
        raise("Unknown data-storage format specified")





def reduceValue(src_id, localOp):
    import astra.mpi_c as ampi
    if localOp == None: raise ValueError("No local operation requested.")

    srcData = astra.data3d.get_shared_local(src_id)
    return    localOp(srcData[ampi.getObjectResponsibleSlices(src_id)])


