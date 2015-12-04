"""

This is an example script that uses image processing
operations such as segmentation and edge detection to
show how to implement custom algorithms
with distributed execution support in the ASTRA Toolbox.


"""
import astra
import numpy as np
import operator

import astra.mpi_c as mpi

import pylab


#Segmentation parameters
segment_rho = [0.0,1.0]
segment_tau = [0.5]
segment_iter  = 3
sirta_iter = 10
sirtb_iter = 5

#Function to segment data in tau_ar segments

#This version copies the data from astra buffers to numpy buffers
def segmentData_local(rho, tau_ar, src_id, dst_id):
    import numpy as np

    dataL = astra.data3d.get_local(src_id)     #Shared to save memory
    tau_ar2 = [-np.inf] + tau_ar + [np.inf]
    for i in range(len(tau_ar)+1):        
        dataL[(dataL > tau_ar2[i]) & (dataL <= tau_ar2[i+1])] = rho[i]
    astra.data3d.store_local(dst_id, dataL)     #Each process stores its own data

#This version gets the pointer of the astra data buffers and uses this 
#for numpy buffers. Thereby reducing memory and memcpy operations
def segmentData(rho, tau_ar, src_id, dst_id):
    import numpy as np

    dataL   = astra.data3d.get_shared(src_id)  #Shared to save memory
    dataDst = astra.data3d.get_shared(dst_id)   

    tau_ar2 = [-np.inf] + tau_ar + [np.inf]
    for i in range(len(tau_ar)+1):        
        dataDst[(dataL > tau_ar2[i]) & (dataL <= tau_ar2[i+1])] = rho[i]

#TODO EDGE
def markSegmentBorders2(source_id, dest_id):
    import numpy as np

    dataL = astra.data3d.get_local(source_id)

    X,Y,Z = dataL.shape
    dataL2 = np.zeros(dataL.shape, dtype=np.float32)
    for x in range(1,X-1):
        for y in range(1,Y-1):
            for z in range(1,Z-1):
               l = dataL[x,y,z]
               if ( l != dataL[x-1,y,z] or 
                  l != dataL[x+1,y,z] or 
                  l != dataL[x,y-1,z] or 
                  l != dataL[x,y+1,z] or 
                  l != dataL[x,y,z-1] or 
                  l != dataL[x,y,z+1]):
                   dataL2[x,y,z] =  1
    astra.data3d.store_local(dest_id, dataL2)

#This version uses get_local and requires more memory than the subMulArray
def subMulArray_local(src_id, src2_id, dst_id, value):
    dataL  = astra.data3d.get_local(src_id)
    dataL2 = astra.data3d.get_local(src2_id)
    dataL  = value-dataL
    dataL  = dataL*dataL2
    astra.data3d.store_local(dst_id, dataL)

#This version only uses pre-allocated memory
def subMulArray(src_id, src2_id, dst_id, value):
    dataS1    = astra.data3d.get_shared(src_id)
    dataS2    = astra.data3d.get_shared(src2_id)
    dataD     = astra.data3d.get_shared(dst_id)    
    dataD[:]  = value-dataS1
    dataD[:]  = dataD*dataS2

def opArray(src_id, src2_id, dst_id, op):
    dataS1   = astra.data3d.get_shared(src_id)
    dataS2   = astra.data3d.get_shared(src2_id)
    dataD    = astra.data3d.get_shared(dst_id)
    dataD[:] = op(dataS1,dataS2)


pylab.gray()
#Setup ASTRA logging
astra.log.setOutputScreen(astra.log.STDOUT,astra.log.DEBUG)
astra.log.setOutputFile("segment.txt", astra.log.DEBUG)

#Input dataset
Z = 128
Y = 192
X = 160

vol_geom = astra.create_vol_geom(Y, X, Z)
angles = np.linspace(0, np.pi, 180,False)
#proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 192, angles)
proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, X, Y+64, angles, 1000, 0)

# Create a simple hollow cube phantom
cube = np.zeros((Z,Y,X))
cube[17:Z-15,17:Y-15,17:X-15] = 1
cube[33:97,33:Y-31,33:X-31] = 0

#Setup the MPI domain distribution
mpi.create(proj_geom, vol_geom, nGhostcellsVolume = 1, nGhostcellsProjection= 0)

        
# Create projection data from this
proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)
# Create a data object for the reconstruction
rec_id = astra.data3d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('SIRT3D_CUDA')
#cfg = astra.astra_dict('CGLS3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId']     = proj_id

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Execute the SIRT algorithm for sirta_iter iterations
astra.algorithm.run(alg_id, sirta_iter)

data3 = astra.data3d.get(rec_id)
pylab.figure(1)
pylab.imshow(data3[:,:,65])

#Release SIRT memory
astra.algorithm.delete(alg_id)

mask_id     = astra.data3d.create('-vol', vol_geom)
seg_id      = astra.data3d.create('-vol', vol_geom)
tempVol_id  = astra.data3d.create('-vol', vol_geom)
tempPrj_id  = astra.data3d.create('-proj3d', proj_geom)


for iter in range(0,segment_iter):
    mpi.run(segmentData, [segment_rho, segment_tau, rec_id, seg_id])
    astra.data3d.sync(seg_id)

    #"Mark borders"
    mpi.run(markSegmentBorders2, [seg_id, mask_id])
    #Random mask pixels, not implemented in this example


    mpi.run(subMulArray,[mask_id, seg_id, tempVol_id, 1.0])
    astra.data3d.sync(tempVol_id)

    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['VolumeDataId']     = tempVol_id
    cfg['ProjectionDataId'] = tempPrj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    astra.algorithm.delete(alg_id)

    mpi.run(opArray,[proj_id, tempPrj_id, tempPrj_id, operator.sub])
    astra.data3d.sync(tempPrj_id)

    mpi.run(opArray,[rec_id, mask_id, rec_id, operator.mul])
    astra.data3d.sync(rec_id)

    #tempPrj_id = sino - FP ( (1-mask_id) * seg_id)  #sino is prj_id
    #x = rec_id * (mask) 


    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId']     = tempPrj_id
    cfg['option'] = {'ReconstructionMaskId' : mask_id }
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, sirtb_iter)
    astra.algorithm.delete(alg_id)

    mpi.run(opArray,[tempVol_id, rec_id, rec_id, operator.add])
    astra.data3d.sync(rec_id)
        
    #rec = tempVol_id + rec_id
    #rec = seg_id *(1-mask_id) + rec_id

    #Blurrrr, not implemented in this example

    
mpi.run(segmentData, [segment_rho, segment_tau, rec_id, rec_id])
data3 = astra.data3d.get(rec_id)

pylab.figure(2)
pylab.imshow(data3[:,:,65])

pylab.figure(3)
pylab.imshow(data3[65,:,:])
pylab.show()

idx = 65
fname = 'segment-%s-%s.txt' % (str(idx), str(MPI.COMM_WORLD.Get_size()))
data = data3[:,:,65]
np.savetxt(fname, data)



