# Example showing the usage of the distributed read and write routines

import astra
import math
import numpy as np
import scipy
import pylab

import astra.mpi_c as mpi

import astraUtilFunctions
from  astraUtilFunctions import readDistributedData
from  astraUtilFunctions import writeDistributedData


filepath = '/media/Data1/PUFoam_17.7um/Projections/'
filename = 'APUFoam_17.7um__prev_0266_prj%04d.img'

#filepath  ="/media/Data1/PUFoam_17.7um/VolumeJB/"
#filename = "angle_testJB-%d"

nAngles         = 1022
downScaleFactor = 8
imSize          = [1000, 524]

detectorRowCount = np.round(imSize[1]/downScaleFactor)
detectorColCount = np.round(imSize[0]/downScaleFactor)

Z = np.round(imSize[1]/downScaleFactor) 
Y = nAngles
X = np.round(imSize[0]/downScaleFactor)


angles              = np.linspace(0, 204.4/180*np.pi, nAngles,False)

#image pixel size: 17.74 um
image_pixel_size    = 17.74e-6
#object to source: 82.52 mm
object_to_source    = 82.52e-3
#detector to source: 218.162 mm
detector_to_source  = 218.162e-3
originToSource      = (object_to_source/ image_pixel_size) / downScaleFactor
originToDetector    = 0


astra.log.setOutputScreen(astra.log.STDOUT,astra.log.DEBUG)
#astra.log.setOutputFile("bla.txt", astra.log.WARN)
astra.log.setOutputFile("realbla.txt", astra.log.DEBUG)

proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, detectorRowCount,
                detectorColCount, angles, originToSource, originToDetector)

postAlignment = 2.5 / downScaleFactor

proj_geom                = astra.functions.geom_2vec(proj_geom)
proj_geom['Vectors'][:,3:5] = proj_geom['Vectors'][:,3:5] + postAlignment *  proj_geom['Vectors'][:,6:8]


vx = detectorColCount;
vy = detectorColCount;
vz = detectorRowCount;

print('Object has size %d x %d x %d\n' % (vx, vy, vz))

vol_geom  = astra.create_vol_geom(vx,vy,vz)

#Setup the MPI domain distribution
proj_geom, vol_geom = mpi.create(proj_geom, vol_geom, nGhostcellsVolume = 5 , nGhostcellsProjection=0)

proj_id = astra.data3d.create('-proj3d', proj_geom)
rec_id  = astra.data3d.create('-vol', vol_geom)

#Use imSize2 when reading source images that have already been downScaled
imSize2 = [imSize[0]/downScaleFactor,imSize[1]/downScaleFactor]
imSize2 = imSize #Using the original source images

#let each process read their own subset of data
mpi.run(readDistributedData, ['angles', filepath, filename, imSize2, proj_id])


# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('SIRT3D_CUDA')
#cfg = astra.astra_dict('CGLS3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id


# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 150 iterations of the algorithm
astra.algorithm.run(alg_id, 150)

#Each process writes their own piece of result data
filepath = '/media/Data1/PUFoam_17.7um/VolumeJB/'
mpi.run(writeDistributedData, ['slices', filepath, "reconStruction-%d", rec_id])

# Get the result
rec = astra.data3d.get(rec_id)
pylab.figure(2,  figsize=(8, 6), dpi=100 )
#pylab.imshow(rec[40,:,:])

idx = int(320. / downScaleFactor)

pylab.imshow(rec[:,idx,:])
fig1 = pylab.gcf()
pylab.show()

from mpi4py import MPI
fname = 'foo-%s.png' % (str(MPI.COMM_WORLD.Get_size()))
fig1.savefig(fname)

#fname = 'skyscan-%s-%s.txt' % (str(idx), str(MPI.COMM_WORLD.Get_size()))
#data = rec[:,idx,:]
#np.savetxt(fname, data)

