import astra
import numpy as np
import operator

import astra.mpi_c as mpi

from mpi4py import MPI 

import segment_plugin
import six

import pylab


"""

This is an example script that uses the SegmentationPlugin.


"""


if __name__=='__main__':

    pylab.gray()
    #Setup ASTRA logging
    astra.log.setOutputScreen(astra.log.STDOUT,astra.log.DEBUG)
    astra.log.setOutputFile("segment.txt", astra.log.DEBUG)


    #Segmentation parameters
    segment_rho   = [0.0,1.0]
    segment_tau   = [0.5]
    segment_iter  = 3
    sirta_iter = 10
    sirtb_iter = 5


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
    cube[33:97,33:Y-31,33:X-31]   = 0

    #Setup the MPI domain distribution
    GPUList = [0,1]
    proj_geom, vol_geom = mpi.create(proj_geom, vol_geom, nGhostcellsVolume = 1,
                                     nGhostcellsProjection= 0, GPUList = GPUList)

            
    # Create projection data from this
    proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)
    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)
   
    
    # Register the plugin with ASTRA
    astra.plugin.register(segment_plugin.SegmentationPlugin)
    six.print_(astra.plugin.get_help('SIMPLE-SEGMENT-PLUGIN'))


    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('SIMPLE-SEGMENT-PLUGIN')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId']     = proj_id
    cfg['option'] = {}
    cfg['option']['segment_rho']   = segment_rho
    cfg['option']['segment_tau']   = segment_tau
    cfg['option']['segment_iter']  = segment_iter
    cfg['option']['sirta_iter'] = sirta_iter
    cfg['option']['sirtb_iter'] = sirtb_iter


    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)

    data3 = astra.data3d.get(rec_id)

    idx = 65
    #print(data3[:,idx,23])
    #print(list(enumerate(data3[:,idx,23])))
    pylab.figure(1)
    pylab.imshow(data3[:,:,idx])

    pylab.figure(2)
    pylab.imshow(data3[idx,:,:])
    pylab.show()


    fname = 'segment-%s-%s.txt' % (str(idx), str(MPI.COMM_WORLD.Get_size()))
    data = data3[:,:,idx]
    #np.savetxt(fname, data)



