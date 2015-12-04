#Example on how to use the SIRT plugin in an MPI enabled execution.
#The script runs both the built in SIRT algorithm as well as the 
#plugin version and compares their output.


import astra
import numpy as np

# Additional import to support multi-node exectution
import astra.mpi_c as mpi

import pylab
import six

import sirt3d_plugin
import time



if __name__ == '__main__':
    
    pylab.gray()
    #Setup ASTRA logging
    #astra.log.setOutputScreen(astra.log.STDOUT,astra.log.DEBUG)
    #astra.log.setOutputFile("sirt3d.txt", astra.log.DEBUG)

    X = 128

    vol_geom  = astra.create_vol_geom(X,X,X)
    angles    = np.linspace(0, np.pi, 180,False)
    proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 192, angles)

    # Create a simple hollow cube phantom
    cube = np.zeros((X,X,X))
    cube[17:113,17:113,17:113] = 1
    cube[33:97,33:97,33:97] = 0

    # Modify the geometry to support distributed execution and then proceed as before
    proj_geom, vol_geom = mpi.create(proj_geom, vol_geom)

    # Create projection data from this volume
    proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)

    # Display a single projection image
    import pylab
    pylab.gray()
    pylab.figure(1)
    pylab.imshow(proj_data[:,20,:])

    rec_id  = astra.data3d.create('-vol',  vol_geom)


    #Execute the plugin-based reconstruction

    # Register the plugin with ASTRA
    astra.plugin.register(sirt3d_plugin.SIRT3DPlugin)
    six.print_(astra.plugin.get_help('SIRT3D-PLUGIN'))
    cfg = astra.astra_dict('SIRT3D-PLUGIN')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId']     = proj_id
    alg_id = astra.algorithm.create(cfg)
    
    startP = time.time()
    astra.algorithm.run(alg_id, 100)
    astra.algorithm.delete(alg_id)
    endP   = time.time()

    
    #Store the result and then reset the storage buffer
    recPlugin = astra.data3d.get(rec_id)
    astra.data3d.store(rec_id,0)

    #Redo the same using the build-in SIRT algorithm
    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    startC = time.time()
    astra.algorithm.run(alg_id, 100)
    endC = time.time()
    astra.algorithm.delete(alg_id)


    # Get the result, display it and show the difference
    # between the plugin and buildin methods
    recBuild = astra.data3d.get(rec_id)

    pylab.figure(2)
    pylab.imshow(recPlugin[:,:,65])
    pylab.show()

    diff = recBuild-recPlugin
    pylab.imshow(diff[65,:,:])
    pylab.show()
    
    print("Max difference: ", np.max(np.abs(diff)))
    print("Plugin time: %s Build in: %s sec" % (str(endP-startP),
        str(endC-startC)))

    #Clean up the astra buffers
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
