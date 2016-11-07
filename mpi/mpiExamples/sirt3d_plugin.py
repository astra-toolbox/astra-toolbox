import astra
import numpy as np
import operator

import astra.mpi_c as mpi


class SIRT3DPlugin(astra.plugin.base):
    """
        SIRT3D plugin class

        This is an example plugin that executes the SIRT iterative 
        algorithm using the buildin FP and BP operators.
        This algorithm is capable to be executed using multiple 
        processes by using the 'mpi.run' mechanism of the 
        ASTRA Toolbox.

    """

    # The astra_name variable defines the name to use to
    # call the plugin from ASTRA
    astra_name = "SIRT3D-PLUGIN"

    

    def initialize(self,cfg): 
        self.rec_id = cfg['ReconstructionDataId']
        self.prj_id = cfg['ProjectionDataId']
        self.precomputed = False
       
    def run(self, iters):
        #Retrieve the geometry and use it to allocate temporary buffers
        vol_geom  = astra.data3d.get_geometry(self.rec_id)
        proj_geom = astra.data3d.get_geometry(self.prj_id)

        pixelWeight_id = astra.data3d.create('-vol',  vol_geom)
        tmpVolume_id   = astra.data3d.create('-vol',  vol_geom)
        projData_id    = astra.data3d.create('-sino', proj_geom)
        lineWeight_id  = astra.data3d.create('-sino', proj_geom)

        #Compute the weights before we start the iteration steps
        self.precomputeWeights(pixelWeight_id, tmpVolume_id, projData_id, lineWeight_id)

        #Iterate
        for i in range(iters):
            #FP part
            astra.data3d.store(projData_id, 0)            
            self.performFP(projData_id, self.rec_id)
            mpi.run(self.opAddScaledMulScalar, [projData_id, self.prj_id, 1.0, -1.0])
            mpi.run(self.opMul,                [projData_id, lineWeight_id])

            #BP part
            astra.data3d.store(tmpVolume_id, 0)
            self.performBP(projData_id, tmpVolume_id)
            mpi.run(self.opAddMul, [self.rec_id, tmpVolume_id,pixelWeight_id])

            
    def precomputeWeights(self,pw_id, tmpV_id, prjD_id, lw_id):
        if self.precomputed:
            return

        #Compute and invert the lineweights
        astra.data3d.store(lw_id, 0)
        astra.data3d.store(tmpV_id, 1)
        self.performFP(lw_id, tmpV_id)
        mpi.run(self.opInvert, [lw_id])

        #Compute and invert the pixelweights
        astra.data3d.store(pw_id, 0)
        astra.data3d.store(prjD_id, 1)
        self.performBP(prjD_id, pw_id)
        mpi.run(self.opInvert, [pw_id])
        self.precomputed = True



    """
        The opInvert method inverts the values of an astra.data3d
        object. This is only applied on the local input data and the input 
        array is modified. Use mpi.run to run this function on all 
        processes.
    """
    @staticmethod 
    def opInvert(data_id):
        import numexpr
        data = astra.data3d.get_shared_local(data_id)
        thr = np.float32(0.000001)
        numexpr.evaluate("where(data > thr, 1 / data, 0)", out=data)
    
    """
        The opAddScaledMulscalar method adds the values of an multiplied astra.data3d
        object to an other astra.data3d object which is multiplied by an other scalar. 
    """
    @staticmethod 
    def opAddScaledMulScalar(dst_id, src_id, scale, scale2):
        import numexpr
        dst     = astra.data3d.get_shared_local(dst_id)
        src     = astra.data3d.get_shared_local(src_id)
        scale   = np.float32(scale)
        scale2  = np.float32(scale2)
        numexpr.evaluate("dst*scale2 + src*scale", out=dst)
        #dst[:]  = (dst*scale2) +  src*scale

    """
        The opMul method multiplies two astra.data3d objects. 
        Use mpi.run to run this function on all processes.
    """
    @staticmethod 
    def opMul(dst_id, src_id):
        dst     = astra.data3d.get_shared_local(dst_id)
        src     = astra.data3d.get_shared_local(src_id)
        dst[:] *= src
    

    """
        opAddMul adds two mulitplied astra.data3d
        objects to a third astra.data3d object.
    """
    @staticmethod 
    def opAddMul(dst_id, src_id1, src_id2):
        import numexpr
        dst     = astra.data3d.get_shared_local(dst_id)
        src1    = astra.data3d.get_shared_local(src_id1)
        src2    = astra.data3d.get_shared_local(src_id2)
        numexpr.evaluate("dst + src1*src2", out=dst)


    """
        Function that calls the built in FP algorithm
    """
    def performFP(self,proj_id, rec_id):
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['ProjectionDataId'] = proj_id
        cfg['VolumeDataId']     = rec_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        astra.algorithm.delete(alg_id)

    """
        Function that calls the built in BP algorithm
    """
    def performBP(self,proj_id, rec_id):
        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['ProjectionDataId']     = proj_id
        cfg['ReconstructionDataId'] = rec_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        astra.algorithm.delete(alg_id)

