import astra
import numpy as np
import operator

import astra.mpi_c as mpi

import pylab

"""

This is a basic implementation of segmentation and edge detection
to present the functionality of the plugin-classes and 
the MPI layer implemented in ASTRA.

"""


class SegmentationPlugin(astra.plugin.base):
    """Example of an ASTRA plugin class

    Options:

    'segment_rho':   rho values for segment (list)
    'segment_tau':   tau values for segment (list)
    'segment_iter':  number of segment iterations
    'sirta_iter': number of initial sirt iterations
    'sirtb_iter': number of sirt iterations inside the segment iteration
    """
    # The astra_name variable defines the name to use to
    # call the plugin from ASTRA
    astra_name = "SIMPLE-SEGMENT-PLUGIN"

    def initialize(self,cfg, segment_rho, segment_tau, segment_iter, sirta_iter, sirtb_iter): 
        self.segment_rho   = segment_rho
        self.segment_tau   = segment_tau
        self.segment_iter  = segment_iter
        self.sirta_iter = sirta_iter
        self.sirtb_iter = sirtb_iter

        self.rec_id = cfg['ReconstructionDataId']
        self.prj_id = cfg['ProjectionDataId']
        
    def run(self, iters):
        print("Hello World, running the Segmentation plugin")


        # Set up the SIRT reconstruction and run it
        cfg = astra.astra_dict('SIRT3D_CUDA')
        #cfg = astra.astra_dict('CGLS3D_CUDA')
        cfg['ReconstructionDataId'] = self.rec_id
        cfg['ProjectionDataId']     = self.prj_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, self.sirta_iter)
        astra.algorithm.delete(alg_id)

        vol_geom  = astra.data3d.get_geometry(self.rec_id)
        proj_geom = astra.data3d.get_geometry(self.prj_id)

        mask_id     = astra.data3d.create('-vol', vol_geom)
        seg_id      = astra.data3d.create('-vol', vol_geom)
        tempVol_id  = astra.data3d.create('-vol', vol_geom)
        tempPrj_id  = astra.data3d.create('-proj3d', proj_geom)


        for iter in range(self.segment_iter):
            mpi.run(self.segmentData, [self.segment_rho, self.segment_tau, self.rec_id, seg_id])
            astra.data3d.sync(seg_id)

            print("Mark borders")
            mpi.run(self.markSegmentBorders2, [seg_id, mask_id])
            print("Mark borders, done")

            #TODO random mask pixels

            mpi.run(self.subMulArray,[mask_id, seg_id, tempVol_id, 1.0])
            astra.data3d.sync(tempVol_id)

            cfg = astra.astra_dict('FP3D_CUDA')
            cfg['VolumeDataId']     = tempVol_id
            cfg['ProjectionDataId'] = tempPrj_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            astra.algorithm.delete(alg_id)

            mpi.run(self.opArray,[self.prj_id, tempPrj_id, tempPrj_id, operator.sub])
            astra.data3d.sync(tempPrj_id)

            mpi.run(self.opArray,[self.rec_id, mask_id, self.rec_id, operator.mul])
            astra.data3d.sync(self.rec_id)

            #tempPrj_id = sino - FP ( (1-mask_id) * seg_id)  #sino is prj_id
            #x = rec_id * (mask) 


            cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ReconstructionDataId'] = self.rec_id
            cfg['ProjectionDataId']     = tempPrj_id
            cfg['option'] = {'ReconstructionMaskId' : mask_id }
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, self.sirtb_iter)
            astra.algorithm.delete(alg_id)

            mpi.run(self.opArray,[tempVol_id, self.rec_id, self.rec_id, operator.add])
            astra.data3d.sync(self.rec_id)
                
            #rec = tempVol_id + rec_id
            #rec = seg_id *(1-mask_id) + rec_id

        mpi.run(self.segmentData, [self.segment_rho, self.segment_tau, self.rec_id, self.rec_id])
        #End def run

    #Function to segment data in tau_ar segments
    #This version gets the pointer of the astra data buffers and uses this 
    #for numpy buffers. Thereby reducing memory and memcpy operations
    @staticmethod
    def segmentData(rho, tau_ar, src_id, dst_id):
        import numpy as np

        dataL   = astra.data3d.get_shared_local(src_id)  #Shared to save memory
        dataDst = astra.data3d.get_shared_local(dst_id)   

        tau_ar2 = [-np.inf] + tau_ar + [np.inf]
        for i in range(len(tau_ar)+1):        
            dataDst[(dataL > tau_ar2[i]) & (dataL <= tau_ar2[i+1])] = rho[i]

    #TODO EDGE
    @staticmethod
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

    #This version only uses pre-allocated memory
    @staticmethod
    def subMulArray(src_id, src2_id, dst_id, value):
        dataS1    = astra.data3d.get_shared_local(src_id)
        dataS2    = astra.data3d.get_shared_local(src2_id)
        dataD     = astra.data3d.get_shared_local(dst_id)    
        dataD[:]  = value-dataS1
        dataD[:]  = dataD*dataS2

    @staticmethod
    def opArray(src_id, src2_id, dst_id, op):
        dataS1   = astra.data3d.get_shared_local(src_id)
        dataS2   = astra.data3d.get_shared_local(src2_id)
        dataD    = astra.data3d.get_shared_local(dst_id)
        dataD[:] = op(dataS1,dataS2)

