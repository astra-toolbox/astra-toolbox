import sys
import os
import codecs

vcppguid = "8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942" # C++ project
siguid = "2150E333-8FDC-42A3-9474-1A3956D46DE8" # project group 

# to generate a new uuid:
#
# import uuid
# uuid.uuid4().__str__().upper()


# see configure.ac
CUDA_CC = {
  (9,0): "compute_30,sm_30;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_70,compute_70",
  (9,2): "compute_30,sm_30;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_70,compute_70",
  (10,0): "compute_30,sm_30;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_75,compute_75",
  (10,1): "compute_30,sm_30;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_75,compute_75",
  (10,2): "compute_30,sm_30;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_75,compute_75",
  (11,0): "compute_35,sm_35;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_80,compute_80",
  (11,1): "compute_35,sm_35;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (11,2): "compute_35,sm_35;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (11,3): "compute_35,sm_35;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (11,4): "compute_35,sm_35;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (11,5): "compute_35,sm_35;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (11,6): "compute_35,sm_35;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (11,7): "compute_35,sm_35;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (11,8): "compute_35,sm_35;compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (12,0): "compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (12,1): "compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (12,2): "compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_86,compute_86",
  (12,3): "compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_87,sm_87;compute_89,sm_89;compute_90,sm_90;compute_90,compute_90",
  (12,4): "compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_87,sm_87;compute_89,sm_89;compute_90,sm_90;compute_90,compute_90",
  (12,5): "compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_87,sm_87;compute_89,sm_89;compute_90,sm_90;compute_90,compute_90",
  (12,8): "compute_50,sm_50;compute_60,sm_60;compute_70,sm_70;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_87,sm_87;compute_89,sm_89;compute_90,sm_90;compute_100,sm_100;compute_101,sm_101;compute_120,sm_120;compute_120,compute_120",
}

def create_mex_project(name, uuid14):
    return { "type": vcppguid, "name": name, "file14": name + "_vc14.vcxproj", "uuid14": uuid14, "files": [] }

P_astra = { "type": vcppguid, "name": "astra_vc14", "file14": "astra_vc14.vcxproj", "uuid14": "DABD9D82-609E-4C71-B1CA-A41B07495290" }

P0 = create_mex_project("astra_mex", "6FDF72C4-A855-4F1C-A401-6500040B5E28")

P1 = create_mex_project("astra_mex_algorithm", "CE5EF874-830C-4C10-8651-CCA2A34ED9E4")
P2 = create_mex_project("astra_mex_data2d", "D2CDCDB3-7AD9-4853-8D87-BDB1DAD9C9C1")
P3 = create_mex_project("astra_mex_data3d", "2A7084C6-62ED-4235-85F4-094C17689DEB")
P4 = create_mex_project("astra_mex_matrix", "6BFA8857-37EB-4E43-A97C-B860E21599F5")
P5 = create_mex_project("astra_mex_projector", "85ECCF1D-C5F6-4E0E-A4F9-0DE7C0B916B2")
P6 = create_mex_project("astra_mex_projector3d", "CA85BDA0-9BDD-495E-B200-BFE863EB6318")
P7 = create_mex_project("astra_mex_log", "88539382-66DB-4BBC-A48E-8B6B3CA6064F")
P8 = create_mex_project("astra_mex_direct", "47460476-912B-4313-8B10-BDF1D60A84C4")

F_astra_mex = { "type": siguid,
                "name": "astra_mex",
                "file14": "astra_mex",
                "uuid14": "2076FB73-ECFE-4B1B-9A8C-E351C500FAAB",
                "entries": [ P0, P1, P2, P3, P4, P5, P6, P7, P8 ] }


P0["files"] = [
"astra_mex_c.cpp",
"mexHelpFunctions.cpp",
"mexHelpFunctions.h",
"mexInitFunctions.cpp",
"mexInitFunctions.h",
]
P1["files"] = [
"astra_mex_algorithm_c.cpp",
"mexHelpFunctions.cpp",
"mexHelpFunctions.h",
"mexInitFunctions.cpp",
"mexInitFunctions.h",
]
P2["files"] = [
"astra_mex_data2d_c.cpp",
"mexHelpFunctions.cpp",
"mexHelpFunctions.h",
"mexCopyDataHelpFunctions.cpp",
"mexCopyDataHelpFunctions.h",
"mexDataManagerHelpFunctions.cpp",
"mexDataManagerHelpFunctions.h",
"mexInitFunctions.cpp",
"mexInitFunctions.h",
]
P3["files"] = [
"astra_mex_data3d_c.cpp",
"mexHelpFunctions.cpp",
"mexHelpFunctions.h",
"mexCopyDataHelpFunctions.cpp",
"mexCopyDataHelpFunctions.h",
"mexDataManagerHelpFunctions.cpp",
"mexDataManagerHelpFunctions.h",
"mexInitFunctions.cpp",
"mexInitFunctions.h",
]
P4["files"] = [
"astra_mex_matrix_c.cpp",
"mexHelpFunctions.cpp",
"mexHelpFunctions.h",
"mexInitFunctions.cpp",
"mexInitFunctions.h",
]
P5["files"] = [
"astra_mex_projector_c.cpp",
"mexHelpFunctions.cpp",
"mexHelpFunctions.h",
"mexInitFunctions.cpp",
"mexInitFunctions.h",
]
P6["files"] = [
"astra_mex_projector3d_c.cpp",
"mexHelpFunctions.cpp",
"mexHelpFunctions.h",
"mexInitFunctions.cpp",
"mexInitFunctions.h",
]
P7["files"] = [
"astra_mex_log_c.cpp",
"mexHelpFunctions.cpp",
"mexHelpFunctions.h",
"mexInitFunctions.cpp",
"mexInitFunctions.h",
]
P8["files"] = [
"astra_mex_direct_c.cpp",
"mexHelpFunctions.cpp",
"mexHelpFunctions.h",
"mexCopyDataHelpFunctions.cpp",
"mexCopyDataHelpFunctions.h",
"mexDataManagerHelpFunctions.cpp",
"mexDataManagerHelpFunctions.h",
"mexInitFunctions.cpp",
"mexInitFunctions.h",
]




P_astra["filter_names"] = [
"Algorithms",
"Data Structures",
"Projectors",
"CUDA",
"Global &amp; Other",
"Geometries",
"Algorithms\\headers",
"Algorithms\\source",
"Data Structures\\headers",
"Data Structures\\source",
"Global &amp; Other\\headers",
"Global &amp; Other\\source",
"Geometries\\headers",
"Geometries\\source",
"Projectors\\headers",
"Projectors\\inline",
"Projectors\\source",
"CUDA\\astra headers",
"CUDA\\astra source",
"CUDA\\cuda headers",
"CUDA\\cuda source",
]
P_astra["filters"] = {}
P_astra["filters"]["Algorithms"] = [ "262b0d17-774a-4cb1-b51a-b358d2d02791" ]
P_astra["filters"]["Data Structures"] = [ "76d6d672-670b-4454-b3ab-10dc8f9b8710" ]
P_astra["filters"]["Projectors"] = [ "77a581a9-60da-4265-97c0-80cdf97408c0" ]
P_astra["filters"]["CUDA"] = [ "c1af0e56-5fcc-4e75-b5db-88eeb4148185" ]
P_astra["filters"]["Global &amp; Other"] = [ "72fbe846-10ef-4c52-88df-13bd66c4cbfc" ]
P_astra["filters"]["Geometries"] = [ "7ef37c12-c98c-4dd6-938d-12f49279eae0" ]
P_astra["filters"]["CUDA\\cuda source"] = [
"04a878ed-77b4-4525-9bc2-38ccd65282c5",
"cuda\\2d\\algo.cu",
"cuda\\2d\\arith.cu",
"cuda\\2d\\astra.cu",
"cuda\\2d\\cgls.cu",
"cuda\\2d\\darthelper.cu",
"cuda\\2d\\em.cu",
"cuda\\2d\\fan_bp.cu",
"cuda\\2d\\fan_fp.cu",
"cuda\\2d\\fbp.cu",
"cuda\\2d\\fft.cu",
"cuda\\2d\\par_bp.cu",
"cuda\\2d\\par_fp.cu",
"cuda\\2d\\sart.cu",
"cuda\\2d\\sirt.cu",
"cuda\\2d\\util.cu",
"cuda\\3d\\algo3d.cu",
"cuda\\3d\\arith3d.cu",
"cuda\\3d\\astra3d.cu",
"cuda\\3d\\cgls3d.cu",
"cuda\\3d\\cone_bp.cu",
"cuda\\3d\\cone_fp.cu",
"cuda\\3d\\darthelper3d.cu",
"cuda\\3d\\fdk.cu",
"cuda\\3d\\mem3d.cu",
"cuda\\3d\\par3d_bp.cu",
"cuda\\3d\\par3d_fp.cu",
"cuda\\3d\\sirt3d.cu",
"cuda\\3d\\util3d.cu",
]
P_astra["filters"]["Algorithms\\source"] = [
"9df653ab-26c3-4bec-92a2-3dda22fda761",
"src\\Algorithm.cpp",
"src\\ArtAlgorithm.cpp",
"src\\BackProjectionAlgorithm.cpp",
"src\\CglsAlgorithm.cpp",
"src\\FilteredBackProjectionAlgorithm.cpp",
"src\\ForwardProjectionAlgorithm.cpp",
"src\\PluginAlgorithmFactory.cpp",
"src\\ReconstructionAlgorithm2D.cpp",
"src\\ReconstructionAlgorithm3D.cpp",
"src\\SartAlgorithm.cpp",
"src\\SirtAlgorithm.cpp",
]
P_astra["filters"]["Data Structures\\source"] = [
"95346487-8185-487b-a794-3e7fb5fcbd4c",
"src\\Data3D.cpp",
"src\\Float32Data.cpp",
"src\\Float32Data2D.cpp",
"src\\Float32ProjectionData2D.cpp",
"src\\Float32VolumeData2D.cpp",
"src\\SheppLogan.cpp",
"src\\SparseMatrix.cpp",
]
P_astra["filters"]["Global &amp; Other\\source"] = [
"1546cb47-7e5b-42c2-b695-ef172024c14b",
"src\\AstraObjectFactory.cpp",
"src\\AstraObjectManager.cpp",
"src\\CompositeGeometryManager.cpp",
"src\\Config.cpp",
"src\\Features.cpp",
"src\\Filters.cpp",
"src\\Fourier.cpp",
"src\\Globals.cpp",
"src\\Logging.cpp",
"src\\PlatformDepSystemCode.cpp",
"src\\Utilities.cpp",
"src\\XMLConfig.cpp",
"src\\XMLDocument.cpp",
"src\\XMLNode.cpp",
]
P_astra["filters"]["Geometries\\source"] = [
"dc27bff7-4256-4311-a131-47612a44af20",
"src\\ConeProjectionGeometry3D.cpp",
"src\\ConeVecProjectionGeometry3D.cpp",
"src\\FanFlatProjectionGeometry2D.cpp",
"src\\FanFlatVecProjectionGeometry2D.cpp",
"src\\GeometryUtil2D.cpp",
"src\\GeometryUtil3D.cpp",
"src\\ParallelProjectionGeometry2D.cpp",
"src\\ParallelProjectionGeometry3D.cpp",
"src\\ParallelVecProjectionGeometry2D.cpp",
"src\\ParallelVecProjectionGeometry3D.cpp",
"src\\ProjectionGeometry2D.cpp",
"src\\ProjectionGeometry2DFactory.cpp",
"src\\ProjectionGeometry3D.cpp",
"src\\ProjectionGeometry3DFactory.cpp",
"src\\SparseMatrixProjectionGeometry2D.cpp",
"src\\VolumeGeometry2D.cpp",
"src\\VolumeGeometry3D.cpp",
]
P_astra["filters"]["Projectors\\source"] = [
"2d60e3c8-7874-4cee-b139-991ac15e811d",
"src\\DataProjector.cpp",
"src\\DataProjectorPolicies.cpp",
"src\\FanFlatBeamLineKernelProjector2D.cpp",
"src\\FanFlatBeamStripKernelProjector2D.cpp",
"src\\ParallelBeamBlobKernelProjector2D.cpp",
"src\\ParallelBeamDistanceDrivenProjector2D.cpp",
"src\\ParallelBeamLinearKernelProjector2D.cpp",
"src\\ParallelBeamLineKernelProjector2D.cpp",
"src\\ParallelBeamStripKernelProjector2D.cpp",
"src\\Projector2D.cpp",
"src\\Projector3D.cpp",
"src\\SparseMatrixProjector2D.cpp",
]
P_astra["filters"]["CUDA\\astra source"] = [
"bbef012e-598a-456f-90d8-416bdcb4221c",
"src\\CudaBackProjectionAlgorithm.cpp",
"src\\CudaBackProjectionAlgorithm3D.cpp",
"src\\CudaCglsAlgorithm.cpp",
"src\\CudaCglsAlgorithm3D.cpp",
"src\\CudaDartMaskAlgorithm.cpp",
"src\\CudaDartMaskAlgorithm3D.cpp",
"src\\CudaDartSmoothingAlgorithm.cpp",
"src\\CudaDartSmoothingAlgorithm3D.cpp",
"src\\CudaDataOperationAlgorithm.cpp",
"src\\CudaEMAlgorithm.cpp",
"src\\CudaFDKAlgorithm3D.cpp",
"src\\CudaFilteredBackProjectionAlgorithm.cpp",
"src\\CudaForwardProjectionAlgorithm.cpp",
"src\\CudaForwardProjectionAlgorithm3D.cpp",
"src\\CudaProjector2D.cpp",
"src\\CudaProjector3D.cpp",
"src\\CudaReconstructionAlgorithm2D.cpp",
"src\\CudaRoiSelectAlgorithm.cpp",
"src\\CudaSartAlgorithm.cpp",
"src\\CudaSirtAlgorithm.cpp",
"src\\CudaSirtAlgorithm3D.cpp",
]
P_astra["filters"]["CUDA\\cuda headers"] = [
"4e17872e-db7d-41bc-9760-fad1c253b583",
"include\\astra\\cuda\\2d\\algo.h",
"include\\astra\\cuda\\2d\\arith.h",
"include\\astra\\cuda\\2d\\astra.h",
"include\\astra\\cuda\\2d\\cgls.h",
"include\\astra\\cuda\\2d\\darthelper.h",
"include\\astra\\cuda\\2d\\dims.h",
"include\\astra\\cuda\\2d\\em.h",
"include\\astra\\cuda\\2d\\fan_bp.h",
"include\\astra\\cuda\\2d\\fan_fp.h",
"include\\astra\\cuda\\2d\\fbp.h",
"include\\astra\\cuda\\2d\\fft.h",
"include\\astra\\cuda\\2d\\par_bp.h",
"include\\astra\\cuda\\2d\\par_fp.h",
"include\\astra\\cuda\\2d\\sart.h",
"include\\astra\\cuda\\2d\\sirt.h",
"include\\astra\\cuda\\2d\\util.h",
"include\\astra\\cuda\\3d\\algo3d.h",
"include\\astra\\cuda\\3d\\arith3d.h",
"include\\astra\\cuda\\3d\\astra3d.h",
"include\\astra\\cuda\\3d\\cgls3d.h",
"include\\astra\\cuda\\3d\\cone_bp.h",
"include\\astra\\cuda\\3d\\cone_fp.h",
"include\\astra\\cuda\\3d\\darthelper3d.h",
"include\\astra\\cuda\\3d\\dims3d.h",
"include\\astra\\cuda\\3d\\fdk.h",
"include\\astra\\cuda\\3d\\mem3d.h",
"include\\astra\\cuda\\3d\\par3d_bp.h",
"include\\astra\\cuda\\3d\\par3d_fp.h",
"include\\astra\\cuda\\3d\\sirt3d.h",
"include\\astra\\cuda\\3d\\util3d.h",
]
P_astra["filters"]["Algorithms\\headers"] = [
"a76ffd6d-3895-4365-b27e-fc9a72f2ed75",
"include\\astra\\Algorithm.h",
"include\\astra\\AlgorithmTypelist.h",
"include\\astra\\ArtAlgorithm.h",
"include\\astra\\BackProjectionAlgorithm.h",
"include\\astra\\CglsAlgorithm.h",
"include\\astra\\CudaBackProjectionAlgorithm.h",
"include\\astra\\CudaBackProjectionAlgorithm3D.h",
"include\\astra\\FilteredBackProjectionAlgorithm.h",
"include\\astra\\ForwardProjectionAlgorithm.h",
"include\\astra\\PluginAlgorithmFactory.h",
"include\\astra\\ReconstructionAlgorithm2D.h",
"include\\astra\\ReconstructionAlgorithm3D.h",
"include\\astra\\SartAlgorithm.h",
"include\\astra\\SirtAlgorithm.h",
]
P_astra["filters"]["Data Structures\\headers"] = [
"444c44b0-6454-483a-be26-7cb9c8ab0b98",
"include\\astra\\Data3D.h",
"include\\astra\\Float32Data.h",
"include\\astra\\Float32Data2D.h",
"include\\astra\\Float32ProjectionData2D.h",
"include\\astra\\Float32VolumeData2D.h",
"include\\astra\\SheppLogan.h",
"include\\astra\\SparseMatrix.h",
]
P_astra["filters"]["Global &amp; Other\\headers"] = [
"1c52efc8-a77e-4c72-b9be-f6429a87e6d7",
"include\\astra\\AstraObjectFactory.h",
"include\\astra\\AstraObjectManager.h",
"include\\astra\\clog.h",
"include\\astra\\CompositeGeometryManager.h",
"include\\astra\\Config.h",
"include\\astra\\Features.h",
"include\\astra\\Filters.h",
"include\\astra\\Fourier.h",
"include\\astra\\Globals.h",
"include\\astra\\Logging.h",
"include\\astra\\PlatformDepSystemCode.h",
"include\\astra\\Singleton.h",
"include\\astra\\TypeList.h",
"include\\astra\\Utilities.h",
"include\\astra\\Vector3D.h",
"include\\astra\\XMLConfig.h",
"include\\astra\\XMLDocument.h",
"include\\astra\\XMLNode.h",
]
P_astra["filters"]["Geometries\\headers"] = [
"eddb31ba-0db7-4ab1-a490-36623aaf8901",
"include\\astra\\ConeProjectionGeometry3D.h",
"include\\astra\\ConeVecProjectionGeometry3D.h",
"include\\astra\\FanFlatProjectionGeometry2D.h",
"include\\astra\\FanFlatVecProjectionGeometry2D.h",
"include\\astra\\GeometryUtil2D.h",
"include\\astra\\GeometryUtil3D.h",
"include\\astra\\ParallelProjectionGeometry2D.h",
"include\\astra\\ParallelProjectionGeometry3D.h",
"include\\astra\\ParallelVecProjectionGeometry2D.h",
"include\\astra\\ParallelVecProjectionGeometry3D.h",
"include\\astra\\ProjectionGeometry2D.h",
"include\\astra\\ProjectionGeometry2DFactory.h",
"include\\astra\\ProjectionGeometry3D.h",
"include\\astra\\ProjectionGeometry3DFactory.h",
"include\\astra\\SparseMatrixProjectionGeometry2D.h",
"include\\astra\\VolumeGeometry2D.h",
"include\\astra\\VolumeGeometry3D.h",
]
P_astra["filters"]["Projectors\\headers"] = [
"91ae2cfd-6b45-46eb-ad99-2f16e5ce4b1e",
"include\\astra\\DataProjector.h",
"include\\astra\\DataProjectorPolicies.h",
"include\\astra\\FanFlatBeamLineKernelProjector2D.h",
"include\\astra\\FanFlatBeamStripKernelProjector2D.h",
"include\\astra\\ParallelBeamBlobKernelProjector2D.h",
"include\\astra\\ParallelBeamDistanceDrivenProjector2D.h",
"include\\astra\\ParallelBeamLinearKernelProjector2D.h",
"include\\astra\\ParallelBeamLineKernelProjector2D.h",
"include\\astra\\ParallelBeamStripKernelProjector2D.h",
"include\\astra\\Projector2D.h",
"include\\astra\\Projector3D.h",
"include\\astra\\ProjectorTypelist.h",
"include\\astra\\SparseMatrixProjector2D.h",
]
P_astra["filters"]["CUDA\\astra headers"] = [
"bd4e1f94-2f56-4db6-b946-20c29d65a351",
"include\\astra\\CudaCglsAlgorithm.h",
"include\\astra\\CudaCglsAlgorithm3D.h",
"include\\astra\\CudaDartMaskAlgorithm.h",
"include\\astra\\CudaDartMaskAlgorithm3D.h",
"include\\astra\\CudaDartSmoothingAlgorithm.h",
"include\\astra\\CudaDartSmoothingAlgorithm3D.h",
"include\\astra\\CudaDataOperationAlgorithm.h",
"include\\astra\\CudaEMAlgorithm.h",
"include\\astra\\CudaFDKAlgorithm3D.h",
"include\\astra\\CudaFilteredBackProjectionAlgorithm.h",
"include\\astra\\CudaForwardProjectionAlgorithm.h",
"include\\astra\\CudaForwardProjectionAlgorithm3D.h",
"include\\astra\\CudaProjector2D.h",
"include\\astra\\CudaProjector3D.h",
"include\\astra\\CudaReconstructionAlgorithm2D.h",
"include\\astra\\CudaRoiSelectAlgorithm.h",
"include\\astra\\CudaSartAlgorithm.h",
"include\\astra\\CudaSirtAlgorithm.h",
"include\\astra\\CudaSirtAlgorithm3D.h",
]
P_astra["filters"]["Projectors\\inline"] = [
"0daffd63-ba49-4a5f-8d7a-5322e0e74f22",
"include\\astra\\DataProjectorPolicies.inl",
"include\\astra\\FanFlatBeamLineKernelProjector2D.inl",
"include\\astra\\FanFlatBeamStripKernelProjector2D.inl",
"include\\astra\\ParallelBeamBlobKernelProjector2D.inl",
"include\\astra\\ParallelBeamDistanceDrivenProjector2D.inl",
"include\\astra\\ParallelBeamLinearKernelProjector2D.inl",
"include\\astra\\ParallelBeamLineKernelProjector2D.inl",
"include\\astra\\ParallelBeamStripKernelProjector2D.inl",
"include\\astra\\SparseMatrixProjector2D.inl",
]

P_astra["files"] = []
for f in P_astra["filters"]:
  P_astra["files"].extend(P_astra["filters"][f][1:])
P_astra["files"].sort()

projects = [ P_astra, F_astra_mex, P0, P1, P2, P3, P4, P5, P6, P7, P8 ]

bom = codecs.BOM_UTF8.decode("utf-8")

class Configuration:
  def __init__(self, debug, cuda):
    self.debug = debug
    self.cuda = cuda
  def type(self):
    if self.debug:
      return "Debug"
    else:
      return "Release"
  def config(self):
    n = self.type()
    if self.cuda:
      n += "_CUDA"
    return n
  def platform(self):
    return "x64"
  def name(self):
    n = self.config()
    n += "|"
    n += self.platform()
    return n
  def target(self):
    n = "Astra"
    if self.cuda:
      n += "Cuda"
    n += "64"
    if self.debug:
      n += "D"
    return n
      


configs = [ Configuration(a,b) for a in [ True, False ] for b in [ True, False ] ]

def write_sln():
  main_project = P_astra
  F = open("astra_vc14.sln", "w", encoding="utf-8")
  print(bom, file=F)
  print("Microsoft Visual Studio Solution File, Format Version 12.00", file=F)
  print("# Visual Studio 14", file=F)
  print("VisualStudioVersion = 14.0.25420.1", file=F)
  print("MinimumVisualStudioVersion = 10.0.40219.1", file=F)
  for p in projects:
    s = '''Project("{%s}") = "%s", "projects\\%s", "{%s}"''' % (p["type"], p["name"], p["file14"], p["uuid14"])
    print(s, file=F)
    if "mex" in p["name"]:
      print("\tProjectSection(ProjectDependencies) = postProject", file=F)
      print("\t\t{%s} = {%s}" % (main_project["uuid14"], main_project["uuid14"]), file=F)
      print("\tEndProjectSection", file=F)
    print("EndProject", file=F)
  print("Global", file=F)
  print("\tGlobalSection(SolutionConfigurationPlatforms) = preSolution", file=F)
  for c in configs:
    print("\t\t" + c.name() + " = " + c.name(), file=F)
  print("\tEndGlobalSection", file=F)
  print("\tGlobalSection(ProjectConfigurationPlatforms) = postSolution", file=F)
  for p in projects:
    if "entries" in p:
      continue
    for c in configs:
      print("\t\t{" + p["uuid14"] + "}." + c.name() + ".ActiveCfg = " + c.name(), file=F)
      print("\t\t{" + p["uuid14"] + "}." + c.name() + ".Build.0 = " + c.name(), file=F)
  print("\tEndGlobalSection", file=F)
  print("\tGlobalSection(SolutionProperties) = preSolution", file=F)
  print("\t\tHideSolutionNode = FALSE", file=F)
  print("\tEndGlobalSection", file=F)
  print("\tGlobalSection(NestedProjects) = preSolution", file=F)
  for p in projects:
    if "entries" not in p:
      continue
    for e in p["entries"]:
      print("\t\t{" + e["uuid14"] + "} = {" + p["uuid14"] + "}", file=F)
  print("\tEndGlobalSection", file=F)
  print("EndGlobal", file=F)
  F.close()

def write_project14_start(P, F):
  print(bom + '<?xml version="1.0" encoding="utf-8"?>', file=F)
  print('<Project DefaultTargets="Build" ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">', file=F)
  print('  <ItemGroup Label="ProjectConfigurations">', file=F)
  for c in configs:
    print('    <ProjectConfiguration Include="' + c.name() + '">', file=F)
    print('      <Configuration>' + c.config() + '</Configuration>', file=F)
    print('      <Platform>' + c.platform() + '</Platform>', file=F)
    print('    </ProjectConfiguration>', file=F)
  print('  </ItemGroup>', file=F)
  print('  <PropertyGroup Label="Globals">', file=F)
  if 'mex' in P["name"]:
    print('    <ProjectName>' + P["name"] + '</ProjectName>', file=F)
  print('    <ProjectGuid>{' + P["uuid14"] + '}</ProjectGuid>', file=F)
  if 'mex' in P["name"]:
    print('    <RootNamespace>astraMatlab</RootNamespace>', file=F)
  else:
    print('    <RootNamespace>' + P["name"] + '</RootNamespace>', file=F)
  print('    <WindowsTargetPlatformVersion>10.0.22621.0</WindowsTargetPlatformVersion>', file=F)
  print('  </PropertyGroup>', file=F)
  print('  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />', file=F)
  for c in configs:
    print('''  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='%s'" Label="Configuration">''' % (c.name(), ), file=F)
    print('    <ConfigurationType>DynamicLibrary</ConfigurationType>', file=F)
    if 'mex' not in P["name"]:
      if c.debug:
        print('    <UseDebugLibraries>true</UseDebugLibraries>', file=F)
      else:
        print('    <UseDebugLibraries>false</UseDebugLibraries>', file=F)
    print('    <PlatformToolset>v141</PlatformToolset>', file=F)
    if 'mex' not in P["name"]:
      if not c.debug:
        print('    <WholeProgramOptimization>true</WholeProgramOptimization>', file=F)
      print('    <CharacterSet>MultiByte</CharacterSet>', file=F)
    print('  </PropertyGroup>', file=F)
  print('  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />', file=F)
  print('  <ImportGroup Label="ExtensionSettings">', file=F)
  if "mex" not in P["name"]:
    print(f'    <Import Project="$(CUDA_PATH_V{CUDA_MAJOR}_{CUDA_MINOR})\\extras\\visual_studio_integration\\MSBuildExtensions\\CUDA {CUDA_MAJOR}.{CUDA_MINOR}.props" />', file=F)
  print('  </ImportGroup>', file=F)
  for c in configs:
    print('''  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='%s'">''' % (c.name(), ), file=F)
    print('''    <Import Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />''', file=F)
    print('''  </ImportGroup>''', file=F)
  print('  <PropertyGroup Label="UserMacros" />', file=F)

def write_project14_end(P, F):
  relpath = '..\\..\\..\\'
  if 'mex' in P["name"]:
      relpath += 'matlab\\mex\\'
  l = [ f for f in P["files"] if len(f) > 4 and f[-4:] == ".cpp" ]
  if l:
    print('  <ItemGroup>', file=F)
    for f in l:
      if ("cuda" in f) or ("Cuda" in f):
        print('    <ClCompile Include="' + relpath + f + '">', file=F)
        for c in configs:
          if not c.cuda:
            print('''      <ExcludedFromBuild Condition="'$(Configuration)|$(Platform)'=='%s'">true</ExcludedFromBuild>''' % (c.name(), ), file=F)
        print('    </ClCompile>', file=F)
      else:
        print('    <ClCompile Include="' + relpath + f + '" />', file=F)
    print('  </ItemGroup>', file=F)
  l = [ f for f in P["files"] if len(f) > 2 and f[-2:] == ".h" ]
  if l:
    print('  <ItemGroup>', file=F)
    for f in l:
      print('    <ClInclude Include="' + relpath + f + '" />', file=F)
    print('  </ItemGroup>', file=F)
  l = [ f for f in P["files"] if len(f) > 3 and f[-3:] == ".cu" ]
  if l:
    print('  <ItemGroup>', file=F)
    for f in l:
      print('    <CudaCompile Include="' + relpath + f + '">', file=F)
      for c in configs:
        if not c.cuda:
          print('''      <ExcludedFromBuild Condition="'$(Configuration)|$(Platform)'=='%s'">true</ExcludedFromBuild>''' % (c.name(), ), file=F)
      print('    </CudaCompile>', file=F)
    print('  </ItemGroup>', file=F)
  l = [ f for f in P["files"] if len(f) > 4 and f[-4:] == ".inl" ]
  if l:
    print('  <ItemGroup>', file=F)
    for f in l:
      print('    <None Include="' + relpath + f + '" />', file=F)
    print('  </ItemGroup>', file=F)
  print('  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />', file=F)
  print('  <ImportGroup Label="ExtensionTargets">', file=F)
  if "mex" not in P["name"]:
    print(f'    <Import Project="$(CUDA_PATH_V{CUDA_MAJOR}_{CUDA_MINOR})\\extras\\visual_studio_integration\\MSBuildExtensions\\CUDA {CUDA_MAJOR}.{CUDA_MINOR}.targets" />', file=F)
  print('  </ImportGroup>', file=F)
  print('</Project>', end="", file=F)


def write_main_project14():
  P = P_astra;
  F = open(os.path.join("projects", P["file14"]), "w", encoding="utf-8")
  write_project14_start(P, F)
  for c in configs:
    print('''  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='%s'">''' % (c.name(), ), file=F)
    print('    <OutDir>..\\bin\\$(Platform)\\$(Configuration)\\</OutDir>', file=F)
    print('    <IntDir>$(OutDir)obj\\</IntDir>', file=F)
    print('    <CudaIntDir>$(OutDir)obj\\</CudaIntDir>', file=F)
    print('    <CudaIntDirFullPath>$(SolutionDir)bin\\$(Platform)\\$(Configuration)\\obj\\</CudaIntDirFullPath>', file=F)
    print('    <TargetExt>.dll</TargetExt>', file=F)
    print('    <TargetName>' + c.target() + '</TargetName>', file=F)
    print('    <GenerateManifest>true</GenerateManifest>', file=F)
    print('  </PropertyGroup>', file=F)
  for c in configs:
    print('''  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='%s'">''' % (c.name(), ), file=F)
    print('    <ClCompile>', file=F)
    if c.debug:
      print('      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>', file=F)
    else:
      print('      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>', file=F)
    print('      <WarningLevel>Level3</WarningLevel>', file=F)
    print('      <AdditionalIncludeDirectories>..\\..\\..\\lib\\include;..\\..\\..\\include\\;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>', file=F)
    print('      <OpenMPSupport>true</OpenMPSupport>', file=F)
    if c.debug:
      print('      <Optimization>Disabled</Optimization>', file=F)
    else:
      print('      <Optimization>MaxSpeed</Optimization>', file=F)
      print('      <FunctionLevelLinking>true</FunctionLevelLinking>', file=F)
      print('      <IntrinsicFunctions>true</IntrinsicFunctions>', file=F)
      print('      <InlineFunctionExpansion>AnySuitable</InlineFunctionExpansion>', file=F)
      print('      <FavorSizeOrSpeed>Speed</FavorSizeOrSpeed>', file=F)
    d='      <PreprocessorDefinitions>'
    if c.cuda:
      d+="ASTRA_CUDA;"
    d+="__SSE2__;"
    d+="DLL_EXPORTS;_CRT_SECURE_NO_WARNINGS;"
    d+='%(PreprocessorDefinitions)</PreprocessorDefinitions>'
    print(d, file=F)
    print('      <MultiProcessorCompilation>true</MultiProcessorCompilation>', file=F)
    print('      <SDLCheck>true</SDLCheck>', file=F)
    print('      <LanguageStandard>stdcpp17</LanguageStandard>', file=F)
    print('    </ClCompile>', file=F)
    print('    <Link>', file=F)
    print('      <GenerateDebugInformation>true</GenerateDebugInformation>', file=F)
    if not c.debug:
      print('      <EnableCOMDATFolding>true</EnableCOMDATFolding>', file=F)
      print('      <OptimizeReferences>true</OptimizeReferences>', file=F)
    print('      <OutputFile>..\\bin\\' + c.platform() + '\\' + c.config() + '\\' + c.target() + '.dll</OutputFile>', file=F)
    if c.cuda:
      print('      <AdditionalDependencies>cudart.lib;cufft.lib;%(AdditionalDependencies)</AdditionalDependencies>', file=F)
    l = '      <AdditionalLibraryDirectories>';
    l += '%(AdditionalLibraryDirectories)'
    if c.cuda:
      l += ';$(CudaToolkitLibDir)'
    l += '</AdditionalLibraryDirectories>'
    print(l, file=F)
    print('    </Link>', file=F)
    if c.cuda:
      print('    <CudaCompile>', file=F)
      print('      <TargetMachinePlatform>64</TargetMachinePlatform>', file=F)
      print('      <GenerateLineInfo>true</GenerateLineInfo>', file=F)
      print(f'      <CodeGeneration>{CUDA_CC[(CUDA_MAJOR,CUDA_MINOR)]}</CodeGeneration>', file=F)
      print('      <AdditionalOptions>-std=c++17</AdditionalOptions>', file=F)
      print('      <AdditionalCompilerOptions>/std:c++17</AdditionalCompilerOptions>', file=F)
      print('    </CudaCompile>', file=F)
    print('  </ItemDefinitionGroup>', file=F)
  write_project14_end(P, F)
  F.close()

def write_mex_project14(P):
  F = open(os.path.join("projects", P["name"] + "_vc14.vcxproj"), "w", encoding="utf-8")
  write_project14_start(P, F)
  print('  <PropertyGroup>', file=F)
  print('    <_ProjectFileVersion>11.0.60610.1</_ProjectFileVersion>', file=F)
  print('  </PropertyGroup>', file=F)
  for c in configs:
    print('''  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='%s'">''' % (c.name(), ), file=F)
    print('    <OutDir>..\\bin\\$(Platform)\\$(Configuration)\\</OutDir>', file=F)
    print('    <IntDir>$(OutDir)obj\\$(ProjectName)\\</IntDir>', file=F)
    print('    <TargetName>$(ProjectName)_c</TargetName>', file=F)
    print('    <TargetExt>.mexw64</TargetExt>', file=F)
    print('  </PropertyGroup>', file=F)
  for c in configs:
    print('''  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='%s'">''' % (c.name(), ), file=F)
    print('    <ClCompile>', file=F)
    if c.debug:
      print('      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>', file=F)
    else:
      print('      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>', file=F)
#    print('      <WarningLevel>Level3</WarningLevel>', file=F)
    #print('      <AdditionalIncludeDirectories>$(MATLAB_ROOT)\\extern\\include\\;..\\..\\lib\\include;..\\..\\include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>', file=F)
    # FIXME: This CUDA_PATH shouldn't be necessary
    print('      <AdditionalIncludeDirectories>$(MATLAB_ROOT)\\extern\\include\\;$(CUDA_PATH)\\include;..\\..\\..\\lib\\include;..\\..\\..\\include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>', file=F)
    print('      <OpenMPSupport>true</OpenMPSupport>', file=F)
    if c.debug:
      print('      <Optimization>Disabled</Optimization>', file=F)
    else:
      print('      <Optimization>MaxSpeed</Optimization>', file=F)
#      print('      <FunctionLevelLinking>true</FunctionLevelLinking>', file=F)
#      print('      <IntrinsicFunctions>true</IntrinsicFunctions>', file=F)
#      print('      <InlineFunctionExpansion>AnySuitable</InlineFunctionExpansion>', file=F)
#      print('      <FavorSizeOrSpeed>Speed</FavorSizeOrSpeed>', file=F)
    d='      <PreprocessorDefinitions>'
    if c.cuda:
      d+="ASTRA_CUDA;"
    d+="__SSE2__;"
    d+="MATLAB_MEXCMD_RELEASE=700;"
#    d+="DLL_EXPORTS;_CRT_SECURE_NO_WARNINGS;"
    d+='%(PreprocessorDefinitions)</PreprocessorDefinitions>'
    print(d, file=F)
    print('      <LanguageStandard>stdcpp17</LanguageStandard>', file=F)
    print('      <MultiProcessorCompilation>true</MultiProcessorCompilation>', file=F)
#    print('      <SDLCheck>true</SDLCheck>', file=F)
#   if c.debug:
#   <DebugInformationFormat>EditAndContinue</DebugInformationFormat> ??
    print('    </ClCompile>', file=F)
    print('    <Link>', file=F)
#    if not c.debug:
#      print('      <EnableCOMDATFolding>true</EnableCOMDATFolding>', file=F)
#      print('      <OptimizeReferences>true</OptimizeReferences>', file=F)
    print('      <OutputFile>$(OutDir)$(ProjectName)_c.mexw64</OutputFile>', file=F)
    print('      <AdditionalDependencies>%s.lib;libmex.lib;libmx.lib;libut.lib;%%(AdditionalDependencies)</AdditionalDependencies>' % (c.target(), ), file=F)
    l = '      <AdditionalLibraryDirectories>';
    l += '..\\bin\\x64\\'
    l += c.config()
    l += ';$(MATLAB_ROOT)\\extern\\lib\\win64\\microsoft'
    l += ';%(AdditionalLibraryDirectories)'
    l += '</AdditionalLibraryDirectories>'
    print(l, file=F)
    print('      <ModuleDefinitionFile>mex.def</ModuleDefinitionFile>', file=F)
    print('      <GenerateDebugInformation>true</GenerateDebugInformation>', file=F)
    print('    </Link>', file=F)
    print('  </ItemDefinitionGroup>', file=F)
  write_project14_end(P, F)
  F.close()

def write_main_filters14():
  P = P_astra
  F = open(os.path.join("projects", P["name"] + ".vcxproj.filters"), "w", encoding="utf-8")
  print(bom + '<?xml version="1.0" encoding="utf-8"?>', file=F)
  print('<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">', file=F)
  print('  <ItemGroup>', file=F)
  for Filter in P_astra["filter_names"]:
    L = P_astra["filters"][Filter][1:]
    l = [ f for f in L if len(f) > 3 and f[-3:] == ".cu" ]
    for f in l:
      print('    <CudaCompile Include="..\\..\\..\\' + f + '">', file=F)
      print('      <Filter>' + Filter + '</Filter>', file=F)
      print('    </CudaCompile>', file=F)
  print('  </ItemGroup>', file=F)
  print('  <ItemGroup>', file=F)
  for Filter in P_astra["filter_names"]:
    L = P_astra["filters"][Filter][1:]
    l = [ f for f in L if len(f) > 4 and f[-4:] == ".cpp" ]
    for f in l:
      print('    <ClCompile Include="..\\..\\..\\' + f + '">', file=F)
      print('      <Filter>' + Filter + '</Filter>', file=F)
      print('    </ClCompile>', file=F)
  print('  </ItemGroup>', file=F)
  print('  <ItemGroup>', file=F)
  for Filter in P_astra["filter_names"]:
    L = P_astra["filters"][Filter][1:]
    l = [ f for f in L if len(f) > 2 and f[-2:] == ".h" ]
    for f in l:
      print('    <ClInclude Include="..\\..\\..\\' + f + '">', file=F)
      print('      <Filter>' + Filter + '</Filter>', file=F)
      print('    </ClInclude>', file=F)
  print('  </ItemGroup>', file=F)
  print('  <ItemGroup>', file=F)
  for Filter in P_astra["filter_names"]:
    L = P_astra["filters"][Filter][1:]
    l = [ f for f in L if len(f) > 4 and f[-4:] == ".inl" ]
    for f in l:
      print('    <None Include="..\\..\\..\\' + f + '">', file=F)
      print('      <Filter>' + Filter + '</Filter>', file=F)
      print('    </None>', file=F)
  print('  </ItemGroup>', file=F)
  print('  <ItemGroup>', file=F)
  for f in P["filter_names"]:
    print('    <Filter Include="' + f + '">', file=F)
    print('      <UniqueIdentifier>{' + P["filters"][f][0] + '}</UniqueIdentifier>', file=F)
    print('    </Filter>', file=F)
  print('  </ItemGroup>', file=F)
  print('</Project>', end="", file=F)
  F.close()


def parse_cuda_version(ver):
  return [ int(x) for x in ver.split('.') ]

def check_cuda_version(ver):
  try:
    major, minor = parse_cuda_version(ver)
    if major >= 9 and minor >= 0:
      return True
  except:
    pass
  return False

if (len(sys.argv) != 2) or not check_cuda_version(sys.argv[1]):
  print("Usage: python gen.py [10.2|11.0|...]", file=sys.stderr)
  sys.exit(1)

CUDA_MAJOR, CUDA_MINOR = parse_cuda_version(sys.argv[1])

try:
  open("../../src/AstraObjectManager.cpp", "r")
except IOError:
  print("Run gen.py from the build/msvc directory", file=sys.stderr)
  sys.exit(1)

# Change directory to main dir
os.makedirs("projects", exist_ok=True)

write_sln()
write_main_project14()
write_main_filters14()
write_mex_project14(P0)
write_mex_project14(P1)
write_mex_project14(P2)
write_mex_project14(P3)
write_mex_project14(P4)
write_mex_project14(P5)
write_mex_project14(P6)
write_mex_project14(P7)
write_mex_project14(P8)
