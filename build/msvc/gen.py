from __future__ import print_function
import sys
import os
import codecs
import six

vcppguid = "8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942" # C++ project
siguid = "2150E333-8FDC-42A3-9474-1A3956D46DE8" # project group 

# to generate a new uuid:
#
# import uuid
# uuid.uuid4().__str__().upper()

def create_mex_project(name, uuid11, uuid09):
  return { "type": vcppguid, "name": name, "file11": "matlab\\mex\\" + name + "_vc11.vcxproj", "file09": "matlab\\mex\\" + name + "_vc09.vcproj", "uuid11": uuid11, "uuid09": uuid09, "files": [] }

P_astra = { "type": vcppguid, "name": "astra_vc11", "file11": "astra_vc11.vcxproj", "file09": "astra_vc09.vcproj", "uuid11": "BE9F1326-527C-4284-AE2C-D1E25D539CEA", "uuid09": "12926444-6723-46A8-B388-12E65E0577FA" }

P0 = create_mex_project("astra_mex", "3FDA35E0-0D54-4663-A3E6-5ABA96F32221", "3FDA35E0-0D54-4663-A3E6-5ABA96F32221") 

P1 = create_mex_project("astra_mex_algorithm", "056BF7A9-294D-487C-8CC3-BE629077CA94", "056BF7A9-294D-487C-8CC3-BE629077CA94")
P2 = create_mex_project("astra_mex_data2d", "E4092269-B19C-46F7-A84E-4F146CC70E44", "E4092269-B19C-46F7-A84E-4F146CC70E44")
P3 = create_mex_project("astra_mex_data3d", "0BEC029B-0929-4BF9-BD8B-9C9806A52065", "0BEC029B-0929-4BF9-BD8B-9C9806A52065")
P4 = create_mex_project("astra_mex_matrix", "9D041710-2119-4230-BCF2-5FBE753FDE49", "9D041710-2119-4230-BCF2-5FBE753FDE49")
P5 = create_mex_project("astra_mex_projector", "4DD6056F-8EEE-4C9A-B2A9-923F01A32E97", "4DD6056F-8EEE-4C9A-B2A9-923F01A32E97")
P6 = create_mex_project("astra_mex_projector3d", "F94CCD79-AA11-42DF-AC8A-6C9D2238A883", "F94CCD79-AA11-42DF-AC8A-6C9D2238A883")
P7 = create_mex_project("astra_mex_log", "03B833F5-4FD6-4FBE-AAF4-E3305CD56D2E", "CA2840B3-DA68-41B5-AC57-F5DFD20ED8F8")
P8 = create_mex_project("astra_mex_direct", "0F68F4E2-BE1B-4A9A-B101-AECF4C069CC7", "85FE09A6-FA49-4314-A2B1-59D77C7442A8")

F_astra_mex = { "type": siguid,
                "name": "astra_mex",
                "file11": "astra_mex",
                "file09": "astra_mex",
                "uuid11": "5E99A109-374E-4102-BE9B-99BA1FA8AA30",
                "uuid09": "33EF0AC5-B475-40BF-BAE5-67075B204D10",
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
"src\\AsyncAlgorithm.cpp",
"src\\BackProjectionAlgorithm.cpp",
"src\\CglsAlgorithm.cpp",
"src\\FilteredBackProjectionAlgorithm.cpp",
"src\\ForwardProjectionAlgorithm.cpp",
"src\\PluginAlgorithm.cpp",
"src\\ReconstructionAlgorithm2D.cpp",
"src\\ReconstructionAlgorithm3D.cpp",
"src\\SartAlgorithm.cpp",
"src\\SirtAlgorithm.cpp",
]
P_astra["filters"]["Data Structures\\source"] = [
"95346487-8185-487b-a794-3e7fb5fcbd4c",
"src\\Float32Data.cpp",
"src\\Float32Data2D.cpp",
"src\\Float32Data3D.cpp",
"src\\Float32Data3DMemory.cpp",
"src\\Float32ProjectionData2D.cpp",
"src\\Float32ProjectionData3D.cpp",
"src\\Float32ProjectionData3DMemory.cpp",
"src\\Float32VolumeData2D.cpp",
"src\\Float32VolumeData3D.cpp",
"src\\Float32VolumeData3DMemory.cpp",
"src\\SparseMatrix.cpp",
]
P_astra["filters"]["Global &amp; Other\\source"] = [
"1546cb47-7e5b-42c2-b695-ef172024c14b",
"src\\AstraObjectFactory.cpp",
"src\\AstraObjectManager.cpp",
"src\\CompositeGeometryManager.cpp",
"src\\Config.cpp",
"src\\Fourier.cpp",
"src\\Globals.cpp",
"src\\Logging.cpp",
"src\\PlatformDepSystemCode.cpp",
"src\\Utilities.cpp",
"src\\XMLDocument.cpp",
"src\\XMLNode.cpp",
]
P_astra["filters"]["Geometries\\source"] = [
"dc27bff7-4256-4311-a131-47612a44af20",
"src\\ConeProjectionGeometry3D.cpp",
"src\\ConeVecProjectionGeometry3D.cpp",
"src\\FanFlatProjectionGeometry2D.cpp",
"src\\FanFlatVecProjectionGeometry2D.cpp",
"src\\GeometryUtil3D.cpp",
"src\\ParallelProjectionGeometry2D.cpp",
"src\\ParallelProjectionGeometry3D.cpp",
"src\\ParallelVecProjectionGeometry3D.cpp",
"src\\ProjectionGeometry2D.cpp",
"src\\ProjectionGeometry3D.cpp",
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
"cuda\\2d\\algo.h",
"cuda\\2d\\arith.h",
"cuda\\2d\\astra.h",
"cuda\\2d\\cgls.h",
"cuda\\2d\\darthelper.h",
"cuda\\2d\\dims.h",
"cuda\\2d\\em.h",
"cuda\\2d\\fan_bp.h",
"cuda\\2d\\fan_fp.h",
"cuda\\2d\\fbp_filters.h",
"cuda\\2d\\fft.h",
"cuda\\2d\\par_bp.h",
"cuda\\2d\\par_fp.h",
"cuda\\2d\\sart.h",
"cuda\\2d\\sirt.h",
"cuda\\2d\\util.h",
"cuda\\3d\\algo3d.h",
"cuda\\3d\\arith3d.h",
"cuda\\3d\\astra3d.h",
"cuda\\3d\\cgls3d.h",
"cuda\\3d\\cone_bp.h",
"cuda\\3d\\cone_fp.h",
"cuda\\3d\\darthelper3d.h",
"cuda\\3d\\dims3d.h",
"cuda\\3d\\fdk.h",
"cuda\\3d\\mem3d.h",
"cuda\\3d\\par3d_bp.h",
"cuda\\3d\\par3d_fp.h",
"cuda\\3d\\sirt3d.h",
"cuda\\3d\\util3d.h",
]
P_astra["filters"]["Algorithms\\headers"] = [
"a76ffd6d-3895-4365-b27e-fc9a72f2ed75",
"include\\astra\\Algorithm.h",
"include\\astra\\AlgorithmTypelist.h",
"include\\astra\\ArtAlgorithm.h",
"include\\astra\\AsyncAlgorithm.h",
"include\\astra\\BackProjectionAlgorithm.h",
"include\\astra\\CglsAlgorithm.h",
"include\\astra\\CudaBackProjectionAlgorithm.h",
"include\\astra\\CudaBackProjectionAlgorithm3D.h",
"include\\astra\\FilteredBackProjectionAlgorithm.h",
"include\\astra\\ForwardProjectionAlgorithm.h",
"include\\astra\\PluginAlgorithm.h",
"include\\astra\\ReconstructionAlgorithm2D.h",
"include\\astra\\ReconstructionAlgorithm3D.h",
"include\\astra\\SartAlgorithm.h",
"include\\astra\\SirtAlgorithm.h",
]
P_astra["filters"]["Data Structures\\headers"] = [
"444c44b0-6454-483a-be26-7cb9c8ab0b98",
"include\\astra\\Float32Data.h",
"include\\astra\\Float32Data2D.h",
"include\\astra\\Float32Data3D.h",
"include\\astra\\Float32Data3DMemory.h",
"include\\astra\\Float32ProjectionData2D.h",
"include\\astra\\Float32ProjectionData3D.h",
"include\\astra\\Float32ProjectionData3DMemory.h",
"include\\astra\\Float32VolumeData2D.h",
"include\\astra\\Float32VolumeData3D.h",
"include\\astra\\Float32VolumeData3DMemory.h",
"include\\astra\\SparseMatrix.h",
]
P_astra["filters"]["Global &amp; Other\\headers"] = [
"1c52efc8-a77e-4c72-b9be-f6429a87e6d7",
"include\\astra\\AstraObjectFactory.h",
"include\\astra\\AstraObjectManager.h",
"include\\astra\\clog.h",
"include\\astra\\CompositeGeometryManager.h",
"include\\astra\\Config.h",
"include\\astra\\Fourier.h",
"include\\astra\\Globals.h",
"include\\astra\\Logging.h",
"include\\astra\\PlatformDepSystemCode.h",
"include\\astra\\Singleton.h",
"include\\astra\\TypeList.h",
"include\\astra\\Utilities.h",
"include\\astra\\Vector3D.h",
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
"include\\astra\\ParallelVecProjectionGeometry3D.h",
"include\\astra\\ProjectionGeometry2D.h",
"include\\astra\\ProjectionGeometry3D.h",
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

if six.PY2:
  bom = "\xef\xbb\xbf"
else:
  bom = codecs.BOM_UTF8.decode("utf-8")

class Configuration:
  def __init__(self, debug, cuda, x64):
    self.debug = debug
    self.cuda = cuda
    self.x64 = x64
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
    if self.x64:
      n = "x64"
    else:
      n = "Win32"
    return n 
  def name(self):
    n = self.config()
    n += "|"
    n += self.platform()
    return n
  def target(self):
    n = "Astra"
    if self.cuda:
      n += "Cuda"
    if self.x64:
      n += "64"
    else:
      n += "32"
    if self.debug:
      n += "D"
    return n
      


configs = [ Configuration(a,b,c) for a in [ True, False ] for b in [ True, False ] for c in [ False, True ] ]

def write_sln(version):
  main_project = P_astra
  if version == 9:
    F = open("astra_vc09.sln", "w")
  elif version == 11:
    F = open("astra_vc11.sln", "w")
  else:
    assert(False)
  print(bom, file=F)
  if version == 9:
    print("Microsoft Visual Studio Solution File, Format Version 10.00", file=F)
    print("# Visual Studio 2008", file=F)
    uuid = "uuid09"
    file_ = "file09"
  elif version == 11:
    print("Microsoft Visual Studio Solution File, Format Version 12.00", file=F)
    print("# Visual Studio 2012", file=F)
    uuid = "uuid11"
    file_ = "file11"
  for p in projects:
    s = '''Project("{%s}") = "%s", "%s", "{%s}"''' % (p["type"], p["name"], p[file_], p[uuid])
    print(s, file=F)
    if "mex" in p["name"]:
      print("\tProjectSection(ProjectDependencies) = postProject", file=F)
      print("\t\t{%s} = {%s}" % (main_project[uuid], main_project[uuid]), file=F)
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
      print("\t\t{" + p[uuid] + "}." + c.name() + ".ActiveCfg = " + c.name(), file=F)
      print("\t\t{" + p[uuid] + "}." + c.name() + ".Build.0 = " + c.name(), file=F)
  print("\tEndGlobalSection", file=F)
  print("\tGlobalSection(SolutionProperties) = preSolution", file=F)
  print("\t\tHideSolutionNode = FALSE", file=F)
  print("\tEndGlobalSection", file=F)
  print("\tGlobalSection(NestedProjects) = preSolution", file=F)
  for p in projects:
    if "entries" not in p:
      continue
    for e in p["entries"]:
      print("\t\t{" + e[uuid] + "} = {" + p[uuid] + "}", file=F)
  print("\tEndGlobalSection", file=F)
  print("EndGlobal", file=F)
  F.close()

def write_project11_start(P, F):
  print(bom + '<?xml version="1.0" encoding="utf-8"?>', file=F)
  print('<Project DefaultTargets="Build" ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">', file=F)
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
  print('    <ProjectGuid>{' + P["uuid11"] + '}</ProjectGuid>', file=F)
  if 'mex' in P["name"]:
    print('    <RootNamespace>astraMatlab</RootNamespace>', file=F)
  else:
    print('    <RootNamespace>' + P["name"] + '</RootNamespace>', file=F)
  print('  </PropertyGroup>', file=F)
  print('  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />', file=F)
  for c in configs:
    print('''  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='%s'" Label="Configuration">''' % (c.name(), ), file=F)
    print('    <ConfigurationType>DynamicLibrary</ConfigurationType>', file=F)
    if 'mex' not in P["name"]:
      if c.debug:
        print('    <UseDebugLibraries>true</UseDebugLibraries>', file=F)
      else:
        print('    <UseDebugLibraries>false</UseDebugLibraries>', file=F)
    print('    <PlatformToolset>v110</PlatformToolset>', file=F)
    if 'mex' not in P["name"]:
      if not c.debug:
        print('    <WholeProgramOptimization>true</WholeProgramOptimization>', file=F)
      print('    <CharacterSet>MultiByte</CharacterSet>', file=F)
    print('  </PropertyGroup>', file=F)
  print('  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />', file=F)
  print('  <ImportGroup Label="ExtensionSettings">', file=F)
  if "mex" not in P["name"]:
    print('    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 5.5.props" />', file=F)
  print('  </ImportGroup>', file=F)
  for c in configs:
    print('''  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='%s'">''' % (c.name(), ), file=F)
    print('''    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />''', file=F)
    print('''  </ImportGroup>''', file=F)
  print('  <PropertyGroup Label="UserMacros" />', file=F)

def write_project11_end(P, F):
  l = [ f for f in P["files"] if len(f) > 4 and f[-4:] == ".cpp" ]
  if l:
    print('  <ItemGroup>', file=F)
    for f in l:
      if ("cuda" in f) or ("Cuda" in f):
        print('    <ClCompile Include="' + f + '">', file=F)
        for c in configs:
          if not c.cuda:
            print('''      <ExcludedFromBuild Condition="'$(Configuration)|$(Platform)'=='%s'">true</ExcludedFromBuild>''' % (c.name(), ), file=F)
        print('    </ClCompile>', file=F)
      else:
        print('    <ClCompile Include="' + f + '" />', file=F)
    print('  </ItemGroup>', file=F)
  l = [ f for f in P["files"] if len(f) > 2 and f[-2:] == ".h" ]
  if l:
    print('  <ItemGroup>', file=F)
    for f in l:
      print('    <ClInclude Include="' + f + '" />', file=F)
    print('  </ItemGroup>', file=F)
  l = [ f for f in P["files"] if len(f) > 3 and f[-3:] == ".cu" ]
  if l:
    print('  <ItemGroup>', file=F)
    for f in l:
      print('    <CudaCompile Include="' + f + '">', file=F)
      for c in configs:
        if not c.cuda:
          print('''      <ExcludedFromBuild Condition="'$(Configuration)|$(Platform)'=='%s'">true</ExcludedFromBuild>''' % (c.name(), ), file=F)
      print('    </CudaCompile>', file=F)
    print('  </ItemGroup>', file=F)
  l = [ f for f in P["files"] if len(f) > 4 and f[-4:] == ".inl" ]
  if l:
    print('  <ItemGroup>', file=F)
    for f in l:
      print('    <None Include="' + f + '" />', file=F)
    print('  </ItemGroup>', file=F)
  print('  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />', file=F)
  print('  <ImportGroup Label="ExtensionTargets">', file=F)
  if "mex" not in P["name"]:
    print('    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 5.5.targets" />', file=F)
  print('  </ImportGroup>', file=F)
  print('</Project>', end="", file=F)


def write_main_project11():
  P = P_astra;
  F = open(P["file11"], "w")
  write_project11_start(P, F)
  for c in configs:
    print('''  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='%s'">''' % (c.name(), ), file=F)
    if c.cuda:
      print('    <IncludePath>$(CUDA_INC_PATH);$(IncludePath)</IncludePath>', file=F)
      print('    <LibraryPath>$(CUDA_LIB_PATH);$(LibraryPath)</LibraryPath>', file=F)
    print('    <OutDir>$(SolutionDir)bin\\$(Platform)\\' + c.config() + '\\</OutDir>', file=F)
    print('    <IntDir>$(OutDir)obj\\</IntDir>', file=F)
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
    print('      <AdditionalIncludeDirectories>lib\include;include\;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>', file=F)
    print('      <OpenMPSupport>true</OpenMPSupport>', file=F)
    if not c.x64: # /arch:SSE2 is implicit on x64
      print('      <EnableEnhancedInstructionSet>StreamingSIMDExtensions2</EnableEnhancedInstructionSet>', file=F)
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
    print('    </ClCompile>', file=F)
    print('    <Link>', file=F)
    print('      <GenerateDebugInformation>true</GenerateDebugInformation>', file=F)
    if not c.debug:
      print('      <EnableCOMDATFolding>true</EnableCOMDATFolding>', file=F)
      print('      <OptimizeReferences>true</OptimizeReferences>', file=F)
    print('      <OutputFile>bin\\' + c.platform() + '\\' + c.config() + '\\' + c.target() + '.dll</OutputFile>', file=F)
    if c.cuda:
      print('      <AdditionalDependencies>cudart.lib;cufft.lib;%(AdditionalDependencies)</AdditionalDependencies>', file=F)
    l = '      <AdditionalLibraryDirectories>';
    if c.x64:
      l += 'lib\\x64'
    else:
      l += 'lib\\win32'
    l += ';%(AdditionalLibraryDirectories)'
    if c.cuda:
      l += ';$(CudaToolkitLibDir)'
    l += '</AdditionalLibraryDirectories>'
    print(l, file=F)
    print('    </Link>', file=F)
    if c.cuda:
      print('    <CudaCompile>', file=F)
      if c.x64:
        print('      <TargetMachinePlatform>64</TargetMachinePlatform>', file=F)
      else:
        print('      <TargetMachinePlatform>32</TargetMachinePlatform>', file=F)
      print('      <GenerateLineInfo>true</GenerateLineInfo>', file=F)
      print('      <CodeGeneration>compute_20,sm_20;compute_30,sm_30;compute_30,sm_35;compute_30,compute_30</CodeGeneration>', file=F)
      print('    </CudaCompile>', file=F)
    print('  </ItemDefinitionGroup>', file=F)
  write_project11_end(P, F)
  F.close()

def write_mex_project11(P):
  F = open("matlab/mex/" + P["name"] + "_vc11.vcxproj", "w")
  write_project11_start(P, F)
  print('  <PropertyGroup>', file=F)
  print('    <_ProjectFileVersion>11.0.60610.1</_ProjectFileVersion>', file=F)
  print('  </PropertyGroup>', file=F)
  for c in configs:
    print('''  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='%s'">''' % (c.name(), ), file=F)
    print('    <OutDir>$(SolutionDir)bin\\$(Platform)\\$(Configuration)\\</OutDir>', file=F)
    print('    <IntDir>$(OutDir)obj\\$(ProjectName)\\</IntDir>', file=F)
    print('    <TargetName>$(ProjectName)_c</TargetName>', file=F)
    if c.x64:
      print('    <TargetExt>.mexw64</TargetExt>', file=F)
    else:
      print('    <TargetExt>.mexw32</TargetExt>', file=F)
    print('  </PropertyGroup>', file=F)
  for c in configs:
    print('''  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='%s'">''' % (c.name(), ), file=F)
    print('    <ClCompile>', file=F)
    if c.debug:
      print('      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>', file=F)
    else:
      print('      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>', file=F)
#    print('      <WarningLevel>Level3</WarningLevel>', file=F)
    #print('      <AdditionalIncludeDirectories>$(MATLAB_ROOT)\extern\include\;..\..\lib\include;..\..\include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>', file=F)
    # FIXME: This CUDA_PATH shouldn't be necessary
    print('      <AdditionalIncludeDirectories>$(MATLAB_ROOT)\extern\include\;$(CUDA_PATH)\include;..\..\lib\include;..\..\include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>', file=F)
    print('      <OpenMPSupport>true</OpenMPSupport>', file=F)
    if not c.x64: # /arch:SSE2 is implicit on x64
      print('      <EnableEnhancedInstructionSet>StreamingSIMDExtensions2</EnableEnhancedInstructionSet>', file=F)
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
#    d+="DLL_EXPORTS;_CRT_SECURE_NO_WARNINGS;"
    d+='%(PreprocessorDefinitions)</PreprocessorDefinitions>'
    print(d, file=F)
    print('      <MultiProcessorCompilation>true</MultiProcessorCompilation>', file=F)
#    print('      <SDLCheck>true</SDLCheck>', file=F)
#   if c.debug:
#   <DebugInformationFormat>EditAndContinue</DebugInformationFormat> ??
    print('    </ClCompile>', file=F)
    print('    <Link>', file=F)
#    if not c.debug:
#      print('      <EnableCOMDATFolding>true</EnableCOMDATFolding>', file=F)
#      print('      <OptimizeReferences>true</OptimizeReferences>', file=F)
    if c.x64:
      print('      <OutputFile>$(OutDir)$(ProjectName)_c.mexw64</OutputFile>', file=F)
    else:
      print('      <OutputFile>$(OutDir)$(ProjectName)_c.mexw32</OutputFile>', file=F)
    print('      <AdditionalDependencies>%s.lib;libmex.lib;libmx.lib;libut.lib;%%(AdditionalDependencies)</AdditionalDependencies>' % (c.target(), ), file=F)
    l = '      <AdditionalLibraryDirectories>';
    if c.x64:
      l += '..\\..\\lib\\x64\\;..\\..\\bin\\x64\\'
    else:
      l += '..\\..\\lib\\win32\\;..\\..\\bin\\win32\\'
    l += c.config()
    if c.x64:
      l += ';$(MATLAB_ROOT)\extern\lib\win64\microsoft'
    else:
      l += ';$(MATLAB_ROOT)\extern\lib\win32\microsoft'
    l += ';%(AdditionalLibraryDirectories)'
    l += '</AdditionalLibraryDirectories>'
    print(l, file=F)
    print('      <ModuleDefinitionFile>mex.def</ModuleDefinitionFile>', file=F)
    print('      <GenerateDebugInformation>true</GenerateDebugInformation>', file=F)
    print('    </Link>', file=F)
    print('  </ItemDefinitionGroup>', file=F)
  write_project11_end(P, F)
  F.close()

def write_main_filters11():
  P = P_astra
  F = open(P["name"] + ".vcxproj.filters", "w")
  print(bom + '<?xml version="1.0" encoding="utf-8"?>', file=F)
  print('<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">', file=F)
  print('  <ItemGroup>', file=F)
  for Filter in P_astra["filter_names"]:
    L = P_astra["filters"][Filter][1:]
    l = [ f for f in L if len(f) > 3 and f[-3:] == ".cu" ]
    for f in l:
      print('    <CudaCompile Include="' + f + '">', file=F)
      print('      <Filter>' + Filter + '</Filter>', file=F)
      print('    </CudaCompile>', file=F)
  print('  </ItemGroup>', file=F)
  print('  <ItemGroup>', file=F)
  for Filter in P_astra["filter_names"]:
    L = P_astra["filters"][Filter][1:]
    l = [ f for f in L if len(f) > 4 and f[-4:] == ".cpp" ]
    for f in l:
      print('    <ClCompile Include="' + f + '">', file=F)
      print('      <Filter>' + Filter + '</Filter>', file=F)
      print('    </ClCompile>', file=F)
  print('  </ItemGroup>', file=F)
  print('  <ItemGroup>', file=F)
  for Filter in P_astra["filter_names"]:
    L = P_astra["filters"][Filter][1:]
    l = [ f for f in L if len(f) > 2 and f[-2:] == ".h" ]
    for f in l:
      print('    <ClInclude Include="' + f + '">', file=F)
      print('      <Filter>' + Filter + '</Filter>', file=F)
      print('    </ClInclude>', file=F)
  print('  </ItemGroup>', file=F)
  print('  <ItemGroup>', file=F)
  for Filter in P_astra["filter_names"]:
    L = P_astra["filters"][Filter][1:]
    l = [ f for f in L if len(f) > 4 and f[-4:] == ".inl" ]
    for f in l:
      print('    <None Include="' + f + '">', file=F)
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

def write_project09_start(P, F):
  print('<?xml version="1.0" encoding="Windows-1252"?>', file=F)
  print('<VisualStudioProject', file=F)
  print('\tProjectType="Visual C++"', file=F)
  print('\tVersion="9.00"', file=F)
  if "mex" in P["name"]:
    print('\tName="%s"' % (P["name"], ), file=F)
  print('\tProjectGUID="{%s}"' % (P["uuid09"],), file=F)
  if "mex" in P["name"]:
    print('\tRootNamespace="astraMatlab"', file=F)
  else:
    print('\tRootNamespace="astra"', file=F)
  print('\tTargetFrameworkVersion="131072"', file=F)
  print('\t>', file=F)
  print(r'''	<Platforms>
		<Platform
			Name="Win32"
		/>
		<Platform
			Name="x64"
		/>
	</Platforms>''', file=F)

def write_project09_unused_tools(F):
    print(r'''			<Tool
				Name="VCPreBuildEventTool"
			/>
			<Tool
				Name="VCCustomBuildTool"
			/>
			<Tool
				Name="VCXMLDataGeneratorTool"
			/>
			<Tool
				Name="VCWebServiceProxyGeneratorTool"
			/>
			<Tool
				Name="VCMIDLTool"
			/>
			<Tool
				Name="VCManagedResourceCompilerTool"
			/>
			<Tool
				Name="VCResourceCompilerTool"
			/>
			<Tool
				Name="VCPreLinkEventTool"
			/>
			<Tool
				Name="VCALinkTool"
			/>
			<Tool
				Name="VCManifestTool"
			/>
			<Tool
				Name="VCXDCMakeTool"
			/>
			<Tool
				Name="VCBscMakeTool"
			/>
			<Tool
				Name="VCFxCopTool"
			/>
			<Tool
				Name="VCAppVerifierTool"
			/>
			<Tool
				Name="VCPostBuildEventTool"
			/>''', file=F)


def write_main_project09():
  P = P_astra;
  F = open(P["file09"], "w")
  write_project09_start(P, F)
  print(r'''	<ToolFiles>
		<DefaultToolFile
			FileName="NvCudaRuntimeApi.v5.5.rules"
		/>
	</ToolFiles>''', file=F)
  print('\t<Configurations>', file=F)
  for c in configs:
    print('\t\t<Configuration', file=F)
    print('\t\t\tName="%s"' % (c.name(), ), file=F)
    print('\t\t\tOutputDirectory="$(SolutionDir)bin\$(PlatformName)\%s"' % (c.config(), ), file=F)
    print(r'''			IntermediateDirectory="$(OutDir)/obj"
			ConfigurationType="2"
			>''', file=F)
    write_project09_unused_tools(F)
    print('\t\t\t<Tool', file=F)
    print('\t\t\t\tName="VCCLCompilerTool"', file=F)
    if c.cuda:
      print('\t\t\t\tAdditionalIncludeDirectories="&quot;$(CUDA_INC_PATH)&quot;;lib\\include;include"', file=F)
      print('\t\t\t\tPreprocessorDefinitions="ASTRA_CUDA;DLL_EXPORTS;__SSE2__"', file=F)
    else:
      print('\t\t\t\tAdditionalIncludeDirectories="lib\\include;include"', file=F)
      print('\t\t\t\tPreprocessorDefinitions="DLL_EXPORTS;__SSE2__"', file=F)
    if c.debug:
      print(r'''				Optimization="0"
				InlineFunctionExpansion="0"
				FavorSizeOrSpeed="0"
				EnableFiberSafeOptimizations="false"
				WholeProgramOptimization="false"
				RuntimeLibrary="3"''', file=F)
    else:
      print(r'''				Optimization="3"
				InlineFunctionExpansion="2"
				FavorSizeOrSpeed="1"
				RuntimeLibrary="2"''', file=F)
    if not c.x64: # /arch:SSE2 is implicit on x64
      print('\t\t\t\tEnableEnhancedInstructionSet="2"', file=F) # SSE2
    print('\t\t\t\tOpenMP="true"', file=F)
    print('\t\t\t\tAdditionalOptions="/MP"', file=F) # build with multiple processes
    print('\t\t\t/>', file=F)
    print('\t\t\t<Tool', file=F)
    print('\t\t\t\tName="VCLinkerTool"', file=F)
    if c.cuda:
      print('\t\t\t\tAdditionalDependencies="cudart.lib cufft.lib"', file=F)
    print('\t\t\t\tOutputFile="bin\\%s\\%s.dll"' % (c.platform(), c.target()), file=F)
    if c.cuda:
      print('\t\t\t\tAdditionalLibraryDirectories="&quot;.\\lib\\%s&quot;;&quot;$(CUDA_LIB_PATH)&quot;"' % (c.platform(), ), file=F)
    else:
      print('\t\t\t\tAdditionalLibraryDirectories="&quot;.\\lib\\%s&quot;"' % (c.platform(), ), file=F)
    print('\t\t\t\tGenerateManifest="true"', file=F)
    print('\t\t\t\tModuleDefinitionFile=""', file=F)
    if c.debug:
      print('\t\t\t\tGenerateDebugInformation="true"', file=F)
    if c.x64:
      print('\t\t\t\tTargetMachine="17"', file=F) # x64
    else:
      print('\t\t\t\tTargetMachine="1"', file=F) # x86
    print('\t\t\t/>', file=F)
    print('\t\t\t<Tool', file=F)
    print('\t\t\t\tName="Cudart Build Rule"', file=F)
    print('\t\t\t\tArch1="20"', file=F)
    print('\t\t\t\tArch2="30"', file=F)
    print('\t\t\t\tArch3="35"', file=F)
    if c.x64:
      print('\t\t\t\tTargetMachinePlatform="1"', file=F) # x64
    else:
      print('\t\t\t\tTargetMachinePlatform="0"', file=F) # x86
    if c.debug:
      print('\t\t\t\tRuntime="3"', file=F) # MDD
    else:
      print('\t\t\t\tRuntime="2"', file=F) # MD
    print('\t\t\t\tExtraCppOptions="-Iinclude -Ilib/include"', file=F)
    if c.cuda:
      print('\t\t\t\tDefines="ASTRA_CUDA;DLL_EXPORTS"', file=F)
    else: # This 'else' doesn't make much sense
      print('\t\t\t\tDefines="DLL_EXPORTS"', file=F)
    # TODO!!!
    print('\t\t\t/>', file=F)
    print('\t\t</Configuration>', file=F)
  print('\t</Configurations>', file=F)
  print('\t<References>', file=F)
  print('\t</References>', file=F)
  print('\t<Files>', file=F)
  print(r'''		<Filter
			Name="Resource Files"
			Filter="rc;ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe;resx;tiff;tif;png;wav"
			UniqueIdentifier="{67DA6AB6-F800-4c08-8B7A-83BB121AAD01}"
			>
			<File
				RelativePath=".\src\astra.def"
				>
			</File>
		</Filter>''', file=F)
  curgroup = None
  for Filter in P["filter_names"]:
    if "\\" not in Filter:
      continue
    # TODO
    [ group, subgroup ] = Filter.split("\\")
    if group != curgroup:
      if curgroup != None:
        print('\t\t</Filter>', file=F)
      print('\t\t<Filter', file=F)
      print('\t\t\tName="%s"' % (group, ), file=F)
      print('\t\t\t>', file=F)
      curgroup = group
    print('\t\t\t<Filter', file=F)
    print('\t\t\t\tName="%s"' % (subgroup, ), file=F)
    print('\t\t\t\t>', file=F)
    for f in P["filters"][Filter][1:]:
      print('\t\t\t\t<File', file=F)
      print('\t\t\t\t\tRelativePath=".\\%s"' % (f, ), file=F)
      print('\t\t\t\t\t>', file=F)
      if (("Cuda" in f) or ("cuda" in f)) and not (f[-2:] == ".h"):
        for c in configs:
          if not c.cuda:
            print('\t\t\t\t\t<FileConfiguration', file=F)
            print('\t\t\t\t\t\tName="%s"' % (c.name(), ), file=F)
            print('\t\t\t\t\t\tExcludedFromBuild="true"', file=F)
            print('\t\t\t\t\t\t>', file=F)
            print('\t\t\t\t\t\t<Tool', file=F)
            if len(f) > 3 and f[-3:] == ".cu":
              print('\t\t\t\t\t\t\tName="Cudart Build Rule"', file=F)
            else:
              print('\t\t\t\t\t\t\tName="VCCLCompilerTool"', file=F)
            print('\t\t\t\t\t\t/>', file=F)
            print('\t\t\t\t\t</FileConfiguration>', file=F)
      print('\t\t\t\t</File>', file=F)
    print('\t\t\t</Filter>', file=F)
  print('\t\t</Filter>', file=F)
  print('\t</Files>', file=F)
  print('\t<Globals>', file=F)
  print('\t</Globals>', file=F)
  print('</VisualStudioProject>', file=F)
  F.close()

def write_mex_project09(P):
  F = open("matlab/mex/" + P["name"] + "_vc09.vcproj", "w")
  write_project09_start(P, F)
  print('\t<ToolFiles>', file=F)
  print('\t</ToolFiles>', file=F)
  print('\t<Configurations>', file=F)
  for c in configs:
    print('\t\t<Configuration', file=F)
    print('\t\t\tName="%s"' % (c.name(), ), file=F)
    print('\t\t\tOutputDirectory="$(SolutionDir)bin\$(PlatformName)\$(ConfigurationName)"', file=F)
    print(r'''			IntermediateDirectory="$(OutDir)\obj\$(ProjectName)"
			ConfigurationType="2"
			>''', file=F)
    write_project09_unused_tools(F)
    print('\t\t\t<Tool', file=F)
    print('\t\t\t\tName="VCCLCompilerTool"', file=F)
    if c.cuda:
      print('\t\t\t\tAdditionalIncludeDirectories="$(MATLAB_ROOT)\\extern\\include\\;&quot;$(CUDA_INC_PATH)&quot;;..\\..\\lib\\include;..\\..\\include"', file=F)
      print('\t\t\t\tPreprocessorDefinitions="ASTRA_CUDA;__SSE2__"', file=F)
    else:
      print('\t\t\t\tAdditionalIncludeDirectories="$(MATLAB_ROOT)\\extern\\include\\;..\\..\\lib\\include;..\\..\\include"', file=F)
      print('\t\t\t\tPreprocessorDefinitions="__SSE2__"', file=F)
    if c.debug:
      print(r'''				Optimization="0"
				RuntimeLibrary="3"''', file=F)
    else:
      print(r'''				Optimization="2"
				RuntimeLibrary="2"''', file=F)
    if not c.x64: # /arch:SSE2 is implicit on x64
      print('\t\t\t\tEnableEnhancedInstructionSet="2"', file=F) # SSE2
    print('\t\t\t\tOpenMP="true"', file=F)
    print('\t\t\t\tAdditionalOptions="/MP"', file=F) # build with multiple processes
    print('\t\t\t/>', file=F)
    print('\t\t\t<Tool', file=F)
    print('\t\t\t\tName="VCLinkerTool"', file=F)
    print('\t\t\t\tAdditionalDependencies="%s.lib libmex.lib libmx.lib libut.lib"' % (c.target(), ), file=F)
    if c.x64:
      print('\t\t\t\tOutputFile="$(OutDir)\\$(ProjectName)_c.mexw64"', file=F)
    else:
      print('\t\t\t\tOutputFile="$(OutDir)\\$(ProjectName)_c.mexw32"', file=F)
    if c.x64:
      print('\t\t\t\tAdditionalLibraryDirectories="..\\..\\bin\\x64;$(MATLAB_ROOT)\\extern\\lib\\win64\\microsoft;..\\..\\lib\\x64"', file=F)
    else:
      print('\t\t\t\tAdditionalLibraryDirectories="..\\..\\bin\\win32;$(MATLAB_ROOT)\\extern\\lib\\win32\\microsoft;..\\..\\lib\\win32"', file=F)
    print('\t\t\t\tModuleDefinitionFile="mex.def"', file=F)
    if c.debug:
      print('\t\t\t\tGenerateDebugInformation="true"', file=F)
    else:
      print('\t\t\t\tGenerateDebugInformation="false"', file=F)
    if c.x64:
      print('\t\t\t\tTargetMachine="17"', file=F) # x64
    else:
      print('\t\t\t\tTargetMachine="1"', file=F) # x86
    print('\t\t\t/>', file=F)
    print('\t\t</Configuration>', file=F)
  print('\t</Configurations>', file=F)
  print('\t<References>', file=F)
  print('\t</References>', file=F)
  print('\t<Files>', file=F)
  for f in P["files"]:
    print('\t\t<File', file=F)
    print('\t\t\tRelativePath=".\\%s"' % (f, ), file=F)
    print('\t\t\t>', file=F)
    print('\t\t</File>', file=F)
  print('\t</Files>', file=F)
  print('\t<Globals>', file=F)
  print('\t</Globals>', file=F)
  print('</VisualStudioProject>', file=F)



if (len(sys.argv) != 2) or (sys.argv[1] not in ["vc09", "vc11", "all"]):
  print("Usage: python gen.py [vc09|vc11|all]", file=sys.stderr)
  sys.exit(1)



try:
  open("../../src/AstraObjectManager.cpp", "r")
except IOError:
  print("Run gen.py from the build/msvc directory", file=sys.stderr)
  sys.exit(1)

# Change directory to main dir
os.chdir("../..")

if sys.argv[1] in ["vc11", "all"]:
  # HACK
  P_astra["name"] = "astra_vc11"
  write_sln(11)
  write_main_project11()
  write_main_filters11()
  write_mex_project11(P0)
  write_mex_project11(P1)
  write_mex_project11(P2)
  write_mex_project11(P3)
  write_mex_project11(P4)
  write_mex_project11(P5)
  write_mex_project11(P6)
  write_mex_project11(P7)
  write_mex_project11(P8)

if sys.argv[1] in ["vc09", "all"]:
  # HACK
  P_astra["name"] = "astra"

  write_sln(9)
  write_main_project09()
  write_mex_project09(P0)
  write_mex_project09(P1)
  write_mex_project09(P2)
  write_mex_project09(P3)
  write_mex_project09(P4)
  write_mex_project09(P5)
  write_mex_project09(P6)
  write_mex_project09(P7)
  write_mex_project09(P8)
