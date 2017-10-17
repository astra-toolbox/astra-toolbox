#include "creators.hpp"

#include <iostream>
#include <fstream>
#include <zlib.h>

#include <astra/ForwardProjectionAlgorithm.h>
#include <astra/CudaForwardProjectionAlgorithm.h>
#include <astra/CudaForwardProjectionAlgorithm3D.h>

#define GZBUFSIZE 0x1000000L /* 16 MiB */

// create_angles
float32* create_angles(double lo, double hi, int ang_count)
{
    float32 *angles = new float32[ang_count];
    for(int i = 0; i < ang_count; i++)
        angles[i] = fma((double)i/ang_count,hi-lo,lo);

    return angles;
}

// create_vol_geom
// 2D
CVolumeGeometry2D* create_vol_geom_2d(int N)
{
    return new CVolumeGeometry2D(N,N);
}

CVolumeGeometry2D* create_vol_geom_2d(int Y, int X)
{
    return new CVolumeGeometry2D(X,Y);
}

CVolumeGeometry2D* create_vol_geom_2d(int Y, int X, float32 min_x, float32 max_x, float32 min_y, float32 max_y)
{
    return new CVolumeGeometry2D(X,Y,min_x,min_y,max_x,max_y);
}
// 3D
CVolumeGeometry3D* create_vol_geom_3d(int N, int Z)
{
    return new CVolumeGeometry3D(N,N,Z);
}

CVolumeGeometry3D* create_vol_geom_3d(int Y, int X, int Z)
{
    return new CVolumeGeometry3D(X,Y,Z);
}

CVolumeGeometry3D* create_vol_geom_3d(int Y, int X, int Z, float32 min_x, float32 max_x, float32 min_y, float32 max_y, float32 min_z, float32 max_z)
{
    return new CVolumeGeometry3D(X,Y,Z,min_x,min_y,min_z,max_x,max_y,max_z);
}

// create_proj_geom
// 2D
CParallelProjectionGeometry2D* create_proj_geom_2d_parallel(float32 det_width, int det_count, int ang_count, const float32 *angles, const float32 *extra_det_offset)
{
    return new CParallelProjectionGeometry2D(ang_count,det_count,det_width,angles,extra_det_offset);
}

CFanFlatProjectionGeometry2D* create_proj_geom_2d_fanflat(float32 det_width, int det_count, int ang_count, const float32 *angles, float32 origin_src, float32 origin_det)
{
    return new CFanFlatProjectionGeometry2D(ang_count,det_count,det_width,angles,origin_src,origin_det);
}

CFanFlatVecProjectionGeometry2D* create_proj_geom_2d_fanflat_vec(int det_count, int ang_count, const double *vectors)
{
    SFanProjection *angles = new SFanProjection[ang_count];
    
    for(int i = 0; i < ang_count; i++) {
        angles[i].fSrcX  = vectors[6*i + 0];
        angles[i].fSrcY  = vectors[6*i + 1];
        angles[i].fDetUX = vectors[6*i + 4];
        angles[i].fDetUY = vectors[6*i + 5];
        angles[i].fDetSX = vectors[6*i + 2] - 0.5f * det_count * angles[i].fDetUX;
        angles[i].fDetSY = vectors[6*i + 3] - 0.5f * det_count * angles[i].fDetUY;
    }

    return new CFanFlatVecProjectionGeometry2D(ang_count,det_count,angles);
}

// TODO: CSparseMatrixProjectionGeometry2D

// 3D
CParallelProjectionGeometry3D* create_proj_geom_3d_parallel(float32 det_width, float32 det_height, int det_row_count, int det_col_count, int ang_count, const float32 *angles)
{
    return new CParallelProjectionGeometry3D(ang_count,det_row_count,det_col_count,det_width,det_height,angles);
}

CConeProjectionGeometry3D* create_proj_geom_3d_cone(float32 det_width, float32 det_height, int det_row_count, int det_col_count, int ang_count, const float32 *angles, float32 origin_src, float32 origin_det)
{
    return new CConeProjectionGeometry3D(ang_count,det_row_count,det_col_count,det_width,det_height,angles,origin_src,origin_det);
}

CParallelVecProjectionGeometry3D* create_proj_geom_3d_parallel_vec(int det_row_count, int det_col_count, int ang_count, const double *vectors)
{
    SPar3DProjection *angles = new SPar3DProjection[ang_count];

    for(int i = 0; i < ang_count; i++) {
        angles[i].fRayX  = vectors[12*i + 0];
        angles[i].fRayY  = vectors[12*i + 1];
        angles[i].fRayZ  = vectors[12*i + 2];
        angles[i].fDetUX = vectors[12*i + 6];
        angles[i].fDetUY = vectors[12*i + 7];
        angles[i].fDetUZ = vectors[12*i + 8];
        angles[i].fDetVX = vectors[12*i + 9];
        angles[i].fDetVY = vectors[12*i + 10];
        angles[i].fDetVZ = vectors[12*i + 11];
        angles[i].fDetSX = vectors[12*i + 3] - 0.5f * det_row_count * angles[i].fDetVX - 0.5f * det_col_count * angles[i].fDetUX;
        angles[i].fDetSY = vectors[12*i + 4] - 0.5f * det_row_count * angles[i].fDetVY - 0.5f * det_col_count * angles[i].fDetUY;
        angles[i].fDetSZ = vectors[12*i + 5] - 0.5f * det_row_count * angles[i].fDetVZ - 0.5f * det_col_count * angles[i].fDetUZ;
    }

    return new CParallelVecProjectionGeometry3D(ang_count,det_row_count,det_col_count,angles);
}

CConeVecProjectionGeometry3D* create_proj_geom_3d_cone_vec(int det_row_count, int det_col_count, int ang_count, const double *vectors)
{
    SConeProjection *angles = new SConeProjection[ang_count];

    for(int i = 0; i < ang_count; i++) {
        angles[i].fSrcX  = vectors[12*i + 0];
        angles[i].fSrcY  = vectors[12*i + 1];
        angles[i].fSrcZ  = vectors[12*i + 2];
        angles[i].fDetUX = vectors[12*i + 6];
        angles[i].fDetUY = vectors[12*i + 7];
        angles[i].fDetUZ = vectors[12*i + 8];
        angles[i].fDetVX = vectors[12*i + 9];
        angles[i].fDetVY = vectors[12*i + 10];
        angles[i].fDetVZ = vectors[12*i + 11];
        angles[i].fDetSX = vectors[12*i + 3] - 0.5f * det_row_count * angles[i].fDetVX - 0.5f * det_col_count * angles[i].fDetUX;
        angles[i].fDetSY = vectors[12*i + 4] - 0.5f * det_row_count * angles[i].fDetVY - 0.5f * det_col_count * angles[i].fDetUY;
        angles[i].fDetSZ = vectors[12*i + 5] - 0.5f * det_row_count * angles[i].fDetVZ - 0.5f * det_col_count * angles[i].fDetUZ;
    }

    return new CConeVecProjectionGeometry3D(ang_count,det_row_count,det_col_count,angles);
}

// create_projector
// 2D
CParallelBeamLineKernelProjector2D* create_projector_2d_line(CParallelProjectionGeometry2D *proj_geom, CVolumeGeometry2D *vol_geom)
{
    return new CParallelBeamLineKernelProjector2D(proj_geom,vol_geom);
}

CParallelBeamStripKernelProjector2D* create_projector_2d_strip(CParallelProjectionGeometry2D *proj_geom, CVolumeGeometry2D *vol_geom)
{
    return new CParallelBeamStripKernelProjector2D(proj_geom,vol_geom);
}

CParallelBeamLinearKernelProjector2D* create_projector_2d_linear(CParallelProjectionGeometry2D *proj_geom, CVolumeGeometry2D *vol_geom)
{
    return new CParallelBeamLinearKernelProjector2D(proj_geom,vol_geom);
}

CFanFlatBeamLineKernelProjector2D* create_projector_2d_line_fanflat(CFanFlatProjectionGeometry2D *proj_geom, CVolumeGeometry2D *vol_geom)
{
    return new CFanFlatBeamLineKernelProjector2D(proj_geom,vol_geom);
}

CFanFlatBeamStripKernelProjector2D* create_projector_2d_strip_fanflat(CFanFlatProjectionGeometry2D *proj_geom, CVolumeGeometry2D *vol_geom)
{
    return new CFanFlatBeamStripKernelProjector2D(proj_geom,vol_geom);
}

// TODO: CSparseMatrixProjector2D
// TODO: CParallelBeamBlobKernelProjector2D

CCudaProjector2D* create_projector_2d_cuda(CProjectionGeometry2D *proj_geom, CVolumeGeometry2D *vol_geom)
{
    return new CCudaProjector2D(proj_geom,vol_geom);
}
// 3D
CCudaProjector3D* create_projector_3d_cuda(CProjectionGeometry3D *proj_geom, CVolumeGeometry3D *vol_geom)
{
    return new CCudaProjector3D(proj_geom,vol_geom);
}

// create_data
// 2D
CFloat32VolumeData2D* create_data_2d_vol(CVolumeGeometry2D *vol_geom)
{
    return new CFloat32VolumeData2D(vol_geom);
}

CFloat32VolumeData2D* create_data_2d_vol(CVolumeGeometry2D *vol_geom, float32 value)
{
    return new CFloat32VolumeData2D(vol_geom,value);
}

CFloat32VolumeData2D* create_data_2d_vol(CVolumeGeometry2D *vol_geom, const float32 *data)
{
    return new CFloat32VolumeData2D(vol_geom,(float32*)data);
}

CFloat32ProjectionData2D* create_data_2d_sino(CProjectionGeometry2D *proj_geom)
{
    return new CFloat32ProjectionData2D(proj_geom);
}

CFloat32ProjectionData2D* create_data_2d_sino(CProjectionGeometry2D *proj_geom, float32 value)
{
    return new CFloat32ProjectionData2D(proj_geom,value);
}

CFloat32ProjectionData2D* create_data_2d_sino(CProjectionGeometry2D *proj_geom, const float32 *data)
{
    return new CFloat32ProjectionData2D(proj_geom,(float32*)data);
}
// 3D
CFloat32VolumeData3DMemory* create_data_3d_vol(CVolumeGeometry3D *vol_geom)
{
    return new CFloat32VolumeData3DMemory(vol_geom);
}

CFloat32VolumeData3DMemory* create_data_3d_vol(CVolumeGeometry3D *vol_geom, float32 value)
{
    return new CFloat32VolumeData3DMemory(vol_geom,value);
}

CFloat32VolumeData3DMemory* create_data_3d_vol(CVolumeGeometry3D *vol_geom, const float32 *data)
{
    return new CFloat32VolumeData3DMemory(vol_geom,(float32*)data);
}

CFloat32ProjectionData3DMemory* create_data_3d_sino(CProjectionGeometry3D *proj_geom)
{
    return new CFloat32ProjectionData3DMemory(proj_geom);
}

CFloat32ProjectionData3DMemory* create_data_3d_sino(CProjectionGeometry3D *proj_geom, float32 value)
{
    return new CFloat32ProjectionData3DMemory(proj_geom,value);
}

CFloat32ProjectionData3DMemory* create_data_3d_sino(CProjectionGeometry3D *proj_geom, const float32 *data)
{
    return new CFloat32ProjectionData3DMemory(proj_geom,(float32*)data);
}

// save_data
// 2D
void save_data_2d(const char *filename, CFloat32Data2D *data)
{
    const float32 *fdat;
    size_t size;
    FILE *file;

    fdat = data->getDataConst();
    size = (size_t)data->getWidth() * data->getHeight();
    file = fopen(filename,"wb");
    ASTRA_ASSERT(file != NULL);
    ASTRA_ASSERT(fwrite(fdat,sizeof(float32),size,file) == size);
    fclose(file);
}
// 2D (gzipped)
void save_data_2d_gz(const char *filename, CFloat32Data2D *data)
{
    const float32 *fdat;
    size_t size;
    gzFile zfp;

    fdat = data->getDataConst();
    size = (size_t)data->getWidth() * data->getHeight();
    zfp = gzopen(filename,"wb");
    ASTRA_ASSERT(zfp != NULL);
    gzbuffer(zfp,GZBUFSIZE);
    ASTRA_ASSERT(gzfwrite(fdat,sizeof(float32),size,zfp) == size);
    gzclose(zfp);
}
// 3D
void save_data_3d(const char *filename, CFloat32Data3DMemory *data)
{
    const float32 *fdat;
    size_t size;
    FILE *file;

    fdat = data->getDataConst();
    size = (size_t)data->getWidth() * data->getHeight() * data->getDepth();
    file = fopen(filename,"wb");
    ASTRA_ASSERT(file != NULL);
    ASTRA_ASSERT(fwrite(fdat,sizeof(float32),size,file) == size);
    fclose(file);
}
// 3D (gzipped)
void save_data_3d_gz(const char *filename, CFloat32Data3DMemory *data)
{
    const float32 *fdat;
    size_t size;
    gzFile zfp;

    fdat = data->getDataConst();
    size = (size_t)data->getWidth() * data->getHeight() * data->getDepth();
    zfp = gzopen(filename,"wb");
    ASTRA_ASSERT(zfp != NULL);
    gzbuffer(zfp,GZBUFSIZE);
    ASTRA_ASSERT(gzfwrite(fdat,sizeof(float32),size,zfp) == size);
    gzclose(zfp);
}

// load_data
// vol
CFloat32VolumeData3DMemory* load_data_3d_vol(const char *filename, CVolumeGeometry3D *vol_geom)
{
    float32 *fdat;
    size_t size;
    FILE *file;
    int ret;

    file = fopen(filename,"rb");
    ASTRA_ASSERT(file != NULL);
    size = (size_t)vol_geom->getGridColCount() * vol_geom->getGridRowCount() * vol_geom->getGridSliceCount();
    ret = posix_memalign((void**)&fdat,16,size*sizeof(float32));
    ASTRA_ASSERT(ret == 0);
    ASTRA_ASSERT(fread(fdat,sizeof(float32),size,file) == size);
    fclose(file);
 
    return new CFloat32VolumeData3DMemory(vol_geom,new CFloat32CustomAstra(fdat));
}
// vol reuse memory
CFloat32VolumeData3DMemory* load_data_3d_vol(const char *filename, CFloat32VolumeData3DMemory *vol_data)
{
    float32 *fdat;
    size_t size;
    FILE *file;

    file = fopen(filename,"rb");
    ASTRA_ASSERT(file != NULL);
    size = (size_t)vol_data->getWidth() * vol_data->getHeight() * vol_data->getDepth();
    fdat = vol_data->getData();
    ASTRA_ASSERT(fread(fdat,sizeof(float32),size,file) == size);
    fclose(file);

    return vol_data;
}
// vol (gzipped)
CFloat32VolumeData3DMemory* load_data_3d_vol_gz(const char *filename, CVolumeGeometry3D *vol_geom)
{
    float32 *fdat;
    size_t size;
    gzFile zfp;
    int ret;

    zfp = gzopen(filename,"rb");
    ASTRA_ASSERT(zfp != NULL);
    gzbuffer(zfp,GZBUFSIZE);
    size = (size_t)vol_geom->getGridColCount() * vol_geom->getGridRowCount() * vol_geom->getGridSliceCount();
    ret = posix_memalign((void**)&fdat,16,size*sizeof(float32));
    ASTRA_ASSERT(ret == 0);
    ASTRA_ASSERT(gzfread(fdat,sizeof(float32),size,zfp) == size);
    gzclose(zfp);

    return new CFloat32VolumeData3DMemory(vol_geom,new CFloat32CustomAstra(fdat));
}
// vol reuse memory (gzipped)
CFloat32VolumeData3DMemory* load_data_3d_vol_gz(const char *filename, CFloat32VolumeData3DMemory *vol_data)
{
    float32 *fdat;
    size_t size;
    gzFile zfp;

    zfp = gzopen(filename,"rb");
    ASTRA_ASSERT(zfp != NULL);
    gzbuffer(zfp,GZBUFSIZE);
    size = (size_t)vol_data->getWidth() * vol_data->getHeight() * vol_data->getDepth();
    fdat = vol_data->getData();
    ASTRA_ASSERT(gzfread(fdat,sizeof(float32),size,zfp) == size);
    gzclose(zfp);

    return vol_data;
}
// sino
CFloat32ProjectionData3DMemory* load_data_3d_sino(const char *filename, CProjectionGeometry3D *proj_geom)
{
    float32 *fdat;
    size_t size;
    FILE *file;
    int ret;

    file = fopen(filename,"rb");
    ASTRA_ASSERT(file != NULL);
    size = (size_t)proj_geom->getProjectionCount() * proj_geom->getDetectorRowCount() * proj_geom->getDetectorColCount();
    ret = posix_memalign((void**)&fdat,16,size*sizeof(float32));
    ASTRA_ASSERT(ret == 0);
    ASTRA_ASSERT(fread(fdat,sizeof(float32),size,file) == size);
    fclose(file);

    return new CFloat32ProjectionData3DMemory(proj_geom,new CFloat32CustomAstra(fdat));
}
// sino reuse memory
CFloat32ProjectionData3DMemory* load_data_3d_sino(const char *filename, CFloat32ProjectionData3DMemory *proj_data)
{
    float32 *fdat;
    size_t size;
    FILE *file;

    file = fopen(filename,"rb");
    ASTRA_ASSERT(file != NULL);
    size = (size_t)proj_data->getWidth() * proj_data->getHeight() * proj_data->getDepth();
    fdat = proj_data->getData();
    ASTRA_ASSERT(fread(fdat,sizeof(float32),size,file) == size);
    fclose(file);

    return proj_data;
}
// sino (gzipped)
CFloat32ProjectionData3DMemory* load_data_3d_sino_gz(const char *filename, CProjectionGeometry3D *proj_geom)
{
    float32 *fdat;
    size_t size;
    gzFile zfp;
    int ret;

    zfp = gzopen(filename,"rb");
    ASTRA_ASSERT(zfp != NULL);
    gzbuffer(zfp,GZBUFSIZE);
    size = (size_t)proj_geom->getProjectionCount() * proj_geom->getDetectorRowCount() * proj_geom->getDetectorColCount();
    ret = posix_memalign((void**)&fdat,16,size*sizeof(float32));
    ASTRA_ASSERT(ret == 0);
    ASTRA_ASSERT(gzfread(fdat,sizeof(float32),size,zfp) == size);
    gzclose(zfp);

    return new CFloat32ProjectionData3DMemory(proj_geom,new CFloat32CustomAstra(fdat));
}
// sino reuse memory (gzipped)
CFloat32ProjectionData3DMemory* load_data_3d_sino_gz(const char *filename, CFloat32ProjectionData3DMemory *proj_data)
{
    float32 *fdat;
    size_t size;
    gzFile zfp;

    zfp = gzopen(filename,"rb");
    ASTRA_ASSERT(zfp != NULL);
    gzbuffer(zfp,GZBUFSIZE);
    size = (size_t)proj_data->getWidth() * proj_data->getHeight() * proj_data->getDepth();
    fdat = proj_data->getData();
    ASTRA_ASSERT(gzfread(fdat,sizeof(float32),size,zfp) == size);
    gzclose(zfp);

    return proj_data;
}

// create_sino
// 2D
CFloat32ProjectionData2D* create_sino_2d(const float32 *data, CProjector2D *proj)
{
    CVolumeGeometry2D *vol_geom = proj->getVolumeGeometry();
    CProjectionGeometry2D *proj_geom = proj->getProjectionGeometry();
    CFloat32VolumeData2D *vol;
    CFloat32ProjectionData2D *sino;
    CForwardProjectionAlgorithm *algo_fp;

    vol = create_data_2d_vol(vol_geom,data);
    sino = create_data_2d_sino(proj_geom);

    algo_fp = new CForwardProjectionAlgorithm();
    algo_fp->initialize(proj,vol,sino);
    algo_fp->run();
    delete algo_fp;

    delete vol;

    return sino;
}

CFloat32ProjectionData2D* create_sino_2d(CFloat32VolumeData2D *vol, CProjector2D *proj)
{
    CProjectionGeometry2D *proj_geom = proj->getProjectionGeometry();
    CFloat32ProjectionData2D *sino;
    CForwardProjectionAlgorithm *algo_fp;

    sino = create_data_2d_sino(proj_geom);

    algo_fp = new CForwardProjectionAlgorithm();
    algo_fp->initialize(proj,vol,sino);
    algo_fp->run();
    delete algo_fp;

    return sino;
}

CFloat32ProjectionData2D* create_sino_2d_cuda(const float32 *data, CCudaProjector2D *proj)
{
    CVolumeGeometry2D *vol_geom = proj->getVolumeGeometry();
    CProjectionGeometry2D *proj_geom = proj->getProjectionGeometry();
    CFloat32VolumeData2D *vol;
    CFloat32ProjectionData2D *sino;
    CCudaForwardProjectionAlgorithm *algo_fp_cuda;

    vol = create_data_2d_vol(vol_geom,data);
    sino = create_data_2d_sino(proj_geom);

    algo_fp_cuda = new CCudaForwardProjectionAlgorithm();
    algo_fp_cuda->initialize(proj,vol,sino);
    algo_fp_cuda->run();
    delete algo_fp_cuda;

    delete vol;

    return sino;
}

CFloat32ProjectionData2D* create_sino_2d_cuda(CFloat32VolumeData2D *vol, CCudaProjector2D *proj)
{
    CProjectionGeometry2D *proj_geom = proj->getProjectionGeometry();
    CFloat32ProjectionData2D *sino;
    CCudaForwardProjectionAlgorithm *algo_fp_cuda;

    sino = create_data_2d_sino(proj_geom);

    algo_fp_cuda = new CCudaForwardProjectionAlgorithm();
    algo_fp_cuda->initialize(proj,vol,sino);
    algo_fp_cuda->run();
    delete algo_fp_cuda;

    return sino;
}
// 3D
CFloat32ProjectionData3DMemory* create_sino_3d_cuda(const float32 *data, CCudaProjector3D *proj)
{
    CVolumeGeometry3D *vol_geom = proj->getVolumeGeometry();
    CProjectionGeometry3D *proj_geom = proj->getProjectionGeometry();
    CFloat32VolumeData3DMemory *vol;
    CFloat32ProjectionData3DMemory *sino;
    CCudaForwardProjectionAlgorithm3D *algo_fp_cuda;

    vol = create_data_3d_vol(vol_geom,data);
    sino = create_data_3d_sino(proj_geom);

    algo_fp_cuda = new CCudaForwardProjectionAlgorithm3D();
    algo_fp_cuda->initialize(proj,sino,vol);
    algo_fp_cuda->run();
    delete algo_fp_cuda;

    delete vol;

    return sino;
}

CFloat32ProjectionData3DMemory* create_sino_3d_cuda(CFloat32VolumeData3DMemory *vol, CCudaProjector3D *proj)
{
    CProjectionGeometry3D *proj_geom = proj->getProjectionGeometry();
    CFloat32ProjectionData3DMemory *sino;
    CCudaForwardProjectionAlgorithm3D *algo_fp_cuda;

    sino = create_data_3d_sino(proj_geom);

    algo_fp_cuda = new CCudaForwardProjectionAlgorithm3D();
    algo_fp_cuda->initialize(proj,sino,vol);
    algo_fp_cuda->run();
    delete algo_fp_cuda;

    return sino;
}
