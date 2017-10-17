#include <astra/Float32VolumeData2D.h>
#include <astra/Float32ProjectionData2D.h>
#include <astra/Float32VolumeData3DMemory.h>
#include <astra/Float32ProjectionData3DMemory.h>

#include <astra/VolumeGeometry2D.h>
#include <astra/VolumeGeometry3D.h>

#include <astra/ParallelProjectionGeometry2D.h>
#include <astra/FanFlatProjectionGeometry2D.h>
#include <astra/FanFlatVecProjectionGeometry2D.h>

#include <astra/ParallelProjectionGeometry3D.h>
#include <astra/ConeProjectionGeometry3D.h>
#include <astra/ParallelVecProjectionGeometry3D.h>
#include <astra/ConeVecProjectionGeometry3D.h>

#include <astra/ParallelBeamLineKernelProjector2D.h>
#include <astra/ParallelBeamStripKernelProjector2D.h>
#include <astra/ParallelBeamLinearKernelProjector2D.h>
#include <astra/FanFlatBeamLineKernelProjector2D.h>
#include <astra/FanFlatBeamStripKernelProjector2D.h>
#include <astra/CudaProjector2D.h>
#include <astra/CudaProjector3D.h>

using namespace std;
using namespace astra;

class CFloat32CustomAstra : public CFloat32CustomMemory {
    public:
        CFloat32CustomAstra(float32 *fdat)
        {
            m_fPtr = fdat;
        }
        ~CFloat32CustomAstra()
        {
            free(m_fPtr);
        }
};

// create_angles
float32* create_angles(double lo, double hi, int ang_count);

// create_vol_geom
// 2D
CVolumeGeometry2D* create_vol_geom_2d(int N);
CVolumeGeometry2D* create_vol_geom_2d(int Y,int X);
CVolumeGeometry2D* create_vol_geom_2d(int Y,int X,float32 min_x,float32 max_x,float32 min_y,float32 max_y);
// 3D
CVolumeGeometry3D* create_vol_geom_3d(int N,int Z);
CVolumeGeometry3D* create_vol_geom_3d(int Y,int X,int Z);
CVolumeGeometry3D* create_vol_geom_3d(int Y,int X,int Z,float32 min_x,float32 max_x,float32 min_y,float32 max_y,float32 min_z,float32 max_z);

// create_proj_geom
// 2D
CParallelProjectionGeometry2D* create_proj_geom_2d_parallel(float32 det_width,int det_count,int ang_count,const float32 *angles,const float32 *extra_det_offset = NULL);
CFanFlatProjectionGeometry2D* create_proj_geom_2d_fanflat(float32 det_width,int det_count,int ang_count,const float32 *angles,float32 origin_src,float32 origin_det);
CFanFlatVecProjectionGeometry2D* create_proj_geom_2d_fanflat_vec(int det_count,int ang_count,const double *vectors);
// TODO: CSparseMatrixProjectionGeometry2D
// 3D
CParallelProjectionGeometry3D* create_proj_geom_3d_parallel(float32 det_width,float32 det_height,int det_row_count,int det_col_count,int ang_count,const float32 *angles);
CConeProjectionGeometry3D* create_proj_geom_3d_cone(float32 det_width,float32 det_height,int det_row_count,int det_col_count,int ang_count,const float32 *angles,float32 origin_src,float32 origin_det);
CParallelVecProjectionGeometry3D* create_proj_geom_3d_parallel_vec(int det_row_count,int det_col_count,int ang_count,const double *vectors);
CConeVecProjectionGeometry3D* create_proj_geom_3d_cone_vec(int det_row_count,int det_col_count,int ang_count,const double *vectors);

// create_projector
// 2D
CParallelBeamLineKernelProjector2D* create_projector_2d_line(CParallelProjectionGeometry2D *proj_geom,CVolumeGeometry2D *vol_geom);
CParallelBeamStripKernelProjector2D* create_projector_2d_strip(CParallelProjectionGeometry2D *proj_geom,CVolumeGeometry2D *vol_geom);
CParallelBeamLinearKernelProjector2D* create_projector_2d_linear(CParallelProjectionGeometry2D *proj_geom,CVolumeGeometry2D *vol_geom);
CFanFlatBeamLineKernelProjector2D* create_projector_2d_line_fanflat(CFanFlatProjectionGeometry2D *proj_geom,CVolumeGeometry2D *vol_geom);
CFanFlatBeamStripKernelProjector2D* create_projector_2d_strip_fanflat(CFanFlatProjectionGeometry2D *proj_geom,CVolumeGeometry2D *vol_geom);
// TODO: CSparseMatrixProjector2D
// TODO: CParallelBeamBlobKernelProjector2D
CCudaProjector2D* create_projector_2d_cuda(CProjectionGeometry2D *proj_geom,CVolumeGeometry2D *vol_geom);
// 3D
CCudaProjector3D* create_projector_3d_cuda(CProjectionGeometry3D *proj_geom,CVolumeGeometry3D *vol_geom);

// create_data
// 2D
CFloat32VolumeData2D* create_data_2d_vol(CVolumeGeometry2D *vol_geom);
CFloat32VolumeData2D* create_data_2d_vol(CVolumeGeometry2D *vol_geom,float32 value);
CFloat32VolumeData2D* create_data_2d_vol(CVolumeGeometry2D *vol_geom,const float32 *data);
CFloat32ProjectionData2D* create_data_2d_sino(CProjectionGeometry2D *proj_geom);
CFloat32ProjectionData2D* create_data_2d_sino(CProjectionGeometry2D *proj_geom,float32 value);
CFloat32ProjectionData2D* create_data_2d_sino(CProjectionGeometry2D *proj_geom,const float32 *data);
// 3D
CFloat32VolumeData3DMemory* create_data_3d_vol(CVolumeGeometry3D *vol_geom);
CFloat32VolumeData3DMemory* create_data_3d_vol(CVolumeGeometry3D *vol_geom,float32 value);
CFloat32VolumeData3DMemory* create_data_3d_vol(CVolumeGeometry3D *vol_geom,const float32 *data);
CFloat32ProjectionData3DMemory* create_data_3d_sino(CProjectionGeometry3D *proj_geom);
CFloat32ProjectionData3DMemory* create_data_3d_sino(CProjectionGeometry3D *proj_geom,float32 value);
CFloat32ProjectionData3DMemory* create_data_3d_sino(CProjectionGeometry3D *proj_geom,const float32 *data);

// save_data
// 2D
void save_data_2d(const char *filename,CFloat32Data2D *data);
void save_data_2d_gz(const char *filename,CFloat32Data2D *data);
// 3D
void save_data_3d(const char *filename,CFloat32Data3DMemory *data);
void save_data_3d_gz(const char *filename,CFloat32Data3DMemory *data);

// load data
// vol
CFloat32VolumeData3DMemory* load_data_3d_vol(const char *filename, CVolumeGeometry3D *vol_geom);
CFloat32VolumeData3DMemory* load_data_3d_vol(const char *filename, CFloat32VolumeData3DMemory *vol_data);
CFloat32VolumeData3DMemory* load_data_3d_vol_gz(const char *filename, CVolumeGeometry3D *vol_geom);
CFloat32VolumeData3DMemory* load_data_3d_vol_gz(const char *filename, CFloat32VolumeData3DMemory *vol_data);
// sino
CFloat32ProjectionData3DMemory* load_data_3d_sino(const char *filename, CProjectionGeometry3D *proj_geom);
CFloat32ProjectionData3DMemory* load_data_3d_sino(const char *filename, CFloat32ProjectionData3DMemory *proj_data);
CFloat32ProjectionData3DMemory* load_data_3d_sino_gz(const char *filename, CProjectionGeometry3D *proj_geom);
CFloat32ProjectionData3DMemory* load_data_3d_sino_gz(const char *filename, CFloat32ProjectionData3DMemory *proj_data);

// create_sino
// 2D
CFloat32ProjectionData2D* create_sino_2d(const float32 *data,CProjector2D *proj);
CFloat32ProjectionData2D* create_sino_2d(CFloat32VolumeData2D *vol,CProjector2D *proj);
CFloat32ProjectionData2D* create_sino_2d_cuda(const float32 *data,CCudaProjector2D *proj);
CFloat32ProjectionData2D* create_sino_2d_cuda(CFloat32VolumeData2D *vol,CCudaProjector2D *proj);
// 3D
CFloat32ProjectionData3DMemory* create_sino_3d_cuda(const float32 *data,CCudaProjector3D *proj);
CFloat32ProjectionData3DMemory* create_sino_3d_cuda(CFloat32VolumeData3DMemory *vol,CCudaProjector3D *proj);

