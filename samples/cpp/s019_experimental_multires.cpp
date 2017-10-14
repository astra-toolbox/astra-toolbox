#include <cpp/creators.hpp>
#include <astra/CompositeGeometryManager.h>

using namespace std;
using namespace astra;

int main (int argc, char *argv[])
{
    int j,i,k;
    float32 *angles1, *angles2;
    float32 *cube1, *cube2;
    CVolumeGeometry3D *vol_geom1, *vol_geom2;
    CParallelProjectionGeometry3D *proj_geom1, *proj_geom2;
    CFloat32ProjectionData3DMemory *proj1, *proj2;
    CFloat32VolumeData3DMemory *vol1, *vol2;
    CCompositeGeometryManager *cgeomgr;
    CCudaProjector3D *proj;
    vector<CFloat32VolumeData3DMemory*> vols;
    vector<CFloat32ProjectionData3DMemory*> projs;

    // create phantom data (hollow cube)
    cube1 = new float32[16*32*32]();
    for(j=4;j<16;j++)
        for(i=4;i<28;i++)
            for(k=4;k<28;k++)
                cube1[j + i*16 + k*16*32] = 1.0;

    cube2 = new float32[64*128*128]();
    for(j=0;j<64;j++)
        for(i=16;i<112;i++)
            for(k=16;k<112;k++)
                cube2[j + i*64 + k*64*128] = 1.0;
    for(j=4;j<28;j++)
        for(i=33;i<97;i++)
            for(k=33;k<97;k++)
                cube2[j + i*64 + k*64*128] = 0.0;

    // volume geometry
    vol_geom1 = create_vol_geom_3d(32,16,32,-64,0,-64,64,-64,64);
    vol_geom2 = create_vol_geom_3d(128,64,128,0,64,-64,64,-64,64);

    // projection geometry
    angles1 = create_angles(0,M_PI_2,90);
    angles2 = create_angles(M_PI_2,M_PI,90);
    proj_geom1 = create_proj_geom_3d_parallel(1.0,1.0,128,192,90,angles1);
    proj_geom2 = create_proj_geom_3d_parallel(1.0,1.0,128,192,90,angles2);

    // data
    vol1 = create_data_3d_vol(vol_geom1,cube1);
    vol2 = create_data_3d_vol(vol_geom2,cube2);

    proj1 = create_data_3d_sino(proj_geom1,0.0);
    proj2 = create_data_3d_sino(proj_geom2,0.0);

    // store in vector
    vols.push_back(vol1);
    vols.push_back(vol2);

    projs.push_back(proj1);
    projs.push_back(proj2);

    // projector
    proj = create_projector_3d_cuda(proj_geom1,vol_geom1);

    // composite manager
    cgeomgr = new CCompositeGeometryManager();
    cgeomgr->doFP(proj,vols,projs);
    delete cgeomgr;

    // write projection data
    save_data_3d("sinogram_comp_1.dat",proj1);
    save_data_3d("sinogram_comp_2.dat",proj2);

    delete[] cube1;
    delete[] cube2;
    delete vol_geom1;
    delete vol_geom2;
    delete[] angles1;
    delete[] angles2;
    delete proj_geom1;
    delete proj_geom2;
    delete proj;
    delete vol1;
    delete vol2;
    delete proj1;
    delete proj2;
    
    return 0;
}
