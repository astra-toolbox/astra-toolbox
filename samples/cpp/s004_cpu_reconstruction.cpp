#include <cpp/creators.hpp>
#include <astra/SirtAlgorithm.h>

#include <iostream>
#include <fstream>

using namespace std;
using namespace astra;

int main (int argc, char *argv[])
{
    int i;
    float32 *angles;
    double *phantom;
    float32 *phantom_f;
    CVolumeGeometry2D *vol_geom;
    CParallelProjectionGeometry2D *proj_geom;
    CProjector2D *proj;
    CFloat32ProjectionData2D *sino;
    CFloat32VolumeData2D *rec;
    CSirtAlgorithm *algo;
    ifstream file;

    // load phantom data
    phantom = new double[256*256];
    file.open("phantom.dat",ios::in|ios::binary);
    file.read((char*)phantom,256*256*sizeof(double));
    file.close();
    phantom_f = new float32[256*256];
    for(i=0;i<256*256;i++)
        phantom_f[i] = (float32)phantom[i];
    delete[] phantom;

    // volume geometry
    vol_geom = create_vol_geom_2d(256,256);

    // projection geometry
    angles = create_angles(0,M_PI,180);
    proj_geom = create_proj_geom_2d_parallel(1.0,384,180,angles,NULL);

    // projector
    proj = create_projector_2d_strip(proj_geom,vol_geom);

    // sinogram
    sino = create_sino_2d(phantom_f,proj);

    // reconstruction data
    rec = create_data_2d_vol(vol_geom,0.0);

    // algorithm
    algo = new CSirtAlgorithm();
    algo->initialize(proj,sino,rec);
    algo->run(20);
    delete algo;

    // write sinogram data
    save_data_2d("sinogram.dat",sino);

    // write reconstruction data
    save_data_2d("reconstruction.dat",rec);

    delete[] phantom_f;
    delete vol_geom;
    delete[] angles;
    delete proj_geom;
    delete proj;
    delete sino;
    delete rec;
    
    return 0;
}
