#include <cpp/creators.hpp>
#include <astra/CudaSirtAlgorithm.h>

#include <iostream>
#include <fstream>

using namespace std;
using namespace astra;

#define SQR(x) ((x)*(x))

int main (int argc, char *argv[])
{
    int j,i;
    float32 *angles;
    double *phantom;
    float32 *phantom_f, *mask_f;
    CVolumeGeometry2D *vol_geom;
    CParallelProjectionGeometry2D *proj_geom;
    CCudaProjector2D *proj;
    CFloat32ProjectionData2D *sino;
    CFloat32VolumeData2D *rec, *rec_msk;
    CCudaSirtAlgorithm *algo;
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

    // create mask data
    mask_f = new float32[256*256]();
    for(j=0;j<256;j++)
        for(i=0;i<256;i++)
            mask_f[j + i*256] = (SQR(-127.5 + j) + SQR(-127.5 + i) < SQR(127.5));

    // volume geometry
    vol_geom = create_vol_geom_2d(256,256);

    // projection geometry
    angles = create_angles(0,M_PI,50);
    proj_geom = create_proj_geom_2d_parallel(1.0,384,50,angles,NULL);

    // projector
    proj = create_projector_2d_cuda(proj_geom,vol_geom);

    // sinogram
    sino = create_sino_2d_cuda(phantom_f,proj);

    // reconstruction data
    rec = create_data_2d_vol(vol_geom,0.0);
    rec_msk = create_data_2d_vol(vol_geom,mask_f);

    // algorithm
    algo = new CCudaSirtAlgorithm();
    algo->initialize(proj,sino,rec);
    algo->setReconstructionMask(rec_msk);
    algo->run(150);
    delete algo;

    // write mask data
    save_data_2d("mask.dat",rec_msk);

    // write sinogram data
    save_data_2d("sinogram_mask.dat",sino);

    // write reconstruction data
    save_data_2d("reconstruction_mask.dat",rec);

    delete[] phantom_f;
    delete[] mask_f;
    delete vol_geom;
    delete[] angles;
    delete proj_geom;
    delete proj;
    delete sino;
    delete rec;
    delete rec_msk;
    
    return 0;
}
