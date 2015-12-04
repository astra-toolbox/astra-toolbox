/*
 *
 * This is a standalone example on how to use the MPI 
 * functionality.
 *
 * Supports the BP, FP and SIRT algorithms.
 *
 *
 *
 * */


#include <stdio.h>
#include <stdlib.h>



#include <cassert>
#include <iostream>
#include <string.h>
#include <sys/time.h>

#include "cuda_runtime.h"

#include "astra/Globals.h"

#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/MPIProjector3D.h"
#include "astra/Config.h"
#include "astra/Float32ProjectionData3DMemory.h"

#include "astra/Float32VolumeData3DMemory.h"
#include "astra/Algorithm.h"
#include "astra/AstraObjectFactory.h"
#include "astra/AstraObjectManager.h"

#include "../cuda/3d/dims3d.h"

#include "astra/Logging.h"

#ifdef USE_MPI
  #include "mpi.h"
#endif


#define DUMPDATA

#ifdef DUMPDATA
	#include "testutil.h"
#endif


enum ALGO
{
	BP,
	FP,
	SIRT,
	CGLS
};



//JB, quick function to read the data files into memory
void readProjectionDataFile(std::string baseName,
                        const int angles,
                        const int cols,
                        const int slices,
                        astra::CFloat32Data3DMemory &data3d)
{
  for (int angle = 0; angle < angles; ++angle)
  {
    //Open input file
    char buff[512];
    sprintf(buff,baseName.c_str(), 1+angle);
    FILE *fin = fopen(buff, "r");
    assert(fin);

    float temp;
    for (int col = 0; col < cols; ++col)
    {   
      for (int slice = 0; slice < slices; ++slice)
      {   
        fscanf(fin, "%f", &temp);
        data3d.getData3D()[slice][angle][col] = temp;
      }   
    }   
    fclose(fin);
  }

 return;
}


void readVolumeDataFile(std::string baseName,
                        const int Xsize,
                        const int Ysize,
                        const int Zsize,
                        astra::CFloat32Data3DMemory &data3d)
{
    //Format:
    //Loops over x, then over y finally over z.

    //Open input file
    FILE *fin = fopen(baseName.c_str(), "r");
    assert(fin);

    int count =0; 
    int temp;
    for (int x = 0; x < Xsize; ++x)
    {   
        for (int y = 0; y < Ysize; ++y)
        {
            for (int z = 0; z < Zsize; ++z)
            {
                fscanf(fin, "%d", &temp);
                data3d.getData3D()[z][y][x] = temp;
                count++;
            }
        }
    }   
    fclose(fin);
    fprintf(stderr,"read: %d items \n", count);
}



template<class T, class S>
astra::CFloat32Data3DMemory* readData(astraCUDA3d::SDimensions3D &dimsFull,
				      const int pType,
	      			      S* pGeometry)
{	      
	T*  pInputDataFull = new T(pGeometry, 0.0f);

#if 1   //If only testing performance then it does not matter what kind of data we use. So use uninitialized.

	if(dynamic_cast<astra::CVolumeGeometry3D*>(pGeometry))
	{
		readVolumeDataFile("fullCubeVolume2.txt", dimsFull.iVolX, dimsFull.iVolY ,dimsFull.iVolZ,*pInputDataFull);
	}
	else
	{
		//Read in our full projection data-set
		if(pType == 0)
		    readProjectionDataFile("testDataExample7/exampleS7Data-%d.txt",dimsFull.iProjAngles, dimsFull.iProjU ,dimsFull.iProjV, *pInputDataFull);
		else if(pType == 1)
		    //readProjectionDataFile("testDataExample7Cone/exampleS7DataCone-%d.txt",dimsFull.iProjAngles, dimsFull.iProjU ,dimsFull.iProjV, *pInputDataFull);
		    readProjectionDataFile("exampleS7DataCUBECone/exampleS7DataCUBECone-%d.txt",dimsFull.iProjAngles, dimsFull.iProjU ,dimsFull.iProjV, *pInputDataFull);
		else
		    assert(0);
	}
	pInputDataFull->updateStatistics();
#endif

	return pInputDataFull;
}


int main( int argc, char *argv[] )
{
 	astra::CLogger::setOutputScreen(2, astra::LOG_DEBUG);

#if USE_MPI
	ASTRA_DEBUG("Launching the standalone client program, built WITH MPI support");
#else	
	ASTRA_DEBUG("Launching the standalone client program, built WITHOUT MPI support");	
#endif	

	int pType = 0, dimS;
	ALGO algorithmToUse    = BP;

	if(argc < 3)
	{
		printf( "usage: %s <reconstruction volume size> < Projection type: 0 = parallel, 1 = cone>\n", argv[0] );
		exit(0);
	}
	else
	{
		dimS       	= atoi(argv[1]);
		pType      	= atoi(argv[2]);
		
		if(argc == 4)
		{
			int temp  = atoi(argv[3]);
			switch(temp)
			{
				case 0:
					algorithmToUse = BP; break;
				case 1:
					algorithmToUse = FP; break;
				case 2:
					algorithmToUse = SIRT; break;
				case 3:
					algorithmToUse = CGLS; break;
				default:
					break;
			}
		}
	}

	int procId = 0, nProcs = 1;
#ifdef USE_MPI
	int isInit = 0;
	MPI_Initialized(&isInit);
	if (!isInit) MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &procId);	
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);	
#endif


	//Setup some full geometry 
	astraCUDA3d::SDimensions3D dimsFull;
	dimsFull.iVolX          = dimS;
	dimsFull.iVolY          = dimS;
	dimsFull.iVolZ          = dimS;
	dimsFull.iProjAngles    = 180;
	dimsFull.iProjU         = 192; //192
	dimsFull.iProjV         = 128; //128
	dimsFull.iRaysPerDetDim = 1;

	float detectorSpacing  = dimS / (float) dimsFull.iProjV;
	float distanceSource   = 1000.0*(dimS/(float)dimsFull.iProjV);
	float distanceDetector = 0.0;

	
	/******************************************************
	***						    ***
	***   Create Volume and Projection configurations.  ***
	***   Followed by executing the domain distribution ***
	***						    ***
	*******************************************************/

	//Compute the angles
	float *angles = new float[dimsFull.iProjAngles];
	for(unsigned int i=0; i < dimsFull.iProjAngles; i++) angles[i] = (M_PI / dimsFull.iProjAngles) * i;	
	
	//Projection configuration
	astra::Config cfg;
	cfg.initialize("MPIProjector3D");
	
	astra::XMLNode curNode = cfg.self.addChildNode("ProjectionGeometry");
	curNode.addChildNode("DetectorRowCount", dimsFull.iProjV);
	curNode.addChildNode("DetectorColCount", dimsFull.iProjU);
	curNode.addChildNode("DetectorSpacingX", detectorSpacing);
	curNode.addChildNode("DetectorSpacingY", detectorSpacing); 
	curNode.addChildNode("ProjectionAngles", angles, dimsFull.iProjAngles); 

	if(pType == 1)
	{
		curNode.addChildNode("DistanceOriginDetector", distanceDetector);
		curNode.addChildNode("DistanceOriginSource"  , distanceSource);
		curNode.addAttribute("type", "cone");
	}
	else
	{
		curNode.addAttribute("type", "parallel3d");
	}
	delete[] angles; angles = 0;
	
	//Volume configuration
	curNode = cfg.self.addChildNode("VolumeGeometry");
	curNode.addChildNode ("GridColCount",   dimsFull.iVolX);
	curNode.addChildNode ("GridRowCount",   dimsFull.iVolY);
	curNode.addChildNode ("GridSliceCount", dimsFull.iVolZ);

	//Setup domain distribution
	astra::CMPIProjector3D * mpiPrj	= new astra::CMPIProjector3D();

	bool libraryHasMPI = mpiPrj->isBuiltWithMPI();

#if USE_MPI
	if(!libraryHasMPI)
	{
		ASTRA_ERROR("Client is built with USE_MPI , but the ASTRA library is NOT built with USE_MPI");
		exit(-1);
	}
#else
	if(libraryHasMPI)
	{
		ASTRA_ERROR("Client is built without USE_MPI , but the ASTRA library is built WITH USE_MPI");
		exit(-1);
	}

#endif
	bool res = mpiPrj->initialize(cfg);

	if(!res)
	{
		ASTRA_ERROR("Failed to setup the domain distribution using the given geometries");
		exit(-1);
	}

	/*****************************************
	**					**
	**	 Setup the input data		**
	**					**
	******************************************/

	astra::CFloat32Data3DMemory*  pInputDataFull = nullptr; 
	astra::CFloat32Data3DMemory*  pInputData;

	//Read and distribute the data over the other MPI clients
	if(algorithmToUse == BP || algorithmToUse == SIRT || algorithmToUse == CGLS)
	{
	   if(procId == 0)
		pInputDataFull = readData<astra::CFloat32ProjectionData3DMemory>
					(dimsFull, pType, mpiPrj->getProjectionGlobal());
	   pInputData     = new astra::CFloat32ProjectionData3DMemory
					(mpiPrj->getProjectionLocal(), 0.0);
	}
	else
	{
   	    if(procId == 0)
		pInputDataFull = readData<astra::CFloat32VolumeData3DMemory>
					 (dimsFull, pType, mpiPrj->getVolumeGlobal());
   	    pInputData     = new astra::CFloat32VolumeData3DMemory
					  (mpiPrj->getVolumeLocal(), 0.0);
	}


	mpiPrj->distributeData(pInputData, pInputDataFull);

	delete pInputDataFull;




	//Setup the output buffer, either volume or projection results
	astra::CFloat32Data3DMemory* pOutputMem;
	if(algorithmToUse == BP || algorithmToUse == SIRT || algorithmToUse == CGLS)
		pOutputMem  = new astra::CFloat32VolumeData3DMemory(mpiPrj->getVolumeLocal(), 0.0);
	else
		pOutputMem  = new astra::CFloat32ProjectionData3DMemory(mpiPrj->getProjectionLocal(), 0.0);



	//Connect the MPIProjector to the data
	pInputData->setMPIProjector3D(mpiPrj);
 	pOutputMem->setMPIProjector3D(mpiPrj);		    

	int outputID = astra::CData3DManager::getSingleton().store(pOutputMem);
	int inputID  = astra::CData3DManager::getSingleton().store(pInputData);

	astra::Config cfgAlg;
	cfgAlg.initialize("ReconstructionAlgo");
	if(algorithmToUse == BP)
	{
		fprintf(stderr,"Using Backprojection\n");
		cfgAlg.self.addAttribute("type", "BP3D_CUDA");
		curNode = cfgAlg.self.addChildNode("ProjectionDataId", inputID);
		curNode = cfgAlg.self.addChildNode("ReconstructionDataId", outputID);
	}
	else if(algorithmToUse == SIRT)
	{
		fprintf(stderr,"Using SIRT\n");
		cfgAlg.self.addAttribute("type", "SIRT3D_CUDA");
		curNode = cfgAlg.self.addChildNode("ProjectionDataId", inputID);
		curNode = cfgAlg.self.addChildNode("ReconstructionDataId", outputID);
	}
	else if(algorithmToUse == CGLS)
	{
		fprintf(stderr,"Using CGLS\n");
		cfgAlg.self.addAttribute("type", "CGLS3D_CUDA");
		curNode = cfgAlg.self.addChildNode("ProjectionDataId", inputID);
		curNode = cfgAlg.self.addChildNode("ReconstructionDataId", outputID);
	}
	else if(algorithmToUse == FP)
	{
		fprintf(stderr,"Using Forwardprojection\n");
		cfgAlg.self.addAttribute("type", "FP3D_CUDA");
		curNode = cfgAlg.self.addChildNode("VolumeDataId", inputID);
		curNode = cfgAlg.self.addChildNode("ProjectionDataId", outputID);
	}
	else
	{
		ASTRA_ERROR("Unknown/unimplemented reconstruction algorithm.");
		exit(-1);
	}
	
	CAlgorithm *algo = astra::CAlgorithmFactory::getSingleton().create(cfgAlg);


	algo->run(150);


	//Merge all data back on the root process for output
	astra::CFloat32Data3DMemory* pOutputDataFull = nullptr;

	if(algorithmToUse == FP)
	   pOutputDataFull = mpiPrj->combineData<astra::CFloat32ProjectionData3DMemory>(pOutputMem,  mpiPrj->getProjectionGlobal());
	else
	   pOutputDataFull = mpiPrj->combineData<astra::CFloat32VolumeData3DMemory>(pOutputMem,  mpiPrj->getVolumeGlobal());



	//Clean up some memory
	astra::CData3DManager::getSingleton().remove(outputID);
	astra::CData3DManager::getSingleton().remove(inputID);


#ifdef DUMPDATA
	if(procId == 0)
	{
	    const int nZz = pOutputDataFull->getHeight();
	    const int nYy = pOutputDataFull->getWidth();
	    const int nXx = pOutputDataFull->getDepth();

	    const int nZ = nXx; const int nY = nZz; const int nX = nYy; //Z slices
	    //const int nZ = nZz; const int nY = nYy; const int nX = nXx; //Y slices
	    //const int nZ = nYy; const int nY = nXx; const int nX = nZz; //X slices

	    for ( int z = 0; z < nZ; ++z)
	    {   
	      char buff[512];

	      float* img = new float[nX*nY];
	      memset(img, 0, nX*nY*sizeof(float));

	      for (int y = 0; y < nY; ++y)
	      {   
		for (int x = 0; x < nX; ++x)
		{
		  //img[x*nY + y] = pOutputMem->getData3D()[y][x][z]; //X-slices
		  img[x*nY + y] =  pOutputDataFull->getData3D()[z][y][x]; //Z-slices
		  //img[x*nY + y] = pOutputMem->getData3D()[x][z][y]; //Y-slices
		}
	      }   
	      sprintf(buff,"results/result-%d-%d.png", 1+z, procId);
	      saveImage(buff, nX, nY,img);
	      delete[] img;
	    }   
	}
#endif

#if USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif	


	delete mpiPrj;
	delete algo;
	if(pOutputDataFull) delete pOutputDataFull;
	
	return 0;
}



