/*
-----------------------------------------------------------------------
Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
$Id$
*/


#include "astra/MPIProjector3D.h"

#include "astra/VolumeGeometry3D.h"
#include "astra/ProjectionGeometry3D.h"

#include "cuda.h"

#if USE_MPI
  #include <mpi.h>
#endif

namespace astra
{

// type of the projector, needed to register with CProjectorFactory
std::string CMPIProjector3D::type = "mpi3d";


//----------------------------------------------------------------------------------------
// Default constructor
CMPIProjector3D::CMPIProjector3D()
{
	_clear();

#if USE_MPI
	int isInit = 0;
	MPI_Initialized(&isInit);
	if(!isInit)
	{  
		int argc = 0;
		char **argv;
		if(MPI_Init(&argc, &argv) != MPI_SUCCESS)
		{
		    ASTRA_ERROR("MPI Initialization failed");
		    ::exit(-1);
		}
	}

	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procId); 
#else
	nProcs = 1;
	procId = 0;
#endif	

	this->processExtends.resize(nProcs);
}

//----------------------------------------------------------------------------------------
// Destructor
CMPIProjector3D::~CMPIProjector3D()
{
	if (m_bIsInitialized) clear();
}

//----------------------------------------------------------------------------------------
// Clear for constructors
void CMPIProjector3D::_clear()
{
	m_pProjectionGeometry 	    = NULL;
	m_pVolumeGeometry 	    = NULL;
	m_bIsInitialized 	    = false;

	m_pProjectionGeometryGlobal = NULL;
	m_pVolumeGeometryGlobal	    = NULL;
}

//----------------------------------------------------------------------------------------
// Clear
void CMPIProjector3D::clear()
{
	ASTRA_DELETE(m_pProjectionGeometry);
	ASTRA_DELETE(m_pVolumeGeometry);
	ASTRA_DELETE(m_pProjectionGeometryGlobal);
	ASTRA_DELETE(m_pVolumeGeometryGlobal);
	m_bIsInitialized = false;
}

//----------------------------------------------------------------------------------------
// Check
bool CMPIProjector3D::_check()
{
	// projection geometry
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry, "MPIProjector3D", "ProjectionGeometry3D not initialized.");
	ASTRA_CONFIG_CHECK(m_pProjectionGeometry->isInitialized(), "MPIProjector3D", "ProjectionGeometry3D not initialized.");

	// volume geometry
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry, "MPIProjector3D", "VolumeGeometry3D not initialized.");
	ASTRA_CONFIG_CHECK(m_pVolumeGeometry->isInitialized(), "MPIProjector3D", "VolumeGeometry3D not initialized.");

	
	// projection geometry
	ASTRA_CONFIG_CHECK(m_pProjectionGeometryGlobal, "MPIProjector3D", "ProjectionGeometry3DGlobal not initialized.");
	ASTRA_CONFIG_CHECK(m_pProjectionGeometryGlobal->isInitialized(), "MPIProjector3D", "ProjectionGeometry3DGlobal not initialized.");

	// volume geometry
	ASTRA_CONFIG_CHECK(m_pVolumeGeometryGlobal, "MPIProjector3D", "VolumeGeometry3DGlobal not initialized.");
	ASTRA_CONFIG_CHECK(m_pVolumeGeometryGlobal->isInitialized(), "MPIProjector3D", "VolumeGeometry3DGlobal not initialized.");


	return true;
}

bool CMPIProjector3D::isBuiltWithMPI() 
{
#if USE_MPI
	return true;
#else
	return false;
#endif		
}

//---------------------------------------------------------------------------------------
// Initialize, use a Config object
bool CMPIProjector3D::initialize(const Config& _cfg, 
		const int nGhostCellsVolume = 0,
		const int nGhostCellsProjection = 0)
{
	assert(_cfg.self);
	ConfigStackCheck<CProjector3D> CC("MPIProjector3D", this, _cfg);

	// if already initialized, clear first
	if (m_bIsInitialized) {
		clear();
	}

	// initialization of parent class
	if (!CProjector3D::initialize(_cfg)) {
		return false;
	}

	XMLNode node = _cfg.self.getSingleNode("GPUList");

	//Check if there is a list with GPU-id entries, or that we assign 
	//the processes to the available GPUs in a round robin fasion.
	int nDev = 1;
	assert(cudaGetDeviceCount(&nDev) == cudaSuccess);
	if(node)
	{
	  std::vector<float32> list = node.getContentNumericalArray();
	  CC.markNodeParsed("GPUList");

	  int listIdx = procId % list.size();
	  int gpuIdx  = (int)(list[listIdx]);

	  if(gpuIdx < 0 || gpuIdx >= nDev)
	  {
	    ASTRA_WARN("Invalid GPU device (%d) requested in supplied list. Using default device.", gpuIdx); //Default device, do not call cudaSetDevice
	  }
	  else
	  {
	    assert(cudaSetDevice(gpuIdx) == cudaSuccess);
	  }
	}
	else
	{
	  assert(cudaSetDevice(procId % nDev) == cudaSuccess);
	}


	m_pProjectionGeometryGlobal = m_pProjectionGeometry;
	m_pVolumeGeometryGlobal	    = m_pVolumeGeometry;


	m_bIsInitialized = _check();
	if (m_bIsInitialized)
	{
	  //Convert the global detector geometry into local geometries
	  this->determineVolumeAndProjectionExtends(m_pProjectionGeometry, m_pVolumeGeometry, nGhostCellsVolume, nGhostCellsProjection);

	  this->createOverlapRegions(overLapRegionsDetector, overLapPerProcDetector, 1);		//Detector overlaps
	  this->createOverlapRegions(overLapRegionsVolSend, overLapPerProcVolSend, 2);  //Volume overlap, regions we send
	  this->createOverlapRegions(overLapRegionsVolRecv, overLapPerProcVolRecv, 3);  //Volume overlap, regions we receive



	  if(nGhostCellsProjection > 0) //Only needed when we actually have extra overlapping areas
	  {  
	     this->createOverlapRegions(overLapRegionsPrjSend, overLapPerProcPrjSend, 4);  //Projection overlap, regions we receive
	     this->createOverlapRegions(overLapRegionsPrjRecv, overLapPerProcPrjRecv, 5);  //Projection overlap, regions we receive
	  }
	  
	  //Full projection (ghost) area. This area can also overlap for cone projections
	  this->createOverlapRegions(overLapRegionsPrjSendFull, overLapPerProcPrjSendFull, 6);  //Projection overlap, regions we receive
	  this->createOverlapRegions(overLapRegionsPrjRecvFull, overLapPerProcPrjRecvFull, 7);  //Projection overlap, regions we receive

	
	  if (dynamic_cast<CParallelProjectionGeometry3D*> (m_pProjectionGeometry)) {
	  	m_pProjectionGeometry = dynamic_cast<CParallelProjectionGeometry3D*>(m_pProjectionGeometry->clone());
	  } else if(dynamic_cast<CParallelVecProjectionGeometry3D*> (m_pProjectionGeometry)) {
	  	m_pProjectionGeometry = dynamic_cast<CParallelVecProjectionGeometry3D*>(m_pProjectionGeometry->clone());
	  } else if(dynamic_cast<CConeProjectionGeometry3D*> (m_pProjectionGeometry)) {
	  	m_pProjectionGeometry = dynamic_cast<CConeProjectionGeometry3D*>(m_pProjectionGeometry->clone());
	  } else if(dynamic_cast<CConeVecProjectionGeometry3D*> (m_pProjectionGeometry)) {
	  	m_pProjectionGeometry = dynamic_cast<CConeVecProjectionGeometry3D*>(m_pProjectionGeometry->clone());
	  }
	  else
	  {
		ASTRA_ERROR("Unknown geometry in MPIProjector3D::initialize!\n");
	  }




/*

  For a cone projection geometry, the setDetectorRowCount below centers the
detector around Z=0. We shift the cone_vec geometry to get the same positioning.
It is then shifted to the correct position in 'transformMPIProjectionGeometry'
in astra3d.cu .

We do something similar for the parallel3d vec geometry.

*/

	  float dV = -0.5*(((int)processExtends[procId].dims.iProjV) - ((int)m_pProjectionGeometry->getDetectorRowCount()));

	  if(dynamic_cast<CConeVecProjectionGeometry3D*> (m_pProjectionGeometry))
	  {
		CConeVecProjectionGeometry3D* tempPrj = dynamic_cast<CConeVecProjectionGeometry3D*>(m_pProjectionGeometry);

		//Cast const away such that we can directly modify the angles
		SConeProjection* pConeProjs = const_cast<SConeProjection*>(tempPrj->getProjectionVectors());

		int nth = tempPrj->getProjectionCount();
		for (int i = 0; i < nth; ++i)
		{
			pConeProjs[i].fDetSX  += dV*pConeProjs[i].fDetVX;
			pConeProjs[i].fDetSY  += dV*pConeProjs[i].fDetVY;
			pConeProjs[i].fDetSZ  += dV*pConeProjs[i].fDetVZ;
		}
	  }
	  
	  if(dynamic_cast<CParallelVecProjectionGeometry3D*> (m_pProjectionGeometry))
	  {
		CParallelVecProjectionGeometry3D* tempPrj = dynamic_cast<CParallelVecProjectionGeometry3D*>(m_pProjectionGeometry);

		//Cast const away such that we can directly modify the angles
		SPar3DProjection* pPar3DProjs = const_cast<SPar3DProjection*>(tempPrj->getProjectionVectors());

		int nth = tempPrj->getProjectionCount();
		for (int i = 0; i < nth; ++i)
		{
			pPar3DProjs[i].fDetSX  += dV*pPar3DProjs[i].fDetVX;
			pPar3DProjs[i].fDetSY  += dV*pPar3DProjs[i].fDetVY;
			pPar3DProjs[i].fDetSZ  += dV*pPar3DProjs[i].fDetVZ;
		}
	  }


	  //Initialize using the new domain decomposition	  
          m_pProjectionGeometry->setDetectorRowCount(processExtends[procId].dims.iProjV);

	  m_pVolumeGeometry = m_pVolumeGeometryGlobal->clone();
	  m_pVolumeGeometry->setGridSliceCount(processExtends[procId].dims.iVolZ);
	}


	m_bIsInitialized = _check();


	return m_bIsInitialized;
}

/*
bool CProjector3D::initialize(astra::CProjectionGeometry3D *, astra::CVolumeGeometry3D *)
{
	ASTRA_ASSERT(false);

	return false;
}
*/

std::string CMPIProjector3D::description() const
{
	return "";
}

//TODO this will likely fail if the nGhostcells is larger than the number of slices
int2 CMPIProjector3D::divideOverProcessGhostCellCount(
		const int volumeOffset,
		const int volumeHeight,
		const int fullVolumeSize,
		const int nGhostcells) 
{
    int2 result = make_int2(nGhostcells, nGhostcells);

    //Test if ghost cells go past the beginning slice
    if(volumeOffset - result.x < 0)
	    result.x += (volumeOffset - result.x); //Add negative number

    //Test if the ghost cells go past the end slice
    int temp = volumeOffset + volumeHeight + result.y;
    if(temp > fullVolumeSize)
	    result.y -= temp - fullVolumeSize; //Remove overshoot

    return result;
}

int CMPIProjector3D::divideOverProcess(const int count, int procIdx, int nWorkers)
{
    nWorkers        = (nWorkers >= 0) ? nWorkers : nProcs; //If not specified use configured number of ranks
    procIdx         = (procIdx >= 0) ? procIdx :  procId;         //For ourself or for another rank
    const int temp  = count /  nWorkers;                          //Equal distribution
    const int rest  = count - (nWorkers*temp);                    //Check if we divide integers evenly
          int newc  = temp  + (procIdx < rest);                   //The even distribution + anything remaining

    return newc;
}


int CMPIProjector3D::divideComputeStartOffset(const int count, int procIdx, int nWorkers)
{
    nWorkers        = (nWorkers >= 0) ? nWorkers : nProcs;       //If not specified use configured number of ranks
    procIdx         = (procIdx >= 0)  ? procIdx  : procId;             //For ourself or for another rank
    const int temp  = count /  nWorkers;                                //Equal distribution
    const int rest  = count - (nWorkers*temp);                          //Check if we divide integers evenly
    int offset      = temp*procIdx + (procIdx < rest ? procIdx : rest); //Computes the offset taking into account any remaining
                                                                        //which is handled by the lowest processor ranks
    return offset;
}



void CMPIProjector3D::determineVolumeAndProjectionExtends(CProjectionGeometry3D* pProjection,
                                                          CVolumeGeometry3D*     pVolume,
							  const int 		 nGhostCells,
							  const int 		 nGhostCellsPrj)
{
#define nullptr NULL
    assert(pProjection != nullptr);
    assert(pVolume     != nullptr);

if(procId == 0)
    ASTRA_DEBUG("Full Volume size [ %f %f %f ] [ %f %f %f ]",
            pVolume->getWindowMinX(),
            pVolume->getWindowMinY(),
            pVolume->getWindowMinZ(),
            pVolume->getWindowMaxX(),
            pVolume->getWindowMaxY(),
            pVolume->getWindowMaxZ());

    for(int idx=0; idx < nProcs; idx++)
    {   
        //Get the number of items and dimensions
        int newRowCount     = divideOverProcess       (pVolume->getGridSliceCount(), idx);
        int startRow        = divideComputeStartOffset(pVolume->getGridSliceCount(), idx);
        int endRow          = startRow+newRowCount;


        float minIdx = pVolume->getGridSliceCount()+1;
        float maxIdx = -1; 

        //Compute the extends of the projection (NOTE the iGhostCellCount to reduce volume size)
        //TODO super-sampling optimization, change delta to 0.5 to reduce overlap when not using super-sampling
	//This elimintes the overlap when using parallel projection
        float delta = 0.0;
        float fX[]  = { delta+pVolume->getWindowMinX(),  -delta+pVolume->getWindowMaxX() };
        float fY[]  = { delta+pVolume->getWindowMinY(),  -delta+pVolume->getWindowMaxY() };
        float fZ[]  = { delta+startRow-(pVolume->getGridSliceCount()/2.0f),
                       -delta+endRow  -(pVolume->getGridSliceCount()/2.0f) };

        //For all 8 corners of our sub-volume test the extends
        for (int a = 0; a < pProjection->getProjectionCount(); ++a)
          for (int i = 0; i < 2; ++i)
             for (int j = 0; j < 2; ++j)
                 for (int k = 0; k < 2; ++k) {
                     double fU, fV; 
                     pProjection->projectPoint(fX[i], fY[j], fZ[k], a, fU, fV);
                     minIdx = std::min<float>(minIdx, fV);
                     maxIdx = std::max<float>(maxIdx, fV);
                 }
if(procId == 0)
        ASTRA_DEBUG("Proc: %d  Projection Min max:\t %f %f\t %d %d", idx, minIdx, maxIdx, (int)(floor(minIdx)), (int)(ceil(maxIdx)));

        //Make sure the rows fall within the boundaries of the projection
        int minProjectionRow = std::max(0,                                    (int)(floor(minIdx)));
        int maxProjectionRow = std::min(pProjection->getDetectorRowCount()-1, (int)(ceil (maxIdx)));

	//Make sure we cover the whole projector, by setting the begin and end rows
	if(idx == 0       ) minProjectionRow = 0;
	if(idx == nProcs-1) maxProjectionRow = pProjection->getDetectorRowCount()-1;


        //Volume settings
        processExtends[idx].dims.iVolX          = pVolume->getGridColCount();
        processExtends[idx].dims.iVolY          = pVolume->getGridRowCount();
        processExtends[idx].dims.iVolZ          = newRowCount;
        processExtends[idx].iVolumeOffset       = startRow;
        processExtends[idx].iVolumeFullSize     = pVolume->getGridSliceCount();
        //Projection/detector settings
        processExtends[idx].dims.iProjAngles    = pProjection->getProjectionCount();
        processExtends[idx].dims.iProjU         = pProjection->getDetectorColCount();
        processExtends[idx].dims.iProjV         = maxProjectionRow-minProjectionRow+1; //+1 to include last row in count
        processExtends[idx].iProjOffset         = minProjectionRow;
        processExtends[idx].iProjectionFullSize = pProjection->getDetectorRowCount();
        //General settings
        processExtends[idx].fDetectorSpacingX   = pProjection->getDetectorSpacingX();
        processExtends[idx].fDetectorSpacingY   = pProjection->getDetectorSpacingY();

        processExtends[idx].dims.iRaysPerVoxelDim  = 1;
        processExtends[idx].dims.iRaysPerDetDim    = 1;



	//This is a reminder to make sure that anything new does not require something
	//we don't know about
        if(pProjection->isOfType("cone") || pProjection->isOfType("cone_vec") || pProjection->isOfType("parallel3d") || pProjection->isOfType("parallel3d_vec"))
        {
        }
        else
	{
  	  ASTRA_ERROR("Unknown projection type");
 	  assert(0); //Implement other projection methods
	}

        //Compute the modification that have to be made to the location of the source and detector

        //fDetZ should be set based on the full detector, compute the original location that
        //we then assign to the detector Z index
        const float fullProjectDetZ     = -1*(pProjection->getDetectorRowCount() / 2.0);
        const float thisProjectDetZ     = fullProjectDetZ + (0.5*processExtends[idx].dims.iProjV) + processExtends[idx].iProjOffset;

        //Compute the offset for the volume block, that has to be added to projection settings
        const float thisVolumeBlock     = -1*(- (processExtends[idx].iVolumeFullSize / 2.0)
                                              + (processExtends[idx].dims.iVolZ      / 2.0)
                                              +  processExtends[idx].iVolumeOffset);
        processExtends[idx].fVolumeDetModificationZ = thisVolumeBlock;
        processExtends[idx].fDetectorStartZ         = thisProjectDetZ;

	//Compute/add ghostcells to the volume. This is done indepedently from the above
	//computations as the ghostCells will be ignored during the FP and BP calls
	//and consequently do not influence the projection geometry.
	//We do update the start offsets as the volume will be larger.
	//This is done both for reconstruction volume as detector volume data.

        //Compute the number of slices on the top and bottom of the volume that 
        //will be added as ghost cells. These should not affect the projection settings
	//and are therefore only added after we computed the projection modifications
        processExtends[idx].iGhostCellCount = divideOverProcessGhostCellCount (
				processExtends[idx].iVolumeOffset,
				processExtends[idx].dims.iVolZ,
				processExtends[idx].iVolumeFullSize,
				nGhostCells);

	//Add the ghostcell slices
	processExtends[idx].dims.iVolZ += processExtends[idx].iGhostCellCount.x;
	processExtends[idx].dims.iVolZ += processExtends[idx].iGhostCellCount.y;

	//Modify the start offset
	processExtends[idx].iVolumeOffset -= processExtends[idx].iGhostCellCount.x;


	//Now the same ghostcell computations for the projection data 
        processExtends[idx].iGhostCellCountPrj = divideOverProcessGhostCellCount (
				processExtends[idx].iProjOffset,
				processExtends[idx].dims.iProjV,
				processExtends[idx].iProjectionFullSize,
				nGhostCellsPrj);
	processExtends[idx].iGhostCellCountPrjDef =  processExtends[idx].iGhostCellCountPrj;
	
	processExtends[idx].dims.iProjV   += processExtends[idx].iGhostCellCountPrj.x;
	processExtends[idx].dims.iProjV   += processExtends[idx].iGhostCellCountPrj.y;
	processExtends[idx].iProjOffset   -= processExtends[idx].iGhostCellCountPrj.x;


if(procId == 0){
         ASTRA_DEBUG("Process %d Volume goes from Slice: %d  to: %d ( %d ) || Projection: from %d to %d ( %d )",
                idx, startRow, endRow, newRowCount,
                minProjectionRow, 
		maxProjectionRow+1, maxProjectionRow-minProjectionRow+1); }
    }// for _nWorkerProcs


    //Determine the area that we are responsible for
    //Method 1: Divide based on current projection settings.
    //Loop over the domains and find the middle point between the (overlapping) areas of two consecutive processes
    //divide the possible overlap equally between the domains. This does not guarantee an equal sized division but 
    //does ensure that the areas fall within the assigned areas. This processes happens for both the volume and 
    //detector areas.
    
    processExtends[0].iResponsibleVolStart  = 0;
    processExtends[0].iResponsibleProjStart = 0;
	 
    processExtends[0].iResponsibleVolEnd = processExtends[0].dims.iVolZ -
		 (processExtends[0].iGhostCellCount.x+processExtends[0].iGhostCellCount.y);

    processExtends[0].iResponsibleProjEnd = processExtends[0].dims.iProjV -
		 (processExtends[0].iGhostCellCountPrj.x+processExtends[0].iGhostCellCountPrj.y);



    for(int idx=1; idx < nProcs; idx++)
    {
        //Compute area of volume that we are responsible for
	int startOfCur = processExtends[idx].iVolumeOffset;
	int endOfPrev  = processExtends[idx-1].iVolumeOffset + processExtends[idx-1].dims.iVolZ;
	int diff       = endOfPrev-startOfCur;

	processExtends[idx].iResponsibleVolStart = 
		processExtends[idx].iGhostCellCount.x + processExtends[idx].iVolumeOffset;
	processExtends[idx].iResponsibleVolEnd = 
		processExtends[idx].iResponsibleVolStart + processExtends[idx].dims.iVolZ -
		(processExtends[idx].iGhostCellCount.x+processExtends[idx].iGhostCellCount.y);

	//Compute detector area we are responsible for
	processExtends[idx].iResponsibleProjStart = 
		processExtends[idx].iGhostCellCountPrj.x + processExtends[idx].iProjOffset;
	processExtends[idx].iResponsibleProjEnd = 
		processExtends[idx].iResponsibleProjStart + processExtends[idx].dims.iProjV -
		(processExtends[idx].iGhostCellCountPrj.x+processExtends[idx].iGhostCellCountPrj.y);

	//Average our start with that of the previous
	//Do the same for the detector area
	startOfCur = processExtends[idx].iResponsibleProjStart;
	endOfPrev  = processExtends[idx-1].iResponsibleProjEnd;
	diff 	   = endOfPrev-startOfCur;
	endOfPrev 				 -= (diff / 2);
	processExtends[idx-1].iResponsibleProjEnd = endOfPrev;
	processExtends[idx].iResponsibleProjStart = endOfPrev;
    }


    processExtends[nProcs-1].iResponsibleVolEnd = 
       processExtends[nProcs-1].iResponsibleVolStart + processExtends[nProcs-1].dims.iVolZ -
      (processExtends[nProcs-1].iGhostCellCount.x+processExtends[nProcs-1].iGhostCellCount.y);

    processExtends[nProcs-1].iResponsibleProjEnd =  processExtends[nProcs-1].iProjectionFullSize;


#if 1
    //We can decrease the ghost-regions by merging them with the overlap  
    //areas of the detector. We compute/exchange those slices anyway, so 
    //by combining them with ghostcells we can make a minor optimization.
    if(nGhostCellsPrj > 0)
    {
	    for(int idx = 0; idx < nProcs; idx++)
	    {
		    int2 diffSum = {0,0};
		    //Ghost cells at the beginning
		    int diff = processExtends[idx].iResponsibleProjStart-processExtends[idx].iProjOffset;
		    if(diff > nGhostCellsPrj) diffSum.x = (diff - nGhostCellsPrj);

		    //Same for trailing ghostcells
		    diff = (processExtends[idx].iProjOffset + processExtends[idx].dims.iProjV) -
			    processExtends[idx].iResponsibleProjEnd;
		    if(diff > nGhostCellsPrj) diffSum.y = (diff - nGhostCellsPrj);

		    processExtends[idx].iProjOffset 	     += diffSum.x;
		    processExtends[idx].dims.iProjV 	     -= (diffSum.x + diffSum.y);
		    processExtends[idx].iGhostCellCountPrj.x -= diffSum.x;
		    processExtends[idx].iGhostCellCountPrj.y -= diffSum.y;
	    }
    } 
#endif



    //Sanity checks  
  if(procId == 0)
    for(int idx=0; idx < nProcs; idx++)
    {
	ASTRA_DEBUG("Proc: %d  volume responsible: %d  -> %d (full: %d ) | Project: %d  -> %d  ( %d , full: %d )",
			    idx,
			    processExtends[idx].iResponsibleVolStart,  processExtends[idx].iResponsibleVolEnd,		
			    processExtends[idx].dims.iVolZ,
			    processExtends[idx].iResponsibleProjStart, processExtends[idx].iResponsibleProjEnd,
			    processExtends[idx].iResponsibleProjEnd-processExtends[idx].iResponsibleProjStart,
			    processExtends[idx].dims.iProjV);
	ASTRA_DEBUG("Proc: %d  volume full: %d  -> %d (full: %d ) | Project: %d  -> %d  ( full: %d )",
			    idx,
			    processExtends[idx].iVolumeOffset,  processExtends[idx].iVolumeOffset+processExtends[idx].dims.iVolZ,		
			    processExtends[idx].dims.iVolZ,
			    processExtends[idx].iProjOffset, processExtends[idx].iProjOffset+processExtends[idx].dims.iProjV,
			    processExtends[idx].dims.iProjV);

	assert(processExtends[idx].iProjOffset   				    <= processExtends[idx].iResponsibleProjStart);
	assert(processExtends[idx].iVolumeOffset 				    <= processExtends[idx].iResponsibleVolStart);
	assert(processExtends[idx].iProjOffset   +  processExtends[idx].dims.iProjV >= processExtends[idx].iResponsibleProjEnd);
	assert(processExtends[idx].iVolumeOffset +  processExtends[idx].dims.iVolZ  >= processExtends[idx].iResponsibleVolEnd); 
    }	    

}



/*
 * This function computes the possible overlap between the distributed domains.
 * See the MPIProjector3D.h file for a parameter description.
 */
void CMPIProjector3D::createOverlapRegions(
    std::vector<uint4> &overLapRegions,
    std::vector<uint4> &overLapPerProc,
    const int method)
{

    assert(method >= 1 && method <= 7);
    //method: 1) Projection data, 2) Which Volume areas will we send, 3) Which Volume areas will we receive
    overLapRegions.reserve(nProcs); overLapRegions.clear();
    overLapPerProc.reserve(nProcs); overLapPerProc.clear();

    auto testOverlap = [&](const int startA, const int endA, const int startB, const int endB)
    {
        return std::max(0 ,std::min(endA, endB) - std::max(startA, startB));
    };

    int4 myStartEndRow     = {0,0,0,0};
    int4 remoteStartEndRow = {0,0,0,0};

    for(int i=0; i < nProcs; i++)
    {
        if(i == procId) continue;
        
        if(method == 1)
        { 
	    //Projection overlap, test local projection against remote projection
	    remoteStartEndRow = getProjectionExtends(i,      true); 
	    myStartEndRow     = getProjectionExtends(procId, true);
        }
        else if(method == 2)
        { 
	    //Volume overlap, data we SEND. 
	    //Ttest local responsible against full remote volume
	    myStartEndRow     = getResponsibleVolumeExtends(procId);
	    remoteStartEndRow = getVolumeExtends(i, false);
        }
        else if(method == 3)
        { 
	    //Volume overlap, data we receive. Opposite of 2
	    //Test full local against remote responsible area
	    remoteStartEndRow =  getResponsibleVolumeExtends(i); 
	    myStartEndRow     =  getVolumeExtends(procId, false);
        }
	else if(method == 4)
	{ 
	    //Projection ghost overlap, data we send.
	    //Test local extends against full remote projection area
	    remoteStartEndRow = getProjectionExtends(i, false);

	    //Projection Ghost overlap, send responsible area, minus
	    //anything we already send after performing the FP sum operation 
	    myStartEndRow    = getResponsibleProjectionExtends(procId);
	    myStartEndRow.x += processExtends[procId].iGhostCellCountPrjDef.x - 
		    	       processExtends[procId].iGhostCellCountPrj.x;
	    myStartEndRow.y -= processExtends[procId].iGhostCellCountPrjDef.y - 
		    	       processExtends[procId].iGhostCellCountPrj.y;
	}
	else if(method == 5)
	{
	   //Projection ghost overlap, data we receive. Opposite of 4
	   //Test remote extends against full local projection area
	   //Determine local reponsible proj area minus parts already send
	   remoteStartEndRow    = getResponsibleProjectionExtends(i); 
      	   remoteStartEndRow.x += processExtends[i].iGhostCellCountPrjDef.x - 
		   		  processExtends[i].iGhostCellCountPrj.x;
     	   remoteStartEndRow.y -= processExtends[i].iGhostCellCountPrjDef.y - 
		   		  processExtends[i].iGhostCellCountPrj.y;
        
       	    //Projection Ghost overlap, Receive , determine full area
	    myStartEndRow = getProjectionExtends(procId, false);
	}
	else if (method == 6)
	{   
	    //Projection ghost overlap, data we send 
	    //Test our responsible area against remote extend projection area
	    myStartEndRow     = getResponsibleProjectionExtends(procId);
	    remoteStartEndRow = getProjectionExtends(i, false);
	}
	else if(method == 7)
	{
	   //Projection ghost overlap, data we receive. Opposite of 6
	   //Test our extend against responsible remote projection area
 	   myStartEndRow     = getProjectionExtends(procId, false);
   	   remoteStartEndRow = getResponsibleProjectionExtends(i);
	}

	int startRemote = std::max(myStartEndRow.x, remoteStartEndRow.x);
	int endRemote   = std::min(myStartEndRow.y, remoteStartEndRow.y);
	int myOffset    = myStartEndRow.z;

                
        int overlap     = std::max(-1, endRemote - startRemote);

        if(overlap >= 0)
        {
            uint sZ =     startRemote - myOffset; //Start slice with myOffset removed its the local index
            uint eZ =     endRemote   - myOffset; //End Slice.

            //Test if this region overlaps with any previous found regions
            int regionsOverlap = -1;
            for(uint j=0; j < overLapRegions.size(); j++)
            {
                int rOverlap = testOverlap(sZ, eZ, overLapRegions[j].x, overLapRegions[j].y);
                if(rOverlap)
                {
                    //Merge these regions
                    overLapRegions[j].x = std::min(overLapRegions[j].x, sZ);
                    overLapRegions[j].y = std::max(overLapRegions[j].y, eZ);
                    regionsOverlap      = j;
                }
            }//for overLapRegions.size()

            if(regionsOverlap == -1) overLapRegions.push_back(make_uint4(sZ,eZ,0, 0)); //new region

            //Store the info for this process
            int regionIdx     = (regionsOverlap >= 0) ? regionsOverlap : overLapRegions.size()-1;
            uint4 overlapProc = make_uint4(sZ,eZ, regionIdx,i);
            overLapPerProc.push_back(overlapProc);
        }//if overlap
    }//For nProcs


    //Sanity check, make sure that there are no overlapping regions. This could happen for
    //example if there are two non-overlapping with a third being added that overlaps with both
    //If this assert is triggered we have to add another loop to merge overlapping regions and update
    //the indices in overLapPerProc
    for(uint i=0; i < overLapRegions.size(); i++)
        for(uint j=i+1; j < overLapRegions.size(); j++)
            assert(testOverlap(overLapRegions[i].x, overLapRegions[i].y, overLapRegions[j].x, overLapRegions[j].y) == 0);

    ASTRA_DEBUG("ProcX: %d Method: %d Regions: %ld overlaps: %ld",procId, method, overLapRegions.size(), overLapPerProc.size());
    for(uint i=0; i < overLapPerProc.size(); i++)
    {
        ASTRA_DEBUG("ProcX: %d Overlap method: %d for process: %d  | Region: %d from: %d to %d ( %d )",
                procId, method, overLapPerProc[i].w,overLapPerProc[i].z,
                overLapPerProc[i].x,overLapPerProc[i].y, overLapPerProc[i].y-overLapPerProc[i].x);
    }
} //createOverlapregions
 

void CMPIProjector3D::exchangeData(std::vector<uint4> &overLapRegionsSend,
				   std::vector<uint4> &overLapPerProcSend,
				   std::vector<uint4> &overLapPerProcRecv,
				   std::vector< std::vector<float> > &regionSendBuffers,
				   std::vector< std::vector<float> > &regionRecvBuffers,
				   const int itemsPerSlice)
{                   
 
#if USE_MPI

    //Fill the buffer that we check during the communication phase with the data that indicates if there are overlaps
  uint4 overLapSend[nProcs];
  for(         int i=0; i < nProcs; i++)                    overLapSend[i]                       = make_uint4(0,0,i,0);
  for(unsigned int i=0; i < overLapPerProcSend.size(); i++) overLapSend[overLapPerProcSend[i].w] = overLapPerProcSend[i];

  uint4 overLapRecv[nProcs];
  for(         int i=0; i < nProcs; i++)                    overLapRecv[i]                       = make_uint4(0,0,i,0);
  for(unsigned int i=0; i < overLapPerProcRecv.size(); i++) overLapRecv[overLapPerProcRecv[i].w] = overLapPerProcRecv[i];

  //Communication phase, send and receive from different processes
  MPI_Status stat;
  for(int i=1; i < nProcs; i++) 
  {
          const int src    = (nProcs + procId - i) % nProcs;
          const int dst    = (nProcs + procId + i) % nProcs;

          //Compute number of slices and startslice location for the process we send to
          int    startSlice      = overLapSend[dst].x;
          int    endSlice        = overLapSend[dst].y;
          const size_t sItems    = itemsPerSlice*(endSlice-startSlice);

	  int sendOffset 	= 0;
	  if(overLapRegionsSend.size() > overLapSend[dst].z)
          	sendOffset      = startSlice-overLapRegionsSend[overLapSend[dst].z].x;
	  sendOffset            *= itemsPerSlice; 

          //Compute the number of items to receive
          startSlice             = overLapRecv[src].x;
          endSlice               = overLapRecv[src].y;
          const size_t rItems    = itemsPerSlice*(endSlice-startSlice);
          regionRecvBuffers[src].resize(rItems);
          
          assert(sItems < INT_MAX);    //TODO, if this is triggered we should change the below in a loop
          assert(rItems < INT_MAX);    //TODO, if this is triggered we should change the below in a loop
          assert(sItems >= 0);  
          assert(rItems >= 0);
          //If there is data to send/receive call the MPI_SendRecv function otherwise go to the next processes
          //Commented out since otherwise we get a deadlock
          //       if(sItems > 0 || rItems > 0)
          {
                  MPI_Sendrecv(&regionSendBuffers[overLapSend[dst].z][sendOffset], (int)sItems, MPI_FLOAT, dst, 145,
                               &regionRecvBuffers[src][0],                         (int)rItems, MPI_FLOAT, src, 145,
                               MPI_COMM_WORLD, &stat);
          }
  } //For      
        
 //TODO, test if replacing Sendrecv by Isend/irecv improves performance
 //it reduces/eliminates the 0-size communications that are required
 //to prevent deadlocks
#endif
}


//exType: 0 Volume, 1 Minimum Proj Ghost, 2: Full Proj Ghost 3: Sum PrjOverlap
void CMPIProjector3D::exchangeOverlapAndGhostRegions(float *volData,
				  	   cudaPitchedPtr &D_data,
				  	   bool dataOnHost,
					   int  exType)
{
#if USE_MPI
	static std::vector< std::vector<float> > regionSendBuffers(nProcs);
	static std::vector< std::vector<float> > receiveBuffers(nProcs);
	    
	astraCUDA3d::SDimensions3D dims = processExtends[procId].dims; 
	int itemsPerSlice 		= dims.iProjU*dims.iProjAngles; 


	std::vector<uint4> regionSend, regionRecv, procSend, procRecv;


	if(exType == 0)
	{
	    regionSend 		= overLapRegionsVolSend;
	    regionRecv 		= overLapRegionsVolRecv;
	    procSend   		= overLapPerProcVolSend;
	    procRecv    	= overLapPerProcVolRecv;
	    itemsPerSlice 	= dims.iVolX*dims.iVolY;
	}
	else if(exType == 1)
	{
	    regionSend = overLapRegionsPrjSend;
	    regionRecv = overLapRegionsPrjRecv;
	    procSend   = overLapPerProcPrjSend;
	    procRecv   = overLapPerProcPrjRecv;
	}
	else if(exType == 2)
	{
	    regionSend = overLapRegionsPrjSendFull;
	    regionRecv = overLapRegionsPrjRecvFull;
	    procSend   = overLapPerProcPrjSendFull;
	    procRecv   = overLapPerProcPrjRecvFull;
	}
	else if(exType == 3)
	{
	    regionSend = overLapRegionsDetector;
	    regionRecv = overLapRegionsDetector;
	    procSend   = overLapPerProcDetector;
	    procRecv   = overLapPerProcDetector;
	}
	else
	{
	    assert(0);
	}

        //Copy the overlapping regions to the host
        for(unsigned int i=0; i < regionSend.size(); i++)
        {
            int startSlice = regionSend[i].x;
            int endSlice   = regionSend[i].y;
            int items      = itemsPerSlice*(endSlice-startSlice);
            regionSendBuffers[i].resize(items);

	    if(dataOnHost)
  	    {
            	//Copy from the full host buffer to the temporary MPI buffers
           	int readOffset = startSlice*itemsPerSlice;
            	memcpy(&regionSendBuffers[i][0], &volData[readOffset], items*sizeof(float));
		//for(int j=0; j < items; j++) volData[readOffset+j] = 0.5 + procId; //used for marking regions
	    }
	    else
	    {
		if(exType == 0)
		{
	          this->copyVolumeFromDeviceExtended(
				&regionSendBuffers[i][0], D_data, dims, 0,
                                0, 0, startSlice,
                                dims.iVolX, dims.iVolY, endSlice-startSlice);
		}
		else
		{
	          this->copyProjectionsFromDeviceExtended(
				&regionSendBuffers[i][0], D_data, dims, 0,
                                0, 0, startSlice,
                                dims.iProjU, dims.iProjAngles, endSlice-startSlice);
		}
	    }
//for(int j=0; j < items; j++) regionSendBuffers[i][j] = 100+procId; //Used for marking regions

        }
	//Exchange
	exchangeData(regionSend, procSend, procRecv,
		     regionSendBuffers, receiveBuffers, itemsPerSlice);


	if(exType == 3)
	{
	  //Sum the received data with our local detector data
	   for(unsigned int i=0; i < procRecv.size(); i++)
	   {
	     int    startSlice   = procRecv[i].x;
	     int    endSlice     = procRecv[i].y;
	     size_t items        = itemsPerSlice*(endSlice-startSlice);
	     int    writeOffset  = (startSlice-regionRecv[procRecv[i].z].x)*itemsPerSlice;

	     #pragma omp parallel for
	     for(unsigned int idx=0; idx < items; idx++)
	     {
	       regionSendBuffers[procRecv[i].z][writeOffset+idx] += receiveBuffers[procSend[i].w][idx];
  	     }
	  }//for
	  //Overwrite procRecv by regionSend info in order to use procRecv loop to copy data back
	  procRecv.clear();
	  for(unsigned int i=0; i < regionSend.size(); i++) {
		  procRecv.push_back(make_uint4(regionSend[i].x, regionSend[i].y, 0, i)); 
	  }
	} //exType 3


	//Assign source buffer based on which method we use
	auto &srcBuff = (exType != 3) ? receiveBuffers : regionSendBuffers;
	
	//Copy the overlapping regions back to the device
	for(unsigned int i=0; i < procRecv.size(); i++)
        {
            int startSlice = procRecv[i].x;
            int endSlice   = procRecv[i].y;
  	    int src        = procRecv[i].w;

	    if(dataOnHost)
	    {
            	//Copy from the temporary MPI buffers to the full host buffer
            	int writeOffset = startSlice*itemsPerSlice;
            	memcpy(&volData[writeOffset], &srcBuff[src][0], srcBuff[src].size()*sizeof(float));
	    }
	    else	    
	    {
		if(exType == 0)
		{
		   this->copyVolumeToDeviceExtended(
				&srcBuff[src][0], D_data, dims, 
				0, 0, 0, startSlice,
                                dims.iVolX, dims.iVolY, endSlice-startSlice);
		}
		else
		{
	          this->copyProjectionsToDeviceExtended(
				&srcBuff[src][0], D_data, dims, 0,
                                0, 0, startSlice,
                                dims.iProjU, dims.iProjAngles, endSlice-startSlice);
		}
	    }
        } //for

#endif
} //exchangeGhostRegions


void CMPIProjector3D::sync() 
{
#if USE_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif	
}

template <class T, class S>
T* CMPIProjector3D::combineData(
		    astra::CFloat32Data3DMemory*  pReconMem,
		    S* pGeometry)
{
	T* pReconDataFull = nullptr;
	int    idx    	  = 0;

	if(dynamic_cast<astra::CProjectionGeometry3D*>(pGeometry))
		idx = 1;  //Used to retrieve projection properties

	size_t sendSize = 1024*1024*256; //Maximum data send size; 1/4 Billion floats = 1GB, and smaller than INT_MAX	

#if USE_MPI		    
	int    res 	  = 0;
	//Recombine the results
	if(procId == 0)
	{
		pReconDataFull = new T(pGeometry, 0.0f);

		size_t writeIdx = 0;	
		//Distribute the data
		for(int i=0; i < nProcs; i++)
	        {
		    const int    nCols    = this->getNumberOfCols(i,idx);   //Width
		    const int    nRows    = this->getNumberOfRows(i,idx);   //Height

		    int start=0, end=0; 
		    if(idx == 0)
		    {
			start = this->getResponsibleVolStartIndex(i);
			end   = this->getResponsibleVolEndIndex(i);
		    }
		    else
		    {
			start = this->getResponsibleProjStartIndex(i);
			end   = this->getResponsibleProjEndIndex(i);
		    }
		
		    size_t nItems    = (end-start)*nCols*nRows;


		    for(size_t recvIdx = 0; recvIdx < nItems; recvIdx += sendSize)
 		    {
			MPI_Status status;
			const size_t nItemsTosend = std::min(sendSize, nItems-recvIdx);
			if( i != 0)
			{
			  res   = MPI_Recv(&pReconDataFull->getData()[writeIdx], nItemsTosend,
		  			   MPI_FLOAT, i, 102, MPI_COMM_WORLD, &status);
			  assert(res == MPI_SUCCESS);
			}
			else
		 	{
			   //Local process local copy
  			   memcpy(&pReconDataFull->getData()[writeIdx], 
			   	  &pReconMem->getData()[writeIdx],
			          nItemsTosend*sizeof(float));
			}
			writeIdx += nItemsTosend;
		    }			    
	        }//for _nWorkerProcs		
	}
	else
	{
		const int    nCols    = this->getNumberOfCols(procId,idx);   //Width
		const int    nRows    = this->getNumberOfRows(procId,idx);   //Height

		int start = 0, end = 0;
	        if(idx == 0)
		{
			start = this->getResponsibleVolStartIndex();
			end   = this->getResponsibleVolEndIndex();
		}
		else
		{
			start = this->getResponsibleProjStartIndex();
			end   = this->getResponsibleProjEndIndex();
		}
		size_t readIdx = start*nCols*nRows;
		size_t nItems  = (end-start)*nCols*nRows; 
		    
		for(size_t sendIdx = 0; sendIdx < nItems; sendIdx += sendSize)
		{
		    const size_t nItemsTosend = std::min(sendSize, nItems-sendIdx);
		    res = MPI_Send(&pReconMem->getData()[readIdx+sendIdx], nItemsTosend, 
				   MPI_FLOAT, 0, 102, MPI_COMM_WORLD);
		    assert(res == MPI_SUCCESS);
		}
	}

	return pReconDataFull;

#else
//Not using MPI
	pReconDataFull = new T(pGeometry, 0.0f);

	const int    nCols    = this->getNumberOfCols(procId,idx);   //Width
	const int    nRows    = this->getNumberOfRows(procId,idx);   //Height
	const size_t nItems   = this->getNumberOfSlices(procId, idx) * nCols * nRows;
	//const int    startIdx = this->getStartSlice(procId,idx)      * nCols * nRows;
	
	//size_t writeIdx = 0;	
	for(size_t writeIdx = 0; writeIdx < nItems; writeIdx += sendSize)
 	{
		//Local process local copy
		const size_t nItemsTosend = std::min(sendSize, nItems-writeIdx);
  		memcpy(&pReconDataFull->getData()[writeIdx], &pReconMem->getData()[writeIdx], nItemsTosend*sizeof(float));
	}

	return pReconDataFull;
#endif	


}

//Function definitions that we use to combine either volume 
//or projection data buffers
//

template astra::CFloat32ProjectionData3DMemory*
       	CMPIProjector3D::combineData<astra::CFloat32ProjectionData3DMemory,
	 	    astra::CProjectionGeometry3D>(
				astra::CFloat32Data3DMemory*  pReconMem,
				astra::CProjectionGeometry3D*);

template astra::CFloat32VolumeData3DMemory*
       	CMPIProjector3D::combineData<astra::CFloat32VolumeData3DMemory,
	 	    astra::CVolumeGeometry3D>(
				astra::CFloat32Data3DMemory*  pReconMem,
				astra::CVolumeGeometry3D*);



void CMPIProjector3D::distributeData(astra::CFloat32Data3DMemory* pOutputMemData,
		    		     astra::CFloat32Data3DMemory* pInputDataFull)
{
	int idx = 1;
	if(dynamic_cast<astra::CFloat32VolumeData3DMemory*>(pOutputMemData))
		idx = 0; //Forward Projection, input is volume

	size_t sendSize = 1024*1024*256; //Maximum data send size; 1/4 Billion floats = 1GB, and smaller than INT_MAX	

#if USE_MPI
	int res = 0;
	if(procId == 0)
	{
		//Distribute the data
		for(int i=0; i < nProcs; i++)
	        {
		    const int    nCols    = this->getNumberOfCols(i,idx);   //Width
		    const int    nRows    = this->getNumberOfRows(i,idx);   //Height
		    const size_t nItems   = this->getNumberOfSlices(i, idx) * nCols * nRows;
		    const int    startIdx = this->getStartSlice(i,idx)      * nCols * nRows;

		    for(size_t sendIdx = 0; sendIdx < nItems; sendIdx += sendSize)
		    {
			const size_t nItemsTosend = std::min(sendSize, nItems-sendIdx);
			if(i != 0)
			{
				res   = MPI_Send(&pInputDataFull->getData()[startIdx+sendIdx], nItemsTosend, MPI_FLOAT, i, 101, MPI_COMM_WORLD);
				assert(res == MPI_SUCCESS);
			}
			else
			{
				//Local process local copy
				memcpy(&pOutputMemData->getData()[sendIdx], 
				       &pInputDataFull->getData()[startIdx+sendIdx],
				       nItemsTosend*sizeof(float));
			}
		    }
	        }//for _nWorkerProcs
	} //if procId == 0
	else
	{
		//Receive our data
		MPI_Status status;
		const int    nCols    = this->getNumberOfCols(procId,idx);   //Width
		const int    nRows    = this->getNumberOfRows(procId,idx);   //Height
		const size_t nItems   = this->getNumberOfSlices(procId, idx) * nCols * nRows;
		for(size_t recvIdx = 0; recvIdx < nItems; recvIdx += sendSize)
		{
			const size_t nItemsTosend = std::min(sendSize, nItems-recvIdx);
			res   = MPI_Recv(&pOutputMemData->getData()[recvIdx], nItemsTosend, 
					 MPI_FLOAT, 0, 101, MPI_COMM_WORLD, &status);
			assert(res == MPI_SUCCESS);
		}	
	}
#else
	//Move data from one buffer to the other
	const int    nCols    = this->getNumberOfCols(procId,idx);   //Width
	const int    nRows    = this->getNumberOfRows(procId,idx);   //Height
	const size_t nItems   = this->getNumberOfSlices(procId, idx) * nCols * nRows;
	
	for(size_t writeIdx = 0; writeIdx < nItems; writeIdx += sendSize)
 	{
		//Local process local copy
		const size_t nItemsTosend = std::min(sendSize, nItems-writeIdx);
  		memcpy(&pOutputMemData->getData()[writeIdx], 
		       &pInputDataFull->getData()[writeIdx], 
		       nItemsTosend*sizeof(float));
	}

#endif	
} //distributeData



float CMPIProjector3D::sum(float in)
{
	float res = in;
#if USE_MPI
	MPI_Allreduce(&in, &res, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#endif	
	return res;
}



bool CMPIProjector3D::copyProjectionsFromDeviceExtended(float* data, const cudaPitchedPtr& D_data, const astraCUDA3d::SDimensions3D& dims,
                                                                           unsigned int pitch,  int startX, int startY, int startZ,
                                                                           int nX, int nY, int nZ) 
{
        if (!pitch)
                pitch = dims.iProjU;

        cudaPitchedPtr ptr;
        ptr.ptr = data;
        ptr.pitch = pitch*sizeof(float);
        ptr.xsize = dims.iProjU*sizeof(float);
        ptr.ysize = dims.iProjAngles;

        //The extend is specified by the number of rows, columns and slices specified in nX, nY, nZ
        cudaExtent extentV;
        extentV.width  = nX*sizeof(float);  //takes jumps of 1
        extentV.height = nY;                //takes jumps of projection width
        extentV.depth  = nZ;                //takes jumps of a full slice

        //Specify the start location of the copy in 3D coordinates
        cudaPos srcPos = make_cudaPos(startX*sizeof(float), startY, startZ);

        cudaPos zp = make_cudaPos(0, 0, 0); 

        cudaMemcpy3DParms p = {0};
        p.srcArray = 0;
        p.srcPos   = srcPos;
        p.srcPtr   = D_data;
        p.dstArray = 0;
        p.dstPos   = zp; 
        p.dstPtr   = ptr;
        p.extent   = extentV;
        p.kind     = cudaMemcpyDeviceToHost;

        cudaError err;
        err = cudaMemcpy3D(&p);
        ASTRA_CUDA_ASSERT(err);

        return err == cudaSuccess;
}

bool CMPIProjector3D::copyVolumeFromDeviceExtended(float* data, const cudaPitchedPtr& D_data, 
				  const astraCUDA3d::SDimensions3D& dims, unsigned int pitch,
				  int startX, int startY, int startZ, 
				  int nX, int nY, int nZ)
{
	if (!pitch)
		pitch = dims.iVolX;

	cudaPitchedPtr ptr;
	ptr.ptr   = data;
	ptr.pitch = pitch*sizeof(float);
	ptr.xsize = dims.iVolX*sizeof(float);
	ptr.ysize = dims.iVolY;

	cudaExtent extentV;
	extentV.width  = nX*sizeof(float);
	extentV.height = nY;
	extentV.depth  = nZ;

	cudaPos srcPos = make_cudaPos(startX*sizeof(float), startY, startZ);
	cudaPos zp     = { 0, 0, 0 };

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos   = srcPos;
	p.srcPtr   = D_data;
	p.dstArray = 0;
	p.dstPos   = zp;
	p.dstPtr   = ptr;
	p.extent   = extentV;
	p.kind     = cudaMemcpyDeviceToHost;

	cudaError err;
	err = cudaMemcpy3D(&p);
	ASTRA_CUDA_ASSERT(err);

	return err == cudaSuccess;
}

bool CMPIProjector3D::copyProjectionsToDeviceExtended(float* data, const cudaPitchedPtr& D_data, const astraCUDA3d::SDimensions3D& dims,
                                                                         unsigned int pitch, int startX, int startY, int startZ,
                                                                         int nX, int nY, int nZ) 
{
        if (!pitch)
                pitch = dims.iProjU;

        cudaPitchedPtr ptr;
        ptr.ptr = data;
        ptr.pitch = pitch*sizeof(float);
        ptr.xsize = dims.iProjU*sizeof(float);
        ptr.ysize = dims.iProjAngles;

        cudaExtent extentV;
        extentV.width  = nX*sizeof(float);  //takes jumps of 1
        extentV.height = nY;                //takes jumps of projection width
        extentV.depth  = nZ;                //takes jumps of a full slice

        //Specify the destination location of the copy in 3D coordinates
        cudaPos dstPos = make_cudaPos(startX*sizeof(float), startY, startZ);

        cudaPos zp = make_cudaPos(0, 0, 0);

        cudaMemcpy3DParms p = {0};
        p.srcArray = 0;
        p.srcPos   = zp;
        p.srcPtr   = ptr;
        p.dstArray = 0;
        p.dstPos   = dstPos;
        p.dstPtr   = D_data;
        p.extent   = extentV;
        p.kind     = cudaMemcpyHostToDevice;

        cudaError err;
        err = cudaMemcpy3D(&p);
        ASTRA_CUDA_ASSERT(err);

        return err == cudaSuccess;
}

bool CMPIProjector3D::copyVolumeToDeviceExtended(const float* data, cudaPitchedPtr& D_data,
	       			const astraCUDA3d::SDimensions3D& dims, unsigned int pitch,
				int startX, int startY, int startZ,
				int nX, int nY, int nZ)
{
	if (!pitch)
		pitch = dims.iVolX;

	cudaPitchedPtr ptr;
	ptr.ptr = (void*)data; // const cast away
	ptr.pitch = pitch*sizeof(float);
	ptr.xsize = dims.iVolX*sizeof(float);
	ptr.ysize = dims.iVolY;

	cudaExtent extentV;
	extentV.width  = nX*sizeof(float);
	extentV.height = nY;
	extentV.depth  = nZ;

	cudaPos dst = make_cudaPos(startX*sizeof(float), startY, startZ);
	cudaPos zp  = { 0, 0, 0 };

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos   = zp;
	p.srcPtr   = ptr;
	p.dstArray = 0;
	p.dstPos   = dst;
	p.dstPtr   = D_data;
	p.extent   = extentV;
	p.kind     = cudaMemcpyHostToDevice;

	cudaError err;
	err = cudaMemcpy3D(&p);
	ASTRA_CUDA_ASSERT(err);

	return err == cudaSuccess;
}


} // end namespace


