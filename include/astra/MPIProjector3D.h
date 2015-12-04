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

#ifndef INC_ASTRA_MPIPROJECTOR3D
#define INC_ASTRA_MPIPROJECTOR3D

#include <cmath>
#include <vector>
#include <limits.h>
#include <string.h>

#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/Logging.h"
#include "astra/Projector3D.h"
#include "astra/Float32Data3DMemory.h"
#include "astra/Float32ProjectionData3DMemory.h"
#include "astra/Float32VolumeData3DMemory.h"


#include "Globals.h"
#include "Config.h"
#include "../cuda/3d/dims3d.h"
#include "../cuda/2d/util.h"
#include "cuda_runtime.h" 

#ifndef USE_MPI
  #define nullptr NULL
#endif


namespace astra
{

/** This is a three-dimensional MPI-projector.
 *  It contains the domain decomposition settings based on the
 *  given Projection and Volume geometries
 */


typedef struct projectionAndVolumeExtends
{
    int  iVolumeOffset;                 ///< Offset/Start-slice in the global volume
    int  iProjOffset;                   ///< Offset/Start-slice in the global projection domain

    int  iProjectionFullSize;           ///< Total number of slices in the projection data
    int  iVolumeFullSize;               ///< Total number of slices in the volume data

    float fDetectorStartZ;              ///< The start location of the detector in the full volume, full detector space
    float fVolumeDetModificationZ;      ///< The modification required to properly set the volume location

    float fDetectorSpacingX;
    float fDetectorSpacingY;

    	     int  iResponsibleProjStart;	///< The start detector slice that we are responsible for
    unsigned int  iResponsibleProjEnd;		///< The first detector slice that we are NOT responsible for. EXCLUSIVE.
    	     int  iResponsibleVolStart;         ///< The start volume slice that we are responsible for
    unsigned int  iResponsibleVolEnd;           ///< The first volume slice that we are NOT responsible for. EXCLUSIVE.

    astraCUDA3d::SDimensions3D dims;	///< Holds the dimensions of this process in the SDimensions3D structure

    int2 iGhostCellCount; 		///< The sizes of the buffer areas/ghost cell areas in the sub-volume. X is top area, Y is bottom area.
    int2 iGhostCellCountPrj; 		///< The sizes of the buffer areas/ghost cell areas in the sub-projector. X is top area, Y is bottom area.
    int2 iGhostCellCountPrjDef; 		///< The sizes of the buffer areas/ghost cell areas in the sub-projector. X is top area, Y is bottom area.


} projectionAndVolumeExtends;


class _AstraExport CMPIProjector3D : public CProjector3D
{

protected:

	/** Check variable values.
	 */
	bool _check();

	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	 * Should only be used by constructors.  Otherwise use the clear() function.
	 */
	void _clear();

public:

	// type of the projector, needed to register with CProjectorFactory
	static std::string type;

	/**
	 * Default Constructor.
	 */
	CMPIProjector3D();

	/** Destructor, is virtual to show that we are aware subclass destructor is called.
	 */
	virtual ~CMPIProjector3D();
	
	/** Clear all member variables, setting all numeric variables to 0 and all pointers to NULL. 
	 */
	void clear();

	/** Initialize the projector with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	//virtual bool initialize(const Config& _cfg);
	virtual bool initialize(const Config& _cfg, const int nGhostCellsVolume, const int nGhostCellsProjection);

	virtual void computeSingleRayWeights(int _iProjectionIndex, 
										 int _iSliceIndex,
										 int _iDetectorIndex, 
										 SPixelWeight* _pWeightedPixels,
		                                 int _iMaxPixelCount, 
										 int& _iStoredPixelCount) {}
	virtual int getProjectionWeightsCount(int _iProjectionIndex) { return 0; }
	template <typename Policy>
	void project(Policy& _policy) {}
	template <typename Policy>
	void projectSingleProjection(int _iProjection, Policy& _policy) {}
	template <typename Policy>
	void projectSingleRay(int _iProjection, int _iSlice, int _iDetector, Policy& _policy) {}



	/** Return the  type of this projector.
	 *
	 * @return identification type of this projector
	 */
	virtual std::string getType() { return type; }

	/** get a description of the class
	 *
	 * @return description string
	 */
	virtual std::string description() const;


	/*
	 * Retrieve the projection geometry for the full dataset of all processes combined
	 * @returns the full projection geometry
	 */
  	CProjectionGeometry3D* getProjectionGlobal() { return m_pProjectionGeometryGlobal; }
	
	/*
	 * Retrieve the projection geometry for this process
	 * @returns the projection geometry of this process only
	 */
  	CProjectionGeometry3D* getProjectionLocal()  { return m_pProjectionGeometry; }
	
	/*
	 * Retrieve the volume geometry for the full dataset of all processes combined
	 * @returns the full volume geometry
	 */
	CVolumeGeometry3D*     getVolumeGlobal()     { return m_pVolumeGeometryGlobal; }
	
	/*
	 * Retrieve the volume geometry for this process
	 * @returns the volume geometry of this process only
	 */
	CVolumeGeometry3D*     getVolumeLocal()      { return m_pVolumeGeometry; }



	/*
	 * Retrieve the number of ghost-slices at the beginning (x) 
	 * and end (y) of our volume.
	 *
	 * @returns the ghost-cell configuration.
	 * */
	int2 getGhostCells() const
	{
		return processExtends[procId].iGhostCellCount;
	}

	int2 getGhostCellsPrj() const
	{
		return processExtends[procId].iGhostCellCountPrj;
	}


	/*
	 * Retrieves the number of slices for a specified process of either the detector or the volume
	 *
	 * @param procIdx The process for which we require the number of slices
	 * @param type Specify the slice type. 0 = volume and 1 = detector
	 * @returns the number of slices
	 *
	 */
	int getNumberOfSlices(int procIdx, int type)
	{
            if(type == 0)
	        return processExtends[procIdx].dims.iVolZ; 
	    else
	        return processExtends[procIdx].dims.iProjV; 
	}

	int getGlobalNumberOfSlices(int procIdx, int type)
	{
            if(type == 0)
	        return processExtends[procIdx].iVolumeFullSize;
	    else
	        return processExtends[procIdx].iProjectionFullSize;
	}

	/*
	 * Retrieves the slice start offset for a specified process of either the detector or the volume
	 *
	 * @param procIdx The process for which we require the start offset
	 * @param type Specify the slice type. 0 = volume and 1 = detector
	 * @returns the startoffset for the slice in the full dataset
	 *
	 */
	int getStartSlice(int procIdx, int type)
	{
            if(type == 0)
	        return processExtends[procIdx].iVolumeOffset; 
	    else
	        return processExtends[procIdx].iProjOffset; 
	}

	/*
	 * Retrieves the number of columns for a specified process of either the detector or the volume
	 *
	 * @param procIdx The process for which we require the number of slices
	 * @param type Specify the column type. 0 = volume and 1 = detector
	 * @returns the number of columns
	 *
	 */
	int getNumberOfCols(int procIdx, int type)
	{
            if(type == 0)
	        return processExtends[procIdx].dims.iVolX; 
	    else
	        return processExtends[procIdx].dims.iProjU; 
	}

	/*
	 * Retrieves the number of rows for a specified process of either the detector or the volume
	 *
	 * @param procIdx The process for which we require the number of rows
	 * @param type Specify the slice type. 0 = volume and 1 = detector
	 * @returns the number of rows
	 *
	 */
	int getNumberOfRows(int procIdx, int type)
	{
            if(type == 0)
	        return processExtends[procIdx].dims.iVolY; 
	    else
	        return processExtends[procIdx].dims.iProjAngles; 
	}

	/*
	 * Retrieve the changes that have to be made to the volume location for this 
	 * process in order to get the correct geometry
	 *
	 * @return the modifications to the volume geometry
	 */
        float getVolumeDetModificationZ() const
        {
                return processExtends[procId].fVolumeDetModificationZ;
        }
	
	/*
	 * Retrieve the changes that have to be made to the detector location for this 
	 * process in order to get the correct geometry
	 *
	 * @return the modifications to the detector geometry
	 */
        float getDetectorStartZ() const
        {
                return processExtends[procId].fDetectorStartZ;
        }

	/* 
	 * Retrieve the startoffset of the detector slices in the full detector
	 * @return the startoffset of the detector slices of the current process
	 */
	int getDetectorStartOffset() 	    { return getDetectorStartOffset(this->procId);}
	
	/* 
	 * Retrieve the startoffset of the detector slices in the full detector 
	 * of the specified process
	 * @param idx the process for which we require the startoffset 
	 * @return the startoffset of the detector slices of process idx
	 */
	int getDetectorStartOffset(int idx) { return processExtends[idx].iProjOffset; }


	/* 
	 * Retrieve the startoffset of the volume slices in the full volume
	 * @return the startoffset of the volume slices of the current process
	 */
	int getVolumeStartOffset() 	   { return getVolumeStartOffset(this->procId);}
	
	/* 
	 * Retrieve the startoffset of the volume slices in the full volume
	 * of the specified process
	 * @param idx the process for which we require the startoffset
	 * @return the startoffset of the volume slices of process idx
	 */
	int getVolumeStartOffset(int idx)  { return processExtends[idx].iVolumeOffset; }

	/* 
	 * Retrieve the startoffset of the volume for which the current process is responsible in the full volume
	 * @return the startoffset of the volume slices of the current process for which the process is responsible
	 */
	int getResponsibleVolStart() 	    {return getResponsibleVolStart(this->procId);}
	
	/* 
	 * Retrieve the startoffset of the volume for which the specified process is responsible in the full volume
	 * @param idx the process for which we require the startoffset
	 * @return the startoffset of the volume slices for which the specified process is responsible
	 */
	int getResponsibleVolStart(int idx) { return processExtends[idx].iResponsibleVolStart;}

	/*
	 * Retrieve the start-slice index in the data array for the volume for which 
	 * the current process is responsible
	 *
	 * @return the start-slice index of the volume for which we are responsible
	 */
	int getResponsibleVolStartIndex() {return getResponsibleVolStartIndex(this->procId);}

	/*
	 * Retrieve the start-slice index in the data array for the volume for which 
	 * the specified process is responsible
	 *
	 * @param idx the process for which we retrieve the start-slice index
	 * @return the start-slice index of the volume for which we are responsible
	 */
	int getResponsibleVolStartIndex(int idx)
	{
		return getResponsibleVolStart(idx) - getVolumeStartOffset(idx);
	}

	/*
	 * Retrieve the end-slice index in the data array for the volume for which 
	 * the current process is responsible
	 *
	 * @return the end-slice index of the volume for which we are responsible
	 */
	int getResponsibleVolEndIndex() {return getResponsibleVolEndIndex(this->procId);}

	/*
	 * Retrieve the end-slice index in the data array for the volume for which 
	 * the specified process is responsible
	 *
	 * @param idx the process for which we retrieve the end-slice index
	 * @return the end-slice index of the volume for which we are responsible
	 */
	int getResponsibleVolEndIndex(int idx)
	{
		return getResponsibleVolEnd(idx) - getVolumeStartOffset(idx);
	}


	/* 
	 * Retrieve the start slice of the detector for which the current process is responsible in the full detector
	 * @return the start index of the detector slices for which the current process is responsible
	 */
	int getResponsibleProjStart() 	     {return getResponsibleProjStart(this->procId); }
	
	/* 
	 * Retrieve the start slice of the detector for which the specified process is responsible in the full detector
	 * @param idx the process for which we require the start index
	 * @return the start index of the detector slices for which the specified process is responsible
	 */
	int getResponsibleProjStart(int idx) {return processExtends[idx].iResponsibleProjStart; }
	
	/*
	 * Retrieve the end-slice index in the data array for the detecotr for which 
	 * the current process is responsible
	 *
	 * @return the emd-slice index of the detector for which we are responsible
	 */
	int getResponsibleProjStartIndex() {return getResponsibleProjStartIndex(this->procId);}

	/*
	 * Retrieve the start-slice index in the data array for the volume for which 
	 * the specified process is responsible
	 *
	 * @param idx the process for which we retrieve the end-slice index
	 * @return the start-slice index of the volume for which we are responsible
	 */
	int getResponsibleProjStartIndex(int idx)
	{
		return getResponsibleProjStart(idx) - getDetectorStartOffset(idx);
	}

	/*
	 * Retrieve the end-slice index in the data array for the detector for which 
	 * the current process is responsible
	 *
	 * @return the end-slice index of the detector for which we are responsible
	 */
	int getResponsibleProjEndIndex() {return getResponsibleProjEndIndex(this->procId);}

	/*
	 * Retrieve the end-slice index in the data array for the detector for which 
	 * the specified process is responsible
	 *
	 * @param idx the process for which we retrieve the start-slice index
	 * @return the end-slice index of the detector for which we are responsible
	 */
	int getResponsibleProjEndIndex(int idx)
	{
		return getResponsibleProjEnd(idx) - getDetectorStartOffset(idx);
	}

	//Get the extends of the detector. With or without the ghostcells
	int4 getProjectionExtends(int idx, bool removeGhostCells)
	{
	    int4 res;
	    res.x = res.z = processExtends[idx].iProjOffset;
	    res.y = res.x + processExtends[idx].dims.iProjV;

	    if(removeGhostCells)
	    {
		    res.x += processExtends[idx].iGhostCellCountPrj.x;
		    res.y -= processExtends[idx].iGhostCellCountPrj.y;
	    }


	    return res;
	}
	
	int4 getVolumeExtends(int idx, bool removeGhostCells)
	{
	    int4 res;
	    res.x = res.z = processExtends[idx].iVolumeOffset;
	    res.y = res.x + processExtends[idx].dims.iVolZ;

	    if(removeGhostCells)
	    {
		    res.x += processExtends[idx].iGhostCellCount.x;
		    res.y -= processExtends[idx].iGhostCellCount.y;
	    }


	    return res;
	}

	int4 getResponsibleVolumeExtends(int idx)
	{
		int4 res;
		res.x = getResponsibleVolStart(idx);
		res.y = getResponsibleVolEnd(idx);
		res.z = getVolumeStartOffset(idx); 
		return res;
	}
	int4 getResponsibleProjectionExtends(int idx)
	{
		int4 res;
		res.x = getResponsibleProjStart(idx);
		res.y = getResponsibleProjEnd(idx);
		res.z = getDetectorStartOffset(idx);
		return res;
	}

	/*
	 * Test if the library is built with MPI support
	 *
	 * @returns if the library is built with MPI support
	 */	
	bool isBuiltWithMPI(); 


	/*
	 * Retrieve this proccess rank / process id in the MPI world
	 * @return the unique procces identifier
	 */
	int getProcId() {return procId;}

	/*
	 * Retrieve the number of proccess in the MPI world
	 * @return the number of processes
	 */
	int getNProcs() {return nProcs;}

	/*
	 * Global barrier. The MPI processes wait untill all processes
	 * have reached this point in execution.
	 */
	void sync();

	/* 
	 * Exchange and summate the overlapping detector regions
	 *
	 * @param prjData the host buffer that contains the data to be exchanged. NULL if data is on the GPU
	 * @param D_data the device/GPU buffer that contains the data to be exchanged. Not used if data is on host
	 * @param dataOnHost indicates if the data is in the prjData buffer or in the D_data buffer
	 *
	 * @return if the exchange was successfull
	 */ 
	void exchangeOverlapRegions(float          *prjData, 
				    cudaPitchedPtr &D_data,
				    bool dataOnHost = true)
	{
		//Method 3: Overlapping projection results that have to be summed
		this->exchangeOverlapAndGhostRegions(prjData, D_data, dataOnHost, 3);
	}



	/* 
	 * Exchange the ghostcell regions of the volume
	 *
	 * @param volData the host buffer that contains the data to be exchanged. NULL if data is on the GPU
	 * @param D_data the device/GPU buffer that contains the data to be exchanged. Not used if data is on host
	 * @param dataOnHost indicates if the data is in the volData buffer or in the D_data buffer
	 * @param exType indiciates which method/data will be exchanged:
	 * 	- 0 Volume ghostcell data
	 * 	- 1 Minimum ghostcells for projection data
	 * 	- 2 Full ghostcells for projection data
	 * 	- 3 Overlapping projection slices that will be summed together.
	 *
	 */ 
	void exchangeOverlapAndGhostRegions(float *volData,
			 	  cudaPitchedPtr &D_data,
			 	  bool dataOnHost = true,
				  int exType  = 0);

	/*
	 * Exchange the ghostcell regions of the volume. This is the same as the `exchangeGhostRegions` function 
	 * with the difference that it requires fewer parameters, making it more suitable to be called from 
	 * Python.
	 * 
	 * @param volData the host buffer that contains the data to be exchanged.U
	 *
	 */
	void pyExchangeGhostRegionsVolume(float *volData)
	{
		assert(volData != nullptr);
		cudaPitchedPtr tmp;
		this->exchangeOverlapAndGhostRegions(volData, tmp, true, 0);
	}
	
	
	void pyExchangeGhostRegionsProjection(float *volData)
	{
		assert(volData != nullptr);
		cudaPitchedPtr tmp;
		this->exchangeOverlapAndGhostRegions(volData, tmp, true, 1);
	}
	
	void pyExchangeGhostRegionsProjectionFull(float *volData)
	{
		assert(volData != nullptr);
		cudaPitchedPtr tmp;
		this->exchangeOverlapAndGhostRegions(volData, tmp, true, 2);
	}


	/*
	 * Exchange projection/volume data with our neighbour processes
	 * 
	 * @param overLapRegionsSend the configurations of regions we send, each process sends a subset of this data 			     
	 * @param overLapPerProcSend the configurations of data we send per process
	 * @param overLapPerProcRecv the configuration of data we receive per process
 	 * @param regionSendBuffers the actual buffers that contain the regions data to send
	 * @param regionRecvBuffers the buffer in which we receive the received regions
	 * @param itemsPerSlice number of items per slice (width*height of a slice)
	 *
	*/
	void exchangeData(std::vector<uint4> &overLapRegionsSend,
		          std::vector<uint4> &overLapPerProcSend,
			  std::vector<uint4> &overLapPerProcRecv,
			  std::vector< std::vector<float> > &regionSendBuffers,
			  std::vector< std::vector<float> > &regionRecvBuffers,
			  const int itemsPerSlice);



	/*
	 * Accumulate a float variable over all processes
	 * @param in the input value
	 * @return the output value that is the sum of all in values
	 */
	float sum(float in);

	/*
	 * Combine buffers from multiple processes together on the root process. This works for projection/detector data
	 * as well as for volume data. Note that each process only sends the data for which it is responsible. So overlapping 
	 * data is not copied twice. The result buffer only contains unique slices.
	 *
	 *
	 * @param pReconMem the input buffer that will be combined with others
	 * @param pGeometry the geometry of the combined buffer
	 * @return The buffer of type T that contains the merged data. Only on the root process, NULL otherwise.
	 *
	 *
	 */
	template <class T, class S>
	T* combineData(astra::CFloat32Data3DMemory*  pReconMem,
		       S* pGeometry);


	/* 
	 * Distribute a data buffer from the root process to the other non-root processes. Note this spreads the buffer
	 * and does not broadcast the buffer.  The domain distribution is used to determine which slices have to go
	 * to which process. This works for both projectioni/detector and volume data.
	 * 
	 * @param pOutputMemData the buffer in which the received data is stored. Needs to be preallocated
	 * @param pInputDataFull the buffer on the root process that contains the data that is to be distributed
	 *
	 */
	void distributeData(astra::CFloat32Data3DMemory* pOutputMemData,
		    	    astra::CFloat32Data3DMemory* pInputDataFull);

protected:
	
	CProjectionGeometry3D*  m_pProjectionGeometryGlobal; ///< Global projection geometry
	CVolumeGeometry3D* 	m_pVolumeGeometryGlobal;     ///< Global volume geometry

private:

	int nProcs; ///< Number of processes in the current MPI environment
	int procId; ///< The ID of this process
	std::vector<projectionAndVolumeExtends> processExtends; ///< Configuration list that holds the domain decomposition
	std::vector<uint4> overLapRegionsDetector;  ///< The data buffer dimensions that are used in the exchange, x = startSlice, y = finalSlice (exclusive), z/w is free
	std::vector<uint4> overLapPerProcDetector;  ///< List of exchange info per process, x = startSlice, y = finalSlice, z is overLapRegion index. The x and y do not have
					    //   to be the same as those in overLapRegions as a process can require a subset of info that is required by another process.
	
	std::vector<uint4> overLapRegionsVolSend;  ///< The data buffer dimensions that are used in the exchange, x = startSlice, y = finalSlice (exclusive), z/w is free
	std::vector<uint4> overLapPerProcVolSend;  ///< List of exchange info per process, x = startSlice, y = finalSlice, z is overLapRegion index. The x and y do not have

	std::vector<uint4> overLapRegionsVolRecv;  ///< The data buffer dimensions that are used in the exchange, x = startSlice, y = finalSlice (exclusive), z/w is free
	std::vector<uint4> overLapPerProcVolRecv;  ///< List of exchange info per process, x = startSlice, y = finalSlice, z is overLapRegion index. The x and y do not have


	//TODO verify comments
	std::vector<uint4> overLapRegionsPrjSend;  ///< The data buffer dimensions that are used in the exchange, x = startSlice, y = finalSlice (exclusive), z/w is free
	std::vector<uint4> overLapPerProcPrjSend;  ///< List of exchange info per process, x = startSlice, y = finalSlice, z is overLapRegion index. The x and y do not have

	std::vector<uint4> overLapRegionsPrjRecv;  ///< The data buffer dimensions that are used in the exchange, x = startSlice, y = finalSlice (exclusive), z/w is free
	std::vector<uint4> overLapPerProcPrjRecv;  ///< List of exchange info per process, x = startSlice, y = finalSlice, z is overLapRegion index. The x and y do not have

	//These ones use full ghost area and not the minimum one as used before with project overlaps
	std::vector<uint4> overLapRegionsPrjSendFull;  ///< The data buffer dimensions that are used in the exchange, x = startSlice, y = finalSlice (exclusive), z/w is free
	std::vector<uint4> overLapPerProcPrjSendFull;  ///< List of exchange info per process, x = startSlice, y = finalSlice, z is overLapRegion index. The x and y do not have

	std::vector<uint4> overLapRegionsPrjRecvFull;  ///< The data buffer dimensions that are used in the exchange, x = startSlice, y = finalSlice (exclusive), z/w is free
	std::vector<uint4> overLapPerProcPrjRecvFull;  ///< List of exchange info per process, x = startSlice, y = finalSlice, z is overLapRegion index. The x and y do not have





	/*
	 * Computes the domain distribution given a projection and volume pair. This function
	 * can not be called from outside the class as it requires that the Projector has been
	 * initialized before.
	 *
	 * @param projectionGeometry the 3D projection geometry that will be divided over the processes
	 * @param volumeGeometry the 3D volume geometry that will be divided over the processes
	 * @param nGhostCells the number of slices that will overlap between the subvolumes on differences processes
	 *
	 */
	void	determineVolumeAndProjectionExtends(CProjectionGeometry3D *projectionGeometry , CVolumeGeometry3D *volumeGeometry, const int nGhostCells,
			const int nGhostCellsPrj);


	/*
	 * Compute the number of slices that are assigned to the specified process (including ghostcells)
	 *
	 * @param count the number of slices in the full volume
	 * @param procId the rank for which to compute the offset
	 * @param nWorkers the number of processes
	 * @returns the number of slices in this processes volume region
	 *
	 */
	int divideOverProcess(const int count, int procId = -1, int nWorkers = -1);


	/*
	 * Compute the startslice in the full volume for the specified process
	 *
	 * @param count the number of slices in the full volume
	 * @param procId the rank for which to compute the offset
	 * @param nWorkers the number of processes
	 * @returns the slice index that is the start of this processes volume region
	 *
	 */
	int divideComputeStartOffset(const int count, int procId = -1, int nWorkers = -1);

	/*
	 * Returns the number of ghostcells/ghostslices given a start offset and height
	 * for a certain process and a full volume size. 
	 *
	 * @param volumeOffset the slice offset in the total volume space
	 * @param volumeHeight the height of the volume/slices for a process
	 * @param fullVolumeSize the total height of the volume/slices over all processes
	 * @param nGhostcells the width of the ghostcell regions
	 * @returns the number of ghostcells at the beginning (in .x) and at the end (in .y) of the volume region
	 */
	int2 divideOverProcessGhostCellCount(const int volumeOffset,
					     const int volumeHeight,
					     const int fullVolumeSize,
					     const int nGhostcells);


	/*
	 * Computes the amount of overlap between the different worker processes  for the
	 * to the projection/detector settings and for the (ghost) volumes. It creates a set of regions that describe the overlap
	 * and info per process if there is overlap. This function is executed by each worker to compare
	 * it's detector settings with that of the other worker settings.Internally it uses the processExtends data.
	 *
	 * Input:
	 *  @param overLapRegions vector to store the overlapping regions
	 *  @param overLapPerProc vector to store the overlap description for each process for which there is overlap
	 *  @param method the area for which we compute the overlaps. Possible values: 
	 *  1) Overlap in the projection domain t hat has to be summed after an FP call
	 *  2) Volume ghostcells, data we send to our neighbours
	 *  3) Volume ghostcells, data we receive from our neighbours
	 *  4) Projection ghostcells, data we send to our neighbours. Exclude slices we send in method 1.
	 *  5) Projection ghostcells, data we receive from our neighbours. Exclude slices we send in method 1.
	 *  6) Projection ghostcells, data we send to our neighbours. Possibly includes slices we send in 1).
	 *  7) Projection ghostcells, data we receive from our neighbours. Possibly includes slices we  send in 1)
	 * Return:
	 *   None, it fills the 'overLapRegions'  and 'overLapPerProc' buffers that describe the overlaps.
	 *
	 */
	void createOverlapRegions(std::vector<uint4> &overLapRegions,
				  std::vector<uint4> &overLapPerProc,
				  const int method);


	/* Copy a subset of the projection data from the device to the host
	 *
	 * @param data host memory buffer in which projection data is placed
	 * @param D_data device memory buffer from which projection data is copied
	 * @param dims projection dimensions
	 * @param pitch pitch of the destination buffer (data)
	 * @param startX start X location of from where to copy 
	 * @param startX start Y location of from where to copy 
	 * @param startX start Z location of from where to copy 
	 * @param nX number of elements in the width direction
	 * @param nY number of elements in the height direction
	 * @param nZ number of slices to copy
	 * @return copy succesfull
	 */
	bool copyProjectionsFromDeviceExtended(float* data, const cudaPitchedPtr& D_data,
					 const astraCUDA3d::SDimensions3D& dims, 
					 unsigned int pitch,  
					 int startX, int startY, int startZ,
					 int nX, int nY, int nZ);



	/* Copy a subset of the volume data from the device to the host
	 *
	 * @param data host memory buffer in which volume data is placed
	 * @param D_data device memory buffer from which volume data is copied
	 * @param dims volume dimensions
	 * @param pitch pitch of the destination buffer (data)
	 * @param startX start X location of from where to copy 
	 * @param startX start Y location of from where to copy 
	 * @param startX start Z location of from where to copy 
	 * @param nX number of elements in the width direction
	 * @param nY number of elements in the height direction
	 * @param nZ number of slices to copy
	 * @return copy succesfull
	 */
	bool copyVolumeFromDeviceExtended(float* data, const cudaPitchedPtr& D_data, 
					  const astraCUDA3d::SDimensions3D& dims, unsigned int pitch,
					  int startX, int startY, int startZ, 
					  int nX, int nY, int nZ);


	/* Copy projection data from the host to a subset of the device projection data
	 *
	 * @param data host memory buffer from which projection data is copied
	 * @param D_data device memory buffer in which projection data is placed
	 * @param dims projection dimensions
	 * @param pitch pitch of the source buffer (data)
	 * @param startX start X location of from where to write 
	 * @param startX start Y location of from where to write
	 * @param startX start Z location of from where to write
	 * @param nX number of elements in the width direction
	 * @param nY number of elements in the height direction
	 * @param nZ number of slices to copy
	 * @return copy succesfull
	 */
	bool copyProjectionsToDeviceExtended(float* data, const cudaPitchedPtr& D_data, 
				     const astraCUDA3d::SDimensions3D& dims,
				     unsigned int pitch, 
				     int startX, int startY, int startZ,
				     int nX, int nY, int nZ);

	/* Copy volume data from the host to a subset of the device volume data
	 *
	 * @param data host memory buffer from which volume data is copied
	 * @param D_data device memory buffer in which volume data is placed
	 * @param dims volume dimensions
	 * @param pitch pitch of the source buffer (data)
	 * @param startX start X location of from where to write 
	 * @param startX start Y location of from where to write
	 * @param startX start Z location of from where to write
	 * @param nX number of elements in the width direction
	 * @param nY number of elements in the height direction
	 * @param nZ number of slices to copy
	 * @return copy succesfull
	 */
	bool copyVolumeToDeviceExtended(const float* data, cudaPitchedPtr& D_data,
	       				 const astraCUDA3d::SDimensions3D& dims, unsigned int pitch,
					 int startX, int startY, int startZ,
					 int nX, int nY, int nZ);



	
	
	/* 
	 * Retrieve the end slice of the volume for which the current process is responsible in the full volume
	 * @return the endslice of the volume slices of the current process for which the process is responsible
	 */
	int getResponsibleVolEnd() 	    {return getResponsibleVolEnd(this->procId);}

	/* 
	 * Retrieve the end slice of the volume for which the specified process is responsible in the full volume
	 * @param idx the process for which we require the endslice
	 * @return the endslice of the volume slices for which the specified process is responsible
	 */
	int getResponsibleVolEnd(int idx)   { return processExtends[idx].iResponsibleVolEnd; }

	

	/* 
	 * Retrieve the end slice of the detector for which the current process is responsible in the full detector
	 * @return the end index of the detector slices for which the current process is responsible
	 */
	int getResponsibleProjEnd() 	   { return getResponsibleProjEnd(this->procId); }
	
	/* 
	 * Retrieve the end slice of the detector for which the specified process is responsible in the full detector
	 * @param idx the process for which we require the end index
	 * @return the end index of the detector slices for which the specified process is responsible
	 */
	int getResponsibleProjEnd(int idx) { return processExtends[idx].iResponsibleProjEnd; }
	

}; //end class

} // namespace astra

#endif /* INC_ASTRA_MPIPROJECTOR3D */
