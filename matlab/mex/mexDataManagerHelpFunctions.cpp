/*
-----------------------------------------------------------------------
Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
           2014-2016, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://www.astra-toolbox.com/

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
*/

#include "mexDataManagerHelpFunctions.h"

#include "mexHelpFunctions.h"

#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/Float32VolumeData3DMemory.h"
#include "astra/Float32ProjectionData3DMemory.h"

#define USE_MATLAB_UNDOCUMENTED

#ifdef USE_MATLAB_UNDOCUMENTED
extern "C" {
mxArray *mxCreateSharedDataCopy(const mxArray *pr);
bool mxUnshareArray(mxArray *pr, bool noDeepCopy);
mxArray *mxUnreference(mxArray *pr);
#if 0
// Unsupported in Matlab R2014b and later
bool mxIsSharedArray(const mxArray *pr);
#endif
}

class CFloat32CustomMemoryMatlab3D : public astra::CFloat32CustomMemory {
public:
	// offset allows linking the data object to a sub-volume (in the z direction)
	// offset is measured in floats.
	CFloat32CustomMemoryMatlab3D(const mxArray* _pArray, bool bUnshare, size_t iOffset)
	{
		// Convert from slice to offset
		mwSize dims[3];
		get3DMatrixDims(_pArray, dims);
		iOffset *= dims[0];
		iOffset *= dims[1];

		//fprintf(stderr, "Passed:\narray: %p\tdata: %p\n", (void*)_pArray, (void*)mxGetData(_pArray));
		// First unshare the input array, so that we may modify it.
		if (bUnshare) {
#if 0
			// Unsupported in Matlab R2014b and later
			if (mxIsSharedArray(_pArray)) {
				fprintf(stderr, "Performance note: unsharing shared array in link\n");
			}
#endif
			mxUnshareArray(const_cast<mxArray*>(_pArray), false);
			//fprintf(stderr, "Unshared:\narray: %p\tdata: %p\n", (void*)_pArray, (void*)mxGetData(_pArray));
		}
		// Then create a (persistent) copy so the data won't be deleted
		// or changed.
		m_pLink = mxCreateSharedDataCopy(_pArray);
		//fprintf(stderr, "SharedDataCopy:\narray: %p\tdata: %p\n", (void*)m_pLink, (void*)mxGetData(m_pLink));
		mexMakeArrayPersistent(m_pLink);
		m_fPtr = (float *)mxGetData(_pArray);
		m_fPtr += iOffset;
	}
	virtual ~CFloat32CustomMemoryMatlab3D() {
		// destroy the shared array
		//fprintf(stderr, "Destroy:\narray: %p\tdata: %p\n", (void*)m_pLink, (void*)mxGetData(m_pLink));
		mxDestroyArray(m_pLink);
	}
private:
	mxArray* m_pLink;
};
#endif

//-----------------------------------------------------------------------------------------
bool
checkID(const astra::int32 & id, astra::CFloat32Data3DMemory *& pDataObj)
{
	pDataObj = dynamic_cast<astra::CFloat32Data3DMemory *>(
			astra::CData3DManager::getSingleton().get(id) );
	return (pDataObj && pDataObj->isInitialized());
}

//-----------------------------------------------------------------------------------------
bool
checkDataType(const mxArray * const in)
{
	return (mexIsScalar(in) || mxIsDouble(in) || mxIsSingle(in) || mxIsLogical(in));
}

//-----------------------------------------------------------------------------------------
bool
checkStructs(const mxArray * const in)
{
	return mxIsStruct(in);
}

//-----------------------------------------------------------------------------------------
bool
checkDataSize(const mxArray * const mArray,
		const astra::CProjectionGeometry3D * const geom)
{
	mwSize dims[3];
	get3DMatrixDims(mArray, dims);
	return (geom->getDetectorColCount() == dims[0]
			&& geom->getProjectionCount() == dims[1]
			&& geom->getDetectorRowCount() == dims[2]);
}

//-----------------------------------------------------------------------------------------
bool
checkDataSize(const mxArray * const mArray,
		const astra::CVolumeGeometry3D * const geom)
{
	mwSize dims[3];
	get3DMatrixDims(mArray, dims);
	return (geom->getGridColCount() == dims[0]
			&& geom->getGridRowCount() == dims[1]
			&& geom->getGridSliceCount() == dims[2]);
}

//-----------------------------------------------------------------------------------------
bool
checkDataSize(const mxArray * const mArray,
		const astra::CProjectionGeometry3D * const geom,
		const mwIndex & zOffset)
{
	mwSize dims[3];
	get3DMatrixDims(mArray, dims);
	return (geom->getDetectorColCount() == dims[0]
			&& geom->getProjectionCount() == dims[1]
			&& (zOffset + geom->getDetectorRowCount()) <= dims[2]);
}

//-----------------------------------------------------------------------------------------
bool
checkDataSize(const mxArray * const mArray,
		const astra::CVolumeGeometry3D * const geom,
		const mwIndex & zOffset)
{
	mwSize dims[3];
	get3DMatrixDims(mArray, dims);
	return (geom->getGridColCount() == dims[0]
			&& geom->getGridRowCount() == dims[1]
			&& (zOffset + geom->getGridSliceCount()) <= dims[2]);
}

//-----------------------------------------------------------------------------------------
void
getDataPointers(const std::vector<astra::CFloat32Data3DMemory *> & vecIn,
		std::vector<astra::float32 *> & vecOut)
{
	const size_t tot_size = vecIn.size();
	vecOut.resize(tot_size);
	for (size_t count = 0; count < tot_size; count++)
	{
		vecOut[count] = vecIn[count]->getData();
	}
}

//-----------------------------------------------------------------------------------------
void
getDataSizes(const std::vector<astra::CFloat32Data3DMemory *> & vecIn,
		std::vector<size_t> & vecOut)
{
	const size_t tot_size = vecIn.size();
	vecOut.resize(tot_size);
	for (size_t count = 0; count < tot_size; count++)
	{
		vecOut[count] = vecIn[count]->getSize();
	}
}

//-----------------------------------------------------------------------------------------
astra::CFloat32Data3DMemory *
allocateDataObject(const std::string & sDataType,
		const mxArray * const geometry, const mxArray * const data,
		const mxArray * const unshare, const mxArray * const zIndex)
{
	astra::CFloat32Data3DMemory* pDataObject3D = NULL;

	bool bUnshare = true;
	if (unshare)
	{
		if (!mexIsScalar(unshare))
		{
			mexErrMsgTxt("Argument 5 (read-only) must be scalar");
			return NULL;
		}
		// unshare the array if we're not linking read-only
		bUnshare = !(bool)mxGetScalar(unshare);
	}

	mwIndex iZ = 0;
	if (zIndex)
	{
		if (!mexIsScalar(zIndex))
		{
			mexErrMsgTxt("Argument 6 (Z) must be scalar");
			return NULL;
		}
		iZ = (mwSignedIndex)mxGetScalar(zIndex);
	}

	// SWITCH DataType
	if (sDataType == "-vol")
	{
		// Read geometry
		astra::Config* cfg = structToConfig("VolumeGeometry3D", geometry);
		astra::CVolumeGeometry3D* pGeometry = new astra::CVolumeGeometry3D();
		if (!pGeometry->initialize(*cfg))
		{
			mexErrMsgTxt("Geometry class not initialized. \n");
			delete pGeometry;
			delete cfg;
			return NULL;
		}
		delete cfg;

		// If data is specified, check dimensions
		if (data && !mexIsScalar(data))
		{
			if (! (zIndex
					? checkDataSize(data, pGeometry, iZ)
					: checkDataSize(data, pGeometry)) )
			{
				mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
				delete pGeometry;
				return NULL;
			}
		}

		// Initialize data object
#ifdef USE_MATLAB_UNDOCUMENTED
		if (unshare) {
			CFloat32CustomMemoryMatlab3D* pHandle =
					new CFloat32CustomMemoryMatlab3D(data, bUnshare, iZ);

			// Initialize data object
			pDataObject3D = new astra::CFloat32VolumeData3DMemory(pGeometry, pHandle);
		}
		else
		{
			pDataObject3D = new astra::CFloat32VolumeData3DMemory(pGeometry);
		}
#else
		pDataObject3D = new astra::CFloat32VolumeData3DMemory(pGeometry);
#endif
		delete pGeometry;
	}
	else if (sDataType == "-sino" || sDataType == "-proj3d" || sDataType == "-sinocone")
	{
		// Read geometry
		astra::Config* cfg = structToConfig("ProjectionGeometry3D", geometry);
		// FIXME: Change how the base class is created. (This is duplicated
		// in Projector3D.cpp.)
		std::string type = cfg->self.getAttribute("type");
		astra::CProjectionGeometry3D* pGeometry = 0;
		if (type == "parallel3d") {
			pGeometry = new astra::CParallelProjectionGeometry3D();
		} else if (type == "parallel3d_vec") {
			pGeometry = new astra::CParallelVecProjectionGeometry3D();
		} else if (type == "cone") {
			pGeometry = new astra::CConeProjectionGeometry3D();
		} else if (type == "cone_vec") {
			pGeometry = new astra::CConeVecProjectionGeometry3D();
		} else {
			mexErrMsgTxt("Invalid geometry type.\n");
			return NULL;
		}

		if (!pGeometry->initialize(*cfg)) {
			mexErrMsgTxt("Geometry class not initialized. \n");
			delete pGeometry;
			delete cfg;
			return NULL;
		}
		delete cfg;

		// If data is specified, check dimensions
		if (data && !mexIsScalar(data))
		{
			if (! (zIndex
					? checkDataSize(data, pGeometry, iZ)
					: checkDataSize(data, pGeometry)) )
			{
				mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry. \n");
				delete pGeometry;
				return NULL;
			}
		}

		// Initialize data object
#ifdef USE_MATLAB_UNDOCUMENTED
		if (unshare)
		{
			CFloat32CustomMemoryMatlab3D* pHandle =
					new CFloat32CustomMemoryMatlab3D(data, bUnshare, iZ);

			// Initialize data object
			pDataObject3D = new astra::CFloat32ProjectionData3DMemory(pGeometry, pHandle);
		}
		else
		{
			pDataObject3D = new astra::CFloat32ProjectionData3DMemory(pGeometry);
		}
#else
		pDataObject3D = new astra::CFloat32ProjectionData3DMemory(pGeometry);
#endif
		delete pGeometry;
	}
	else
	{
		mexErrMsgTxt("Invalid datatype.  Please specify '-vol' or '-proj3d'. \n");
		return NULL;
	}

	// Check initialization
	if (!pDataObject3D->isInitialized())
	{
		mexErrMsgTxt("Couldn't initialize data object.\n");
		delete pDataObject3D;
		return NULL;
	}

	return pDataObject3D;
}

