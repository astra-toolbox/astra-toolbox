/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
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

#include "astra/ProjectionGeometry3DFactory.h"

#ifdef USE_MATLAB_UNDOCUMENTED
extern "C" {
mxArray *mxCreateSharedDataCopy(const mxArray *pr);
int mxUnshareArray(mxArray *pr, int level);
mxArray *mxUnreference(mxArray *pr);
#if 0
// Unsupported in Matlab R2014b and later
bool mxIsSharedArray(const mxArray *pr);
#endif
}

class CDataStorageMatlab : public astra::CDataMemory<float> {
public:
	// offset allows linking the data object to a sub-volume (in the z direction)
	// offset is measured in floats.
	CDataStorageMatlab(const mxArray* _pArray, bool bUnshare, size_t iOffset)
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
			mxUnshareArray(const_cast<mxArray*>(_pArray), 0);
			//fprintf(stderr, "Unshared:\narray: %p\tdata: %p\n", (void*)_pArray, (void*)mxGetData(_pArray));
		}
		// Then create a (persistent) copy so the data won't be deleted
		// or changed.
		m_pLink = mxCreateSharedDataCopy(_pArray);
		//fprintf(stderr, "SharedDataCopy:\narray: %p\tdata: %p\n", (void*)m_pLink, (void*)mxGetData(m_pLink));
		mexMakeArrayPersistent(m_pLink);
		this->m_pfData = (float *)mxGetData(_pArray);
		this->m_pfData += iOffset;
	}
	virtual ~CDataStorageMatlab() {
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
checkID(const astra::int32 & id, astra::CData3D *& pDataObj)
{
	pDataObj = astra::CData3DManager::getSingleton().get(id);
	return (pDataObj && pDataObj->isFloat32Memory() && pDataObj->isInitialized());
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
astra::CData3D *
allocateDataObject(const std::string & sDataType,
		const mxArray * const geometry, const mxArray * const data,
		const mxArray * const unshare, const mxArray * const zIndex)
{
	astra::CData3D* pDataObject3D = NULL;

	bool bUnshare = true;
	if (unshare)
	{
		if (!mexIsScalar(unshare))
		{
			mexErrMsgTxt("Argument 5 (read-only) must be scalar.");
		}
		// unshare the array if we're not linking read-only
		bUnshare = !(bool)mxGetScalar(unshare);
	}

	mwIndex iZ = 0;
	if (zIndex)
	{
		if (!mexIsScalar(zIndex))
		{
			mexErrMsgTxt("Argument 6 (Z) must be scalar.");
		}
		iZ = (mwSignedIndex)mxGetScalar(zIndex);
	}

	// SWITCH DataType
	if (sDataType == "-vol")
	{
		// Read geometry
		astra::XMLConfig* cfg = structToConfig("VolumeGeometry3D", geometry);
		astra::CVolumeGeometry3D* pGeometry = new astra::CVolumeGeometry3D();
		if (!pGeometry->initialize(*cfg))
		{
			delete pGeometry;
			delete cfg;
			mexErrMsgWithAstraLog("Geometry class could not be initialized.");
		}
		delete cfg;

		// If data is specified, check dimensions
		if (data && !mexIsScalar(data))
		{
			if (! (zIndex
					? checkDataSize(data, pGeometry, iZ)
					: checkDataSize(data, pGeometry)) )
			{
				delete pGeometry;
				mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry.");
			}
		}

		// Initialize data object
		size_t dataSize = pGeometry->getGridColCount();
		dataSize *= pGeometry->getGridRowCount();
		dataSize *= pGeometry->getGridSliceCount();
#ifdef USE_MATLAB_UNDOCUMENTED
		if (unshare) {
			astra::CDataStorage* pHandle = new CDataStorageMatlab(data, bUnshare, iZ);

			// Initialize data object
			pDataObject3D = new astra::CFloat32VolumeData3D(*pGeometry, pHandle);
		}
		else
		{
			astra::CDataStorage* pStorage = new astra::CDataMemory<float>(dataSize);
			pDataObject3D = new astra::CFloat32VolumeData3D(*pGeometry, pStorage);
		}
#else
		astra::CDataStorage* pStorage = new astra::CDataMemory<float>(dataSize);
		pDataObject3D = new astra::CFloat32VolumeData3D(*pGeometry, pStorage);
#endif
		delete pGeometry;
	}
	else if (sDataType == "-sino" || sDataType == "-proj3d" || sDataType == "-sinocone")
	{
		// Read geometry
		astra::XMLConfig* cfg = structToConfig("ProjectionGeometry3D", geometry);
		std::string type = cfg->self.getAttribute("type");
		std::unique_ptr<astra::CProjectionGeometry3D> pGeometry = astra::constructProjectionGeometry3D(type);
		if (!pGeometry) {
			delete cfg;
			std::string message = "'" + type + "' is not a valid 3D geometry type.";
			mexErrMsgTxt(message.c_str());
		}

		if (!pGeometry->initialize(*cfg)) {
			delete cfg;
			mexErrMsgWithAstraLog("Geometry class could not be initialized.");
		}
		delete cfg;

		// If data is specified, check dimensions
		if (data && !mexIsScalar(data))
		{
			if (! (zIndex
					? checkDataSize(data, pGeometry.get(), iZ)
					: checkDataSize(data, pGeometry.get())) )
			{
				mexErrMsgTxt("The dimensions of the data do not match those specified in the geometry.");
			}
		}

		// Initialize data object
		size_t dataSize = pGeometry->getDetectorColCount();
		dataSize *= pGeometry->getProjectionCount();
		dataSize *= pGeometry->getDetectorRowCount();
#ifdef USE_MATLAB_UNDOCUMENTED
		if (unshare)
		{
			astra::CDataStorage* pHandle = new CDataStorageMatlab(data, bUnshare, iZ);

			// Initialize data object
			pDataObject3D = new astra::CFloat32ProjectionData3D(*pGeometry, pHandle);
		}
		else
		{
			astra::CDataStorage* pStorage = new astra::CDataMemory<float>(dataSize);
			pDataObject3D = new astra::CFloat32ProjectionData3D(*pGeometry, pStorage);
		}
#else
		astra::CDataStorage* pStorage = new astra::CDataMemory<float>(dataSize);
		pDataObject3D = new astra::CFloat32ProjectionData3D(*pGeometry, pStorage);
#endif
	}
	else
	{
		mexErrMsgTxt("Invalid datatype. Please specify '-vol' or '-proj3d'.");
	}

	// Check initialization
	if (!pDataObject3D->isInitialized())
	{
		delete pDataObject3D;
		mexErrMsgTxt("Couldn't initialize data object.");
	}

	return pDataObject3D;
}

