
#ifndef _INC_ASTRA_CUDASARTALGORITHM
#define _INC_ASTRA_CUDASARTALGORITHM

#include "Globals.h"
#include "Config.h"

#include "CudaReconstructionAlgorithm2D.h"

#include "astra/cuda/2d/sart.h"

#ifdef ASTRA_CUDA

namespace astra {

class _AstraExport CCudaSartAlgorithm : public CCudaReconstructionAlgorithm2D
{

public:
    
    // type of the algorithm, needed to register with CAlgorithmFactory
    static std::string type;
    
	/** Default constructor, containing no code.
	 */
	CCudaSartAlgorithm();
	
	/** Destructor.
	 */
	virtual ~CCudaSartAlgorithm();
    
    /** Initialize the algorithm with a config object.
         *
         * @param _cfg Configuration Object
         * @return initialization successful?
         */
    virtual bool initialize(const Config& _cfg);
    
    /** Initialize class.
     * @param _pProjector        Projector Object. (Optional)
     * @param _pSinogram        ProjectionData2D object containing the sinogram data.
     * @param _pReconstruction    VolumeData2D object for storing the reconstructed volume.
     * @param  std::string          Projection Order in Measurement Matrix.
     */
    bool initialize(CProjector2D* _pProjector,
                    CFloat32ProjectionData2D* _pSinogram,
                    CFloat32VolumeData2D* _pReconstruction);
    
    void updateProjOrder(std::string& projOrder);
    
    void updateSlice(CFloat32ProjectionData2D* _pSinogram,
                     CFloat32VolumeData2D* _pReconstruction);
    
    void setRelaxationParameter(float lambda);
    
    astraCUDA::SART *sart;
 
    float m_fLambda;
    
protected:

    virtual void initCUDAAlgorithm();

};

} // end namespace

#endif // ASTRA_CUDA

#endif
