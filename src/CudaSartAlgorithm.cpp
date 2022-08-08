#ifdef ASTRA_CUDA

#include "astra/CudaSartAlgorithm.h"

#include "astra/cuda/2d/sart.h"

using namespace std;
using namespace astra;

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CCudaSartAlgorithm::type = "SART_CUDA";

//----------------------------------------------------------------------------------------
// Constructor
CCudaSartAlgorithm::CCudaSartAlgorithm()
{
    m_bIsInitialized = false;
    CCudaReconstructionAlgorithm2D::_clear();
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaSartAlgorithm::~CCudaSartAlgorithm()
{
    // The actual work is done by ~CCudaReconstructionAlgorithm2D
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaSartAlgorithm::initialize(const Config& _cfg)
{
    ASTRA_ASSERT(_cfg.self);
    ConfigStackCheck<CAlgorithm> CC("CudaSartAlgorithm", this, _cfg);

    m_bIsInitialized = CCudaReconstructionAlgorithm2D::initialize(_cfg);

    if (!m_bIsInitialized)
        return false;

    sart = new astraCUDA::SART();

    m_pAlgo = sart;
    m_bAlgoInit = false;

    // projection order
    int projectionCount = m_pSinogram->getGeometry()->getProjectionAngleCount();
    int* projectionOrder = NULL;
    string projOrder = _cfg.self.getOption("ProjectionOrder", "random");
    CC.markOptionParsed("ProjectionOrder");
    if (projOrder == "sequential") {
        projectionOrder = new int[projectionCount];
        for (int i = 0; i < projectionCount; i++) {
            projectionOrder[i] = i;
        }
        sart->setProjectionOrder(projectionOrder, projectionCount);
        delete[] projectionOrder;
    } else if (projOrder == "random") {
        projectionOrder = new int[projectionCount];
        for (int i = 0; i < projectionCount; i++) {
            projectionOrder[i] = i;
        }
        for (int i = 0; i < projectionCount-1; i++) {
            int k = (rand() % (projectionCount - i));
            int t = projectionOrder[i];
            projectionOrder[i] = projectionOrder[i + k];
            projectionOrder[i + k] = t;
        }
        sart->setProjectionOrder(projectionOrder, projectionCount);
        delete[] projectionOrder;
    } else if (projOrder == "custom") {
        vector<float32> projOrderList = _cfg.self.getOptionNumericalArray("ProjectionOrderList");
        projectionOrder = new int[projOrderList.size()];
        for (unsigned int i = 0; i < projOrderList.size(); i++) {
            projectionOrder[i] = static_cast<int>(projOrderList[i]);
        }
        sart->setProjectionOrder(projectionOrder, projectionCount);
        delete[] projectionOrder;
        CC.markOptionParsed("ProjectionOrderList");
    }

    m_fLambda = _cfg.self.getOptionNumerical("Relaxation", 1.0f);
    CC.markOptionParsed("Relaxation");

    return true;
}

//---------------------------------------------------------------------------------------
// Initialize
bool CCudaSartAlgorithm::initialize(CProjector2D* _pProjector,
                                         CFloat32ProjectionData2D* _pSinogram,
                                         CFloat32VolumeData2D* _pReconstruction)
{
    m_bIsInitialized = CCudaReconstructionAlgorithm2D::initialize(_pProjector, _pSinogram, _pReconstruction);

    if (!m_bIsInitialized)
        return false;
    
    m_fLambda = 1.0f;

    sart = new astraCUDA::SART();
    
    m_pAlgo = sart;
    m_bAlgoInit = false;
    
    return true;
}

//----------------------------------------------------------------------------------------

void CCudaSartAlgorithm::updateProjOrder(string& projOrder)
{
    // projection order
    int projectionCount = m_pSinogram->getGeometry()->getProjectionAngleCount();
    int* projectionOrder = NULL;
    
    if (projOrder == "sequential") {
        projectionOrder = new int[projectionCount];
        for (int i = 0; i < projectionCount; i++) {
            projectionOrder[i] = i;
        }
        sart->setProjectionOrder(projectionOrder, projectionCount);
        delete[] projectionOrder;
    } else if (projOrder == "random") {
        projectionOrder = new int[projectionCount];
        for (int i = 0; i < projectionCount; i++) {
            projectionOrder[i] = i;
        }
        for (int i = 0; i < projectionCount-1; i++) {
            int k = (rand() % (projectionCount - i));
            int t = projectionOrder[i];
            projectionOrder[i] = projectionOrder[i + k];
            projectionOrder[i + k] = t;
        }
        sart->setProjectionOrder(projectionOrder, projectionCount);
        delete[] projectionOrder;
    }
}




//----------------------------------------------------------------------------------------

void CCudaSartAlgorithm::updateSlice(CFloat32ProjectionData2D* _pSinogram,
                                    CFloat32VolumeData2D* _pReconstruction)
{
    m_pSinogram = _pSinogram;
    m_pReconstruction = _pReconstruction;
}

//----------------------------------------------------------------------------------------

void CCudaSartAlgorithm::setRelaxationParameter(float lambda)
{
    m_fLambda = lambda;
    sart->setRelaxation(m_fLambda);
}


//----------------------------------------------------------------------------------------

void CCudaSartAlgorithm::initCUDAAlgorithm()
{
    CCudaReconstructionAlgorithm2D::initCUDAAlgorithm();

    astraCUDA::SART* pSart = dynamic_cast<astraCUDA::SART*>(m_pAlgo);

    pSart->setRelaxation(m_fLambda);
}


#endif // ASTRA_CUDA
