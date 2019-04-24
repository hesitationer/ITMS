#include "./detector/BaseDetector.h"
#include "./detector/MotionDetector.h"

///
/// \brief CreateDetector
/// \param detectorType
/// \param collectPoints
/// \param gray
/// \return
///
BaseDetector* CreateDetector(
        itms::tracking::Detectors detectorType,
        const config_t& config,
        bool collectPoints,
        cv::UMat& gray
        )
{
    BaseDetector* detector = nullptr;

    switch (detectorType)
    {
	case itms::tracking::Motion_MOG2:
        detector = new MotionDetector(BackgroundSubtract::BGFG_ALGS::ALG_MOG2, collectPoints, gray);
        break;
    

    default:
        break;
    }

    if (!detector->Init(config))
    {
        delete detector;
        detector = nullptr;
    }
    return detector;
}
