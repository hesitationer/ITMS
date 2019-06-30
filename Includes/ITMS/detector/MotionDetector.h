
#pragma once

#include "BaseDetector.h"
#include "BackgroundSubtract.h"
#include "utils\itms_utils.h"

///
/// \brief The MotionDetector class
///
class MotionDetector : public BaseDetector
{
public:
    MotionDetector(BackgroundSubtract::BGFG_ALGS algType, bool collectPoints, cv::UMat& gray);
    ~MotionDetector(void);

    bool Init(const config_t& config);

    void Detect(cv::UMat& gray);

	void CalcMotionMap(cv::Mat frame);

	cv::UMat GetForeGround(void) { return m_fg; };
	regions_t getRegions(void) { return m_regions; };
	cv::UMat GetBackGround(void) { return m_backgroundSubst->GetBGImg(); };

private:
    void DetectContour();

    std::unique_ptr<BackgroundSubtract> m_backgroundSubst;

    cv::UMat m_fg;

    BackgroundSubtract::BGFG_ALGS m_algType;	
};