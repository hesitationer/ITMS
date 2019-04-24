










































#pragma once
#ifndef ITMS_BACKGROUNDSUBTRACT_H
#define ITMS_BACKGROUNDSUBTRACT_H


#include "defines.h"

#ifdef USE_OCV_BGFG
#include <opencv2/bgsegm.hpp>
#endif

///
/// \brief The BackgroundSubtract class
///
class BackgroundSubtract
{
public:
	enum BGFG_ALGS
	{
        ALG_VIBE,
        ALG_MOG,
        ALG_GMG,
        ALG_CNT,
        ALG_SuBSENSE,
        ALG_LOBSTER,
        ALG_MOG2
	};

    BackgroundSubtract(BGFG_ALGS algType, int channels);
	~BackgroundSubtract();

    bool Init(const config_t& config);

    void Subtract(const cv::UMat& image, cv::UMat& foreground);
	
	int m_channels;
	BGFG_ALGS m_algType;

private:	
	cv::Ptr<cv::BackgroundSubtractor> m_modelOCV;    
	cv::UMat preImg;
};

#endif // !ITMS_BACKGROUNDSUBTRACT_H