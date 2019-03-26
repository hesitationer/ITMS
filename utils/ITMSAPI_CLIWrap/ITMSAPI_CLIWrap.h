// ITMSAPI_CLIWrap.h

#pragma once
#include "Stdafx.h"
using namespace System;
using namespace itms;

namespace ITMSAPI_CLIWrap {

	public ref class ITMS_CLIWrap
	{
	protected:
		itms::ITMSAPINativeClass* m_pNativeClass;

	public:
		ITMS_CLIWrap();
		virtual ~ITMS_CLIWrap();

		// TODO: 여기에 이 클래스에 대한 메서드를 추가합니다.
		int addValue(int a, int b);

		int Init() { return m_pNativeClass->Init(); };
		int ResetAndProcessFrame(int iCh, unsigned char * pImage, int lSize) { return (m_pNativeClass->ResetAndProcessFrame(iCh, pImage, lSize)); }; // reset and process
		int ResetAndProcessFrame(const cv::Mat& curImg1) { return m_pNativeClass->ResetAndProcessFrame(curImg1); };		
		
		std::vector<std::pair<int, int>> getObjectStatus(void) {return m_pNativeClass->getObjectStatus();};
		std::vector<std::pair<int, int>> getObjectClass(void) {	return m_pNativeClass->getObjectClass();};
		std::vector<cv::Rect> getObjectRect(void) { return m_pNativeClass->getObjectRect(); };
		std::vector<track_t> getObjectSpeed(void) { return m_pNativeClass->getObjectSpeed(); };

		// From Here, you can define the functions to be used in C# as many as possible 		
		int GetObjectNumber(void) { return m_pNativeClass->getObjectClass().size(); };
	};
}
