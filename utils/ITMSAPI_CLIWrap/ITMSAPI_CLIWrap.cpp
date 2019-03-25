// 기본 DLL 파일입니다.

#include "stdafx.h"

#include "ITMSAPI_CLIWrap.h"

namespace ITMSAPI_CLIWrap {
	ITMS_CLIWrap::ITMS_CLIWrap() :m_pNativeClass(new itms::ITMSAPINativeClass){
		;
	}
	ITMS_CLIWrap::~ITMS_CLIWrap() {
		if (m_pNativeClass) {
			delete m_pNativeClass;
			m_pNativeClass = NULL;
		}		
		cout << "Leaving tghe ITMS_CLIWrap!!\n";
	}
	int ITMS_CLIWrap::addValue(int a, int b) {
		return (a + b);
	}
}