
// ITMS_MFC.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CITMS_MFCApp:
// See ITMS_MFC.cpp for the implementation of this class
//

class CITMS_MFCApp : public CWinApp
{
public:
	CITMS_MFCApp();

// Overrides
public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CITMS_MFCApp theApp;