
// ITMS_MFC.h : PROJECT_NAME ���� ���α׷��� ���� �� ��� �����Դϴ�.
//

#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif

#include "resource.h"		// �� ��ȣ�Դϴ�.


// CITMS_MFCApp:
// �� Ŭ������ ������ ���ؼ��� ITMS_MFC.cpp�� �����Ͻʽÿ�.
//

class CITMS_MFCApp : public CWinApp
{
public:
	CITMS_MFCApp();

// �������Դϴ�.
public:
	virtual BOOL InitInstance();

// �����Դϴ�.

	DECLARE_MESSAGE_MAP()
};

extern CITMS_MFCApp theApp;