
// TestAtms.h : PROJECT_NAME ���� ���α׷��� ���� �� ��� �����Դϴ�.
//

#pragma once

#ifndef __AFXWIN_H__
	#error "PCH�� ���� �� ������ �����ϱ� ���� 'stdafx.h'�� �����մϴ�."
#endif

#include "resource.h"		// �� ��ȣ�Դϴ�.


// CTestAtmsApp:
// �� Ŭ������ ������ ���ؼ��� TestAtms.cpp�� �����Ͻʽÿ�.
//

class CTestAtmsApp : public CWinApp
{
public:
	CTestAtmsApp();

// �������Դϴ�.
public:
	virtual BOOL InitInstance();

// �����Դϴ�.

	DECLARE_MESSAGE_MAP()
};

extern CTestAtmsApp theApp;