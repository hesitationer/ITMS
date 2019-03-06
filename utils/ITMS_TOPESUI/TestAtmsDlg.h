
// TestAtmsDlg.h : ��� ����
//

#pragma once
#include "afxwin.h"


// CTestAtmsDlg ��ȭ ����
class CTestAtmsDlg : public CDialogEx
{
// �����Դϴ�.
public:
	CTestAtmsDlg(CWnd* pParent = NULL);	// ǥ�� �������Դϴ�.

// ��ȭ ���� �������Դϴ�.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_TESTATMS_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV �����Դϴ�.


// �����Դϴ�.
protected:
	HICON m_hIcon;

	// ������ �޽��� �� �Լ�
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()


public:

	HANDLE  ThreadIP;

	gcroot<RemoteObject^> service;

	CImage cimage_mfc; // to display in MFC DC

public:

	void ConnectIpcChaanel();
	CStatic m_picture;

	// ---------------------------------------------------- HOW TO USE THE API --------------------------------------------------------------
	itms::Config conf; // program configuration	
	// construct an instance 
	std::unique_ptr<itms::itmsFunctions> itmsFncs;   // itms main class		
	std::unique_ptr<itms::ITMSResult> itmsres;       // itms result structure	
	// ------------------- ITMS API -----------------------------------
};
