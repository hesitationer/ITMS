
// TestAtmsDlg.h : 헤더 파일
//

#pragma once
#include "afxwin.h"


// CTestAtmsDlg 대화 상자
class CTestAtmsDlg : public CDialogEx
{
// 생성입니다.
public:
	CTestAtmsDlg(CWnd* pParent = NULL);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_TESTATMS_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
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
