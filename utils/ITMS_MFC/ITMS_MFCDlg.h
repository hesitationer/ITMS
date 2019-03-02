
// ITMS_MFCDlg.h : header file
//

#pragma once
#include "afxwin.h"


// CITMS_MFCDlg dialog
class CITMS_MFCDlg : public CDialogEx
{
// Construction
public:
	CITMS_MFCDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ITMS_MFC_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	CStatic m_picture;
	afx_msg void OnDestroy();
	afx_msg void OnTimer(UINT_PTR nIDEvent);

	VideoCapture *capture;
	Mat mat_frame;
	CImage cimage_mfc;
};
