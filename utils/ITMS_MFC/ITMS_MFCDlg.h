
// ITMS_MFCDlg.h : ��� ����
//

#pragma once
#include "afxwin.h"


// CITMS_MFCDlg ��ȭ ����
class CITMS_MFCDlg : public CDialogEx
{
// �����Դϴ�.
public:
	CITMS_MFCDlg(CWnd* pParent = NULL);	// ǥ�� �������Դϴ�.

// ��ȭ ���� �������Դϴ�.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ITMS_MFC_DIALOG };
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
	CStatic m_picture;

	/*VideoCapture *capture;
	Mat mat_frame;
	CImage cimage_mfc;*/
};
