
// ITMS_MFCDlg.cpp : implementation file
//

#include "stdafx.h"
#include "ITMS_MFC.h"
#include "ITMS_MFCDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CITMS_MFCDlg dialog



CITMS_MFCDlg::CITMS_MFCDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_ITMS_MFC_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CITMS_MFCDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_PICTURE, m_picture);
}

BEGIN_MESSAGE_MAP(CITMS_MFCDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_DESTROY()
	ON_WM_TIMER()
END_MESSAGE_MAP()


// CITMS_MFCDlg message handlers

BOOL CITMS_MFCDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here	

	if (itms::loadConfig(conf)) {
		std::cout << " configarion is done !!\n\n";
	}
	else {
		std::cout << " configuration is not finished. However, default values are used instead!! Please double check the all files and values correctly (!)(!)\n";
	}
	// ------------- HOW TO USE THE API ---------------------------------
	itmsFncs = std::make_unique<itmsFunctions>(&conf); // create instance and initialize
	itmsres = std::make_unique<ITMSResult>();

	//capture = new VideoCapture(conf.VideoPath);
	bool b = capture.open(conf.VideoPath);
	if (!capture.isOpened())
	{
		MessageBox(_T("캠을 열수 없습니다. \n"));
	}

	//웹캠 크기를  320x240으로 지정    
	capture.set(CAP_PROP_FRAME_WIDTH, 320);
	capture.set(CAP_PROP_FRAME_HEIGHT, 240);

	SetTimer(1000, 30, NULL);   //	webcam 을 이용할 경우

	int max_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);

	if (max_frames< 2) {
		std::cout << "error: video file must have at least two frames" << std::endl;
		// it may be necessary to change or remove this line if not using Windows
		return(0);
	}
	/* Event Notice */
	std::cout << "Press 'ESC' to quit..." << std::endl;

	// video information
	int fps = 15;
	bool hasFile = true;
	if (hasFile)
	{
		fps = int(capture.get(CAP_PROP_FPS));
		conf.fps = fps;
		cout << "Video FPS: " << fps << endl;
	}
	// set the start point 
	int m_startFrame = 0;	 // 240
	int frameCount = m_startFrame + 1;

	capture.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);

	capture.read(imgFrame1);
	capture.read(imgFrame2);
	if (imgFrame1.empty() || imgFrame2.empty())
		return 0;


	return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CITMS_MFCDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CITMS_MFCDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CITMS_MFCDlg::OnDestroy()
{

	CDialogEx::OnDestroy();

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	if (capture.isOpened())
		capture.release();
}


void CITMS_MFCDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.
	//mat_frame가 입력 이미지입니다. 
	capture.read(imgFrame2);
	//capture.read(imgFrame2);  // if we use this option, we need to adjust the computation of speeed in Config file
	
	if (imgFrame2.empty()) {		
		EndDialog(IDOK);	// close the program or CDialog::OnTimer(nIDEvent) if you want to stay
		
		return;
	}	

	// 이곳에 OpenCV 등 API 함수들을 적용합니다.
	// 필요시, 그레이스케일 이미지로 변환합니다.
	//cvtColor(imgFrame2, imgFrame2, COLOR_BGR2GRAY);
	// -------------------------------------- HOW TO USE THE API --------------------------------------------
	itmsres->reset();
	itmsFncs->process(imgFrame2, *itmsres); // with current Frame 
											// check the object events 
	if (itmsres->objClass.size()) {
		cout << "///////////// object events occurred /////////////\n";
		for (size_t i = 0; i < itmsres->objClass.size(); i++) {
			cout << "objID: " << itmsres->objClass.at(i).first << ", class: " << itmsres->objClass.at(i).second << endl;
			cout << "Status: " << itmsres->objStatus.at(i).second << endl;
			cout << "Speed: " << itmsres->objSpeed.at(i) << endl;
		}
	}
	// -------------------------------------------------------------------------------------------------------
	// now we prepare for the next iteration
	//cv::resize(imgFrame2, imgFrame2, cv::Size(), conf.scaleFactor, conf.scaleFactor); // if you resize the output
	imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is
	// -------------------------------------------------------------------------------------------------------

	//화면에 보여주기 위한 처리입니다.
	int bpp = 8 * imgFrame2.elemSize();
	assert((bpp == 8 || bpp == 24 || bpp == 32));

	int padding = 0;
	//32 bit image is always DWORD aligned because each pixel requires 4 bytes
	if (bpp < 32)
		padding = 4 - (imgFrame2.cols % 4);

	if (padding == 4)
		padding = 0;

	int border = 0;
	//32 bit image is always DWORD aligned because each pixel requires 4 bytes
	if (bpp < 32)
	{
		border = 4 - (imgFrame2.cols % 4);
	}



	Mat mat_temp;
	if (border > 0 || imgFrame2.isContinuous() == false)
	{
		// Adding needed columns on the right (max 3 px)
		cv::copyMakeBorder(imgFrame2, mat_temp, 0, 0, 0, border, cv::BORDER_CONSTANT, 0);
	}
	else
	{
		mat_temp = imgFrame2;
	}


	RECT r;
	m_picture.GetClientRect(&r);
	cv::Size winSize(r.right, r.bottom);

	cimage_mfc.Create(winSize.width, winSize.height, 24);


	BITMAPINFO *bitInfo = (BITMAPINFO*)malloc(sizeof(BITMAPINFO) + 256 * sizeof(RGBQUAD));
	bitInfo->bmiHeader.biBitCount = bpp;
	bitInfo->bmiHeader.biWidth = mat_temp.cols;
	bitInfo->bmiHeader.biHeight = -mat_temp.rows;
	bitInfo->bmiHeader.biPlanes = 1;
	bitInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitInfo->bmiHeader.biCompression = BI_RGB;
	bitInfo->bmiHeader.biClrImportant = 0;
	bitInfo->bmiHeader.biClrUsed = 0;
	bitInfo->bmiHeader.biSizeImage = 0;
	bitInfo->bmiHeader.biXPelsPerMeter = 0;
	bitInfo->bmiHeader.biYPelsPerMeter = 0;


	//그레이스케일 인경우 팔레트가 필요
	if (bpp == 8)
	{
		RGBQUAD* palette = bitInfo->bmiColors;
		for (int i = 0; i < 256; i++)
		{
			palette[i].rgbBlue = palette[i].rgbGreen = palette[i].rgbRed = (BYTE)i;
			palette[i].rgbReserved = 0;
		}
	}


	// Image is bigger or smaller than into destination rectangle
	// we use stretch in full rect

	if (mat_temp.cols == winSize.width  && mat_temp.rows == winSize.height)
	{
		// source and destination have same size
		// transfer memory block
		// NOTE: the padding border will be shown here. Anyway it will be max 3px width

		SetDIBitsToDevice(cimage_mfc.GetDC(),
			//destination rectangle
			0, 0, winSize.width, winSize.height,
			0, 0, 0, mat_temp.rows,
			mat_temp.data, bitInfo, DIB_RGB_COLORS);
	}
	else
	{
		// destination rectangle
		int destx = 0, desty = 0;
		int destw = winSize.width;
		int desth = winSize.height;

		// rectangle defined on source bitmap
		// using imgWidth instead of mat_temp.cols will ignore the padding border
		int imgx = 0, imgy = 0;
		int imgWidth = mat_temp.cols - border;
		int imgHeight = mat_temp.rows;

		StretchDIBits(cimage_mfc.GetDC(),
			destx, desty, destw, desth,
			imgx, imgy, imgWidth, imgHeight,
			mat_temp.data, bitInfo, DIB_RGB_COLORS, SRCCOPY);
	}


	HDC dc = ::GetDC(m_picture.m_hWnd);
	cimage_mfc.BitBlt(dc, 0, 0);


	::ReleaseDC(m_picture.m_hWnd, dc);

	cimage_mfc.ReleaseDC();
	cimage_mfc.Destroy();


	CDialogEx::OnTimer(nIDEvent);
}
