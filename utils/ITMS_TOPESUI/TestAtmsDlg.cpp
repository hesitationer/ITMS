
// TestAtmsDlg.cpp : 구현 파일
//

#include "stdafx.h"
#include "TestAtms.h"
#include "TestAtmsDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
#pragma comment(linker, "/entry:WinMainCRTStartup /subsystem:console")

// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CTestAtmsDlg 대화 상자



CTestAtmsDlg::CTestAtmsDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_TESTATMS_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CTestAtmsDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_PICTURE, m_picture);
}

BEGIN_MESSAGE_MAP(CTestAtmsDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
END_MESSAGE_MAP()


CTestAtmsDlg *pwnd;

DWORD WINAPI IPThread(LPVOID lpThreadParameter)
{	
	cli::array<unsigned char, 1>^ pImage;
	pImage = gcnew cli::array<Byte>(1920 * 1080 * 3);

	byte* pimg = new byte[1920*1080*3];
	

	int iWidth = 1920;
	int iHeight = 1080;

	Mat Frame = Mat::zeros(Size(iWidth, iHeight), CV_8UC3);

	try
	{
		while (1) {
			Sleep(1);

			try
			{
				if (!pwnd->service->GetImage(0, pImage, iWidth * iHeight * 3))
					continue;

				pin_ptr<byte> data_array_start = &pImage[0];				
				memcpy(pimg, data_array_start, iWidth * iHeight * 3);
				memcpy(Frame.data, data_array_start, iWidth * iHeight * 3);
				cv::imshow("main window", Frame);
				cv::waitKey(1);
				// -------------------------------------- HOW TO USE THE API --------------------------------------------
				pwnd->itmsres->reset();
				pwnd->itmsFncs->process(Frame, *pwnd->itmsres); // with current Frame 
														// check the object events 
				if (pwnd->itmsres->objClass.size()) {
					cout << "///////////// object events occurred /////////////\n";
					for (size_t i = 0; i < pwnd->itmsres->objClass.size(); i++) {
						cout << "objID: " << pwnd->itmsres->objClass.at(i).first << ", class: " << pwnd->itmsres->objClass.at(i).second << endl;
						cout << "Status: " << pwnd->itmsres->objStatus.at(i).second << endl;
						cout << "Speed: " << pwnd->itmsres->objSpeed.at(i) << endl;
					}
				}

				RECT r;
				pwnd->m_picture.GetClientRect(&r);

				pwnd->cimage_mfc.Create(iWidth, iHeight, 24);


				BITMAPINFO *bitInfo = (BITMAPINFO*)malloc(sizeof(BITMAPINFO) + 256 * sizeof(RGBQUAD));
				bitInfo->bmiHeader.biBitCount = 24;
				bitInfo->bmiHeader.biWidth = iWidth;
				bitInfo->bmiHeader.biHeight = -iHeight;
				bitInfo->bmiHeader.biPlanes = 1;
				bitInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
				bitInfo->bmiHeader.biCompression = BI_RGB;
				bitInfo->bmiHeader.biClrImportant = 0;
				bitInfo->bmiHeader.biClrUsed = 0;
				bitInfo->bmiHeader.biSizeImage = 0;
				bitInfo->bmiHeader.biXPelsPerMeter = 0;
				bitInfo->bmiHeader.biYPelsPerMeter = 0;

			
				// Image is bigger or smaller than into destination rectangle
				// we use stretch in full rect

			
					// destination rectangle
					int destx = 0, desty = 0;
					int destw = r.right;
					int desth = r.bottom;

					// rectangle defined on source bitmap
					// using imgWidth instead of mat_temp.cols will ignore the padding border
					int imgx = 0, imgy = 0;
					int imgWidth = iWidth;
					int imgHeight = iHeight;

					StretchDIBits(pwnd->cimage_mfc.GetDC(),
						destx, desty, destw, desth,
						imgx, imgy, imgWidth, imgHeight,
						pimg, bitInfo, DIB_RGB_COLORS, SRCCOPY);
		


				HDC dc = ::GetDC(pwnd->m_picture.m_hWnd);
				pwnd->cimage_mfc.BitBlt(dc, 0, 0);


				::ReleaseDC(pwnd->m_picture.m_hWnd, dc);

				pwnd->cimage_mfc.ReleaseDC();
				pwnd->cimage_mfc.Destroy();
			}
			catch (System::Exception^ ex)
			{
				Console::WriteLine(ex->ToString());
			}
			catch (...)
			{
			}
		}
	}
	catch (System::Exception^ ex)
	{
		Console::WriteLine(ex->ToString());
	}
	catch (...)
	{
	}

	return 0;
}


// CTestAtmsDlg 메시지 처리기

BOOL CTestAtmsDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	// ------------- HOW TO USE THE API ---------------------------------
	if (itms::loadConfig(conf)) {
		std::cout << " configarion is done !!\n\n";
	}
	else {
		std::cout << " configuration is not finished. However, default values are used instead!! Please double check the all files and values correctly (!)(!)\n";
	}

	itmsFncs = std::make_unique<itmsFunctions>(&conf); // create instance and initialize
	itmsres = std::make_unique<ITMSResult>();
	// -----------------------------------------------------------------------------------

	ConnectIpcChaanel();

	pwnd = this;
	DWORD dwThreadID;
	ThreadIP = CreateThread(NULL, 0, IPThread, NULL, 0, &dwThreadID);

	
	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CTestAtmsDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 응용 프로그램의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CTestAtmsDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CTestAtmsDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CTestAtmsDlg::ConnectIpcChaanel()
{
	try
	{
		// Create the channel.
		IpcChannel^ channel = gcnew IpcChannel;

		// Register the channel.
		System::Runtime::Remoting::Channels::ChannelServices::RegisterChannel(channel);

		// Register as client for remote object.
		System::Runtime::Remoting::WellKnownClientTypeEntry^ remoteType = gcnew
			System::Runtime::Remoting::WellKnownClientTypeEntry(
				RemoteObject::typeid, L"ipc://localhost:9090/RemoteObject.rem");
		System::Runtime::Remoting::RemotingConfiguration::RegisterWellKnownClientType(remoteType);

		// Create a message sink.
		System::String^ objectUri;
		System::Runtime::Remoting::Messaging::IMessageSink^ messageSink = channel->CreateMessageSink(
			L"ipc://localhost:9090/RemoteObject.rem", nullptr, objectUri);
		Console::WriteLine(L"The URI of the message sink is {0}.", objectUri);
		if (messageSink != nullptr)
		{
			Console::WriteLine(L"The type of the message sink is {0}.", messageSink->GetType());
		}


		// Create an instance of the remote object.
		service = gcnew RemoteObject;
	}
	catch (System::Exception^ ex)
	{
		Console::WriteLine(ex->ToString());
	}
	catch (...)
	{
	}
}
