
// TestAtmsDlg.cpp : ���� ����
//

#include "stdafx.h"
#include "TestAtms.h"
#include "TestAtmsDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
#pragma comment(linker, "/entry:WinMainCRTStartup /subsystem:console")

// ���� ���α׷� ������ ���Ǵ� CAboutDlg ��ȭ �����Դϴ�.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// ��ȭ ���� �������Դϴ�.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV �����Դϴ�.

// �����Դϴ�.
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


// CTestAtmsDlg ��ȭ ����



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


// CTestAtmsDlg �޽��� ó����

BOOL CTestAtmsDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// �ý��� �޴��� "����..." �޴� �׸��� �߰��մϴ�.

	// IDM_ABOUTBOX�� �ý��� ��� ������ �־�� �մϴ�.
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

	// �� ��ȭ ������ �������� �����մϴ�.  ���� ���α׷��� �� â�� ��ȭ ���ڰ� �ƴ� ��쿡��
	//  �����ӿ�ũ�� �� �۾��� �ڵ����� �����մϴ�.
	SetIcon(m_hIcon, TRUE);			// ū �������� �����մϴ�.
	SetIcon(m_hIcon, FALSE);		// ���� �������� �����մϴ�.

	// TODO: ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�.
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

	
	return TRUE;  // ��Ŀ���� ��Ʈ�ѿ� �������� ������ TRUE�� ��ȯ�մϴ�.
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

// ��ȭ ���ڿ� �ּ�ȭ ���߸� �߰��� ��� �������� �׸�����
//  �Ʒ� �ڵ尡 �ʿ��մϴ�.  ����/�� ���� ����ϴ� MFC ���� ���α׷��� ��쿡��
//  �����ӿ�ũ���� �� �۾��� �ڵ����� �����մϴ�.

void CTestAtmsDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // �׸��⸦ ���� ����̽� ���ؽ�Ʈ�Դϴ�.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Ŭ���̾�Ʈ �簢������ �������� ����� ����ϴ�.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// �������� �׸��ϴ�.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// ����ڰ� �ּ�ȭ�� â�� ���� ���ȿ� Ŀ���� ǥ�õǵ��� �ý��ۿ���
//  �� �Լ��� ȣ���մϴ�.
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
