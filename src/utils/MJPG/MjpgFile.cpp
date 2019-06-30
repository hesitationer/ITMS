//
//////////////////////////////////////////////////////////////////////


//#include <afx.h>  // CFile class by sangkny MFC related
//#include "afx.h"
#include "MjpgFile.h"

#include <fstream>
#include <iostream>


#include "ijl.h"
#pragma comment( lib, "ijl.lib" )

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

// sangkny definition 

//bool MgIsFileExists(CString fileName);
//bool MgIsFileExists(CString fileName) {
//  std::ifstream infile(fileName);
//  return infile.good();
//}
inline BOOL MgIsFileExists(LPCTSTR szFilePath)
{
  DWORD dwAttr = GetFileAttributes(szFilePath);
  if (dwAttr == 0xffffffff)
    return FALSE;
  else
    return TRUE;
}
//CString MgGetFileExt(CString fileName);
//CString MgGetFileExt(CString fileName) {
//  int pos = fileName.ReverseFind('.');
//  int len = fileName.GetLength();
//  CString strResult = fileName.Right(len-(pos));
//
//  return strResult;
//}
inline CString MgGetFileExt(LPCTSTR pszFullPath)
{
  TCHAR ext[_MAX_EXT];
  _tsplitpath(pszFullPath, NULL, NULL, NULL, ext);
  return ext;
}

//LONGLONG MgFileLength(LPCTSTR lpszFileName);
//ULONGLONG MgFileLength(LPCTSTR lpszFileName) {
//  ULONGLONG filelength = 0;
//  CFile* pFile = NULL;
//  // Constructing a CFile object with this override may throw
//  // a CFile exception, and won't throw any other exceptions.
//  // Calling CString::Format() may throw a CMemoryException,
//  // so we have a catch block for such exceptions, too. Any
//  // other exception types this function throws will be
//  // routed to the calling function.
//  try
//  {
//    pFile = new CFile(lpszFileName,
//      CFile::modeRead | CFile::shareDenyNone);
//    filelength = pFile->GetLength();
//
//  }
//  catch (CFileException* pEx)
//  {
//    // Simply show an error message to the user.
//    pEx->ReportError();
//    pEx->Delete();
//  }
//  catch (CMemoryException* pEx)
//  {
//    pEx->ReportError();
//    pEx->Delete();
//    // We can't recover from this memory exception, so we'll
//    // just terminate the app without any cleanup. Normally,
//    // an application should do everything it possibly can to
//    // clean up properly and _not_ call AfxAbort().
//    AfxAbort();
//  }
//
//  // If an exception occurs in the CFile constructor,
//  // the language will free the memory allocated by new
//  // and will not complete the assignment to pFile.
//  // Thus, our clean-up code needs to test for NULL.
//  if (pFile != NULL)
//  {
//    pFile->Close();
//    delete pFile;
//  }
//  return filelength;
//}
inline CString MgFormatErrorString(DWORD dwError = NO_ERROR)
{
  TCHAR buffer[4096];
  int nErrNum;
  nErrNum = (dwError == NO_ERROR) ? GetLastError() : dwError;

  LPSTR lpMsgBuf;

  FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, nErrNum, LANG_NEUTRAL, (LPTSTR)&lpMsgBuf, 0, NULL);
  _stprintf(buffer, _T("Error[%d] Reason: %s\n"), nErrNum, lpMsgBuf);

  LocalFree(lpMsgBuf);
  return CString(buffer);
}

inline ULONGLONG MgFileLength(LPCTSTR lpszFileName)
{
  HANDLE hFile = INVALID_HANDLE_VALUE;
  // attempt file creation
  hFile = ::CreateFile(lpszFileName, GENERIC_READ, FILE_SHARE_READ, NULL,
    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile == INVALID_HANDLE_VALUE) {
    TRACE(MgFormatErrorString());
    throw lpszFileName;
  }

  ULARGE_INTEGER liSize;
  liSize.LowPart = ::GetFileSize(hFile, &liSize.HighPart);
  if (liSize.LowPart == INVALID_FILE_SIZE)
    if (::GetLastError() != NO_ERROR)
      throw lpszFileName;

  if (hFile != INVALID_HANDLE_VALUE)
    ::CloseHandle(hFile);

  return liSize.QuadPart;
}

// jpg 압축
int Compress(BYTE* src, int width, int height, int channels, int quality, BYTE* dst)
{
	JPEG_CORE_PROPERTIES jcprops;

	if (ijlInit(&jcprops) != IJL_OK)
	{
		ijlFree(&jcprops);
		return 0;
	}

	jcprops.DIBWidth = width;
	jcprops.DIBHeight = height; // -height;
	jcprops.JPGWidth = width;
	jcprops.JPGHeight = height;
	jcprops.DIBBytes = (BYTE*)src;
	jcprops.DIBChannels = channels;
	jcprops.JPGChannels = channels;
	jcprops.DIBPadBytes = 0; //width%4;

	if (channels == 3)
	{
		jcprops.DIBColor = IJL_BGR;
		jcprops.JPGColor = IJL_YCBCR;
		jcprops.JPGSubsampling = IJL_411;
		jcprops.DIBSubsampling = (IJL_DIBSUBSAMPLING)0;
	}
	else
	{
		jcprops.DIBColor = IJL_G;
		jcprops.JPGColor = IJL_G;
		jcprops.JPGSubsampling = (IJL_JPGSUBSAMPLING)0;
		jcprops.DIBSubsampling = (IJL_DIBSUBSAMPLING)0;
	}

	int size = width*height*channels;
	jcprops.JPGSizeBytes = size;
	jcprops.JPGBytes = dst;
	jcprops.jquality = quality;

	IJLERR code = ijlWrite(&jcprops, IJL_JBUFF_WRITEWHOLEIMAGE);
	if (code != IJL_OK)
	{
		//const char* errStr = ijlErrorStr(code);
		//TRACE("!CMgIJL::Compress:ijlWrite-%s\n", errStr);
		ijlFree(&jcprops);
		return 0;
	}

	if (ijlFree(&jcprops) != IJL_OK)
	{
		return 0;
	}

	return jcprops.JPGSizeBytes;
}

// jpg 압축 풀기
BOOL Decompress(BYTE* src, int size, BYTE* dst, int& width, int& height, int& channels, int flip = 0)
{
	JPEG_CORE_PROPERTIES jcprops;

	if (ijlInit(&jcprops) != IJL_OK)
	{
		ijlFree(&jcprops);
		return FALSE;
	}

	jcprops.JPGBytes = (BYTE*)src;
	jcprops.JPGSizeBytes = size;
	jcprops.jquality = 100;

	if (ijlRead(&jcprops, IJL_JBUFF_READPARAMS) != IJL_OK)
	{
		ijlFree(&jcprops);
		return FALSE;
	}

	width = jcprops.JPGWidth;
	height = jcprops.JPGHeight;
	channels = jcprops.JPGChannels;
	if (dst == NULL)
	{
		ijlFree(&jcprops);
		return TRUE;
	}

	jcprops.DIBWidth = width;
	//jcprops.DIBHeight = height; // -height(flip image)
	jcprops.DIBHeight = flip ? -height : height;; // -height(flip image)
	jcprops.DIBChannels = channels;
	jcprops.DIBBytes = dst;
	jcprops.DIBPadBytes = 0; //width%4;

	if (jcprops.JPGChannels == 3)
	{
		jcprops.DIBColor = IJL_BGR;
		jcprops.JPGColor = IJL_YCBCR;
		jcprops.JPGSubsampling = IJL_411;
		jcprops.DIBSubsampling = (IJL_DIBSUBSAMPLING)0;
	}
	else
	{
		jcprops.DIBColor = IJL_G;
		jcprops.JPGColor = IJL_G;
		jcprops.JPGSubsampling = (IJL_JPGSUBSAMPLING)0;
		jcprops.DIBSubsampling = (IJL_DIBSUBSAMPLING)0;
	}

	if (ijlRead(&jcprops, IJL_JBUFF_READWHOLEIMAGE) != IJL_OK)
	{
		ijlFree(&jcprops);
		return FALSE;
	}

	if (ijlFree(&jcprops) != IJL_OK)
		return FALSE;

	return TRUE;
}


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CMjpgFile::CMjpgFile(LPCTSTR lpszFileName, UINT nOpenFlags, int* pWidth, int* pHeight, int* pnChannels)
	: m_frameBuff(NULL), m_iFrame(0)
{
	m_imageData = NULL;
	if (lpszFileName)
	{
		Open(lpszFileName, nOpenFlags, pWidth, pHeight, pnChannels);
	}
}

CMjpgFile::~CMjpgFile()
{
	Close();
}

BOOL CMjpgFile::Open(LPCTSTR lpszFileName, UINT nOpenFlags, int* pWidth, int* pHeight, int* pnChannels)
{
	try
	{
		if (pWidth)
		{
			m_width = *pWidth;
		}
		if (pHeight)
		{
			m_height = *pHeight;
		}
		if (pnChannels)
		{
			m_nChannels = *pnChannels;
		}

		ZeroMemory(m_frameHeader, MJPG_HEADER_SIZE);
		m_frameBuff = new BYTE[MJPG_MAX_SIZE];
		m_midxVec.clear();
		m_nFrameQuality = 95;

		if(!CFile::Open(lpszFileName, nOpenFlags)) { return FALSE; }

		CString strMidx = GetFilePath();
		CString strExt = MgGetFileExt(strMidx);
		strMidx.Replace(strExt, _T(".midx"));

		if (nOpenFlags & CFile::modeWrite) {
			DeleteFile(strMidx);
			return TRUE;
		}
	
		if (MgIsFileExists(strMidx))
		{
			TRACE("CMgCVMJPG::Read %s\n", strMidx);
			CFile idxFile(strMidx, CFile::modeRead|CFile::shareDenyNone);

			DWORD nFrame = 0;
			idxFile.Read(&nFrame, sizeof(DWORD));
			m_midxVec.reserve(nFrame);
			DWORD dwIndex=0;
			for (size_t i=0; i<nFrame; i++) {
				idxFile.Read(&dwIndex, sizeof(DWORD));
				m_midxVec.push_back(dwIndex);
			}
			idxFile.Close();
		} else {
			/////////////////////////////////////////////////
			// read index info
			LONGLONG fileLength=0;
			fileLength = MgFileLength(lpszFileName);
			TRACE("CMgCVMJPG:FileLength=%d\n", fileLength);
			CFile file(lpszFileName, CFile::modeRead|CFile::shareDenyNone);

			LONGLONG nextFramePos = 0;
			m_midxVec.push_back((DWORD)nextFramePos); // first frame
			DWORD nLength=0;
			DWORD nFrame = 0;
		
			while (true)
			{
				file.Seek(MJPG_HEADER_SIZE, CFile::current);
				UINT nRead = file.Read(&nLength, sizeof(DWORD));
				if (nRead!=sizeof(DWORD)) {
					TRACE("CMgCVMJPG::file.Read=%d\n", nRead);
					break;
				}

				nextFramePos += (MJPG_HEADER_SIZE + 4 + nLength);
			
				if (nextFramePos >= UINT_MAX) {
					TRACE("CMgCVMJPG::if (nextFramePos(%d) >= UINT_MAX(%d)\n", nextFramePos, UINT_MAX);
					break;
				}

				m_midxVec.push_back((DWORD)nextFramePos); // next frame
				LONG lActual = (LONG)file.Seek(nLength, CFile::current);

			}
			file.Close();
			m_midxVec.erase(m_midxVec.end()-1);

			// write midx pszFile
			CFile idxFile(strMidx, CFile::modeCreate|CFile::modeWrite);
			nFrame = m_midxVec.size();
			idxFile.Write(&nFrame, sizeof(DWORD));
			DWORD nFrameSize=0;
			for (size_t i=0; i<nFrame; i++)
			{
				nFrameSize = m_midxVec[i];
				idxFile.Write(&nFrameSize, sizeof(DWORD));
			}
			idxFile.Close();

		}

		DWORD nFrameSize;
		CFile::Read(m_frameHeader, MJPG_HEADER_SIZE);
		CFile::Read(&nFrameSize, sizeof(DWORD));
		CFile::Read(m_frameBuff, nFrameSize);

		Decompress(m_frameBuff, nFrameSize, NULL, m_width, m_height, m_nChannels);
		m_imageData = new char[m_width * m_height * 3 + 64];

		if (pWidth)
		{
			*pWidth = m_width;
		}
		if (pHeight)
		{
			*pHeight = m_height;
		}
		if (pnChannels)
		{
			*pnChannels = m_nChannels;
		}
		SeekFrame(0);
	}
	catch (CException* e)
	{
		char buff[256];
		e->GetErrorMessage((LPTSTR)buff, 256);
		TRACE("%s\n",buff);
		e->Delete();
	} catch(...) {
		TRACE("catch(...) - CMjpgFile::Open(%s)\n", lpszFileName);
	}

	return TRUE;
}

void CMjpgFile::Close()
{
	if (m_frameBuff==NULL) {
		return;
	} else {
		delete [] m_frameBuff;
		m_frameBuff = NULL;
	}

	if (m_imageData)
	{
		delete[] m_imageData;
		m_imageData = NULL;
	}

	TRACE("%s\n", CFile::GetFilePath());
	m_midxVec.clear();
	try {
		CFile::Flush();
		CFile::Close();
	} catch (CFileException* e) {
		TRACE("CFileException - CMjpgFile::Close()\n");
		e->Delete();
	}

}

void CMjpgFile::SeekFrame(DWORD iFrame)
{
	m_iFrame = iFrame;
	CFile::Seek((LONG)m_midxVec[iFrame], CFile::begin);
}

DWORD CMjpgFile::GetFramePosition() const
{
	return m_iFrame;
}

DWORD CMjpgFile::GetFrameLength() const
{
	return m_midxVec.size();
}

void CMjpgFile::SetFrameQuality(int quality)
{
	m_nFrameQuality = quality;
}
int CMjpgFile::GetWidth() const
{
	return m_width;
}
int CMjpgFile::GetHeight() const
{
	return m_height;
}
int CMjpgFile::GetDepth() const
{
	return 8;
}
int CMjpgFile::GetChannels() const
{
	return m_nChannels;
}

char* CMjpgFile::ReadFrame()
{
	DWORD nFrameSize;
	try
	{
		CFile::Read(m_frameHeader, MJPG_HEADER_SIZE);
		CFile::Read(&nFrameSize, sizeof(DWORD));
		CFile::Read(m_frameBuff, nFrameSize);
		m_iFrame++;
	}
	catch (CFileException* e)
	{
		char buff[256];
		e->GetErrorMessage((LPTSTR)buff, 256);
		TRACE("%s\n", buff);
		e->Delete();
	}

	Decompress(m_frameBuff, nFrameSize, (unsigned char*)m_imageData, m_width, m_height, m_nChannels);
	return m_imageData;
}

int CMjpgFile::WriteFrame(char* pBits, char* pHeader)
{
	DWORD dwComp = (DWORD)Compress((BYTE*)pBits, m_width, m_height, 3, m_nFrameQuality, m_frameBuff);
	if (dwComp == 0) {
		//TRACE("m_ijl.Compress:%dx%dx%d,%d\n", pImage->width, pImage->height, pImage->nChannels, m_nFrameQuality);
		//return 0;
		return 1;
	}

	if ((GetPosition() + MJPG_HEADER_SIZE + 4 + dwComp) > ULONG_MAX)
	{
		CString fileName = GetFilePath();
		if (fileName[fileName.GetLength() - 6] == ']')
		{
			CString strIndex = fileName.Mid(fileName.GetLength() - 8, 2);
			int nIndex = atoi((char *)(LPCTSTR)strIndex);
			CString strOldIndex;
			strOldIndex.Format(_T("[%02d]"), nIndex);
			nIndex += 1;
			CString strNewIndex;
			strNewIndex.Format(_T("[%02d]"), nIndex);
			fileName.Replace(strOldIndex, strNewIndex);
		}
		else {
			CString strExt = fileName.Right(5);
			fileName.Replace(strExt, _T("[01].mjpg"));
		}
		CFile::Flush();
		CFile::Close();

		CFile::Open(fileName, CFile::modeWrite | CFile::modeCreate | CFile::typeBinary);
	}

	if (pHeader == NULL) {
		CFile::Seek(MJPG_HEADER_SIZE, CFile::current);
	}
	else {
		CFile::Write(pHeader, MJPG_HEADER_SIZE);
	}
	CFile::Write(&dwComp, sizeof(DWORD));
	CFile::Write(m_frameBuff, dwComp);

	++m_iFrame;
	return (MJPG_HEADER_SIZE + sizeof(DWORD) + dwComp);
}
