/********************************************************************
	// 데이터 파일 구조
	// topes mjpg structure : *.mjpg
	[ frame1 ] + [ frame2 ] + ... + [ frameN ]
	frame  = header + length + data
	header = 256 bytes : reserved
	length = 4 bytes(sizeof(DWORD)) : sizeof(data)
	data   = length bytes : real jpg image byte stream
	ex) 
	BYTE header[256];
	DWORD length;
	BYTE* data = jpgStream; // usually IJL1.0 or 1.5

	// 인덱스 파일 구조
	// topes mjpeg index structure : *.midx
	[frame count] + [frame1 offset] + [frame2 offset] + ... + [frameN offset]
	frame count   = 4 bytes : total frame count
	frameN offset = 4 bytes : frameN offset
	ex)
	DWORD frameCount = 100;
	DWORD frameOffset = 0, 20111, 40191, ...
	100 0 20111 40191 ...

*********************************************************************/
//
//////////////////////////////////////////////////////////////////////

#if!defined(AFX_MGMJPGFRAMEFILE_H__8E6B8094_545B_4DFE_8606_0BE05FE5202B__INCLUDED_)
#define AFX_MGMJPGFRAMEFILE_H__8E6B8094_545B_4DFE_8606_0BE05FE5202B__INCLUDED_


#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <vector>

#include <afx.h>

#define MJPG_HEADER_SIZE 256
#define MJPG_MAX_SIZE (1920*1080*2)



class CMjpgFile : public CFile 
{
public:
	// 생성자/소멸자
	CMjpgFile(LPCTSTR lpszFileName = NULL, UINT nOpenFlags=CFile::modeRead | CFile::shareDenyNone, int* pWidth = NULL, int* pHeight = NULL, int* pnChannels = NULL);
	virtual ~CMjpgFile();

	// 열기/닫기
	virtual BOOL Open(LPCTSTR lpszFileName, UINT nOpenFlags=CFile::modeRead | CFile::shareDenyNone, int* pWidth = NULL, int* pHeight = NULL, int* pnChannels = NULL);
  
	virtual void Close();

	
	// 프레임 이동
	void SeekFrame(DWORD iFrame);
	// 프레임 위치
	DWORD GetFramePosition() const;
	// 프레임 길이
	DWORD GetFrameLength() const;

	// 영상 정보
	int GetWidth() const;
	int GetHeight() const;
	int GetDepth() const;
	int GetChannels() const;

	// jpg 품질 설정
	void SetFrameQuality(int quality=95);

	// 프레임 읽기/쓰기
	char* ReadFrame();
	int WriteFrame(char* pBits, char* pHeader = NULL);

	// 인덱스 벡터
	std::vector<DWORD> m_midxVec;

	// 프레임 헤더
	char m_frameHeader[MJPG_HEADER_SIZE];
	// 프레임 정보
	BYTE* m_frameBuff;
	DWORD m_iFrame;
	int m_nFrameQuality; // 95

	// 프레임 사이즈
	int m_width, m_height, m_nChannels;
	// 프레임 데이터
	char* m_imageData;

};

#endif // !defined(AFX_MGMJPGFRAMEFILE_H__8E6B8094_545B_4DFE_8606_0BE05FE5202B__INCLUDED_)
