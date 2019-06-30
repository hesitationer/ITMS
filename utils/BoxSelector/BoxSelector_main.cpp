#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <windows.h>

#include "../../src/utils/BoxSeletor/cv_ext/init_box_selector.hpp"

#define _sk_Memory_Leakag_Detector
#ifdef _sk_Memory_Leakag_Detector
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <vld.h>

#if _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define malloc(s) _malloc_dbg(s, _NORMAL_BLOCK, __FILE__, __LINE__)
#endif
#endif

using namespace std;
using namespace cv;


cv::Mat show_img;
// Tracker results
//Rect result;
cv::Rect showRect;
bool select_flag = false;
bool selected = false;
bool _bROI_Selected = false;
Point origin;
bool SILENT = false;

int _frameIdx = 0;
bool _isPaused = false;
bool _isStep = false;
bool _exit0 = false;
bool _hasInitBox = false;
bool _isTrackerInitialzed = false;
bool _updateAtPos = false; // delete a point
bool _targetOnFrame = false;
std::vector<double> fourPts;

cv::Rect_<double> _boundingBox;
std::string windowTitle = "BoxSelector";
std::string _windowTitle(windowTitle);
bool debugShowImage = true;

bool update(cv::Mat &_image);


int main(int argc, char* argv[]) {
#ifdef _sk_Memory_Leakag_Detector
#if _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif	
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;
	

	string filename = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/20180912_112338_cam_0.avi"/* 20180911_113611_cam_0 20180912_192157_cam_0.avi*//*_paras.device*/;
	VideoCapture cam(/*0*/filename); //webcam : 0
	
		
	double duration = 0;

	/*
	std::ofstream resultsFile;
	resultsFile.open(seq_name + ".txt");*/

	// Write Results
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);

	// Frame counter
	int nFrames = 0;
	float fscaleFactor = 0.5;

	//std::string imgFinalPath;	
	
	while (1)
	{	
		if (!_isPaused || _frameIdx == 0 || _isStep)
		{
			cam >> show_img;

			if (show_img.empty())						
				break;
			cv::resize(show_img, show_img, Size(), fscaleFactor, fscaleFactor);
			++_frameIdx;
		}

		if (show_img.empty())
		{
			std::cout << "End of File" << std::endl;
			break;
		}
		

		//set Rect
		//setMouseCallback(window_name, onMouse, 0);
		if(!update(show_img))
			break;

		resultsFile << showRect.x << "," << showRect.y << "," << showRect.width << "," << showRect.height << endl;

		char key = (char)waitKey(1); //delay N millis, usually long enough to display and capture input

		switch (key) {
		case 'q':
		case 'Q':
		case 27: //escape key

			if (resultsFile.is_open())
				resultsFile.close();
#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
			_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	
			return 0;
		default:
			break;
		}
	}	
	if (resultsFile.is_open())
		resultsFile.close();

#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	

	return 0;

}

 
bool update(cv::Mat &_image)
{
	//if (!_isPaused || _frameIdx == 0 || _isStep)
	//{
	//	/*_cap >> _image;*/
	 
	//	if (_image.empty())			return false;
	//	// resize comes here
	//	++_frameIdx;
	//}

	if (!_isTrackerInitialzed)
	{
		if (!_hasInitBox)
		{
			Rect box;

			if (!InitBoxSelector::selectBox(_image, box))
			return false;

			_boundingBox = Rect_<double>(static_cast<double>(box.x),
			static_cast<double>(box.y),
			static_cast<double>(box.width),
			static_cast<double>(box.height));

			_hasInitBox = true;
		}
		// check the validity
		_targetOnFrame = (_boundingBox.x < 0 || _boundingBox.y<0 || (_boundingBox.x + _boundingBox.width)>_image.cols || (_boundingBox.y + _boundingBox.height)>_image.rows)? false:true;
		if(_targetOnFrame)
			_isTrackerInitialzed = true;
	}
	else if (_isTrackerInitialzed && (!_isPaused || _isStep))
	{
		_isStep = false;

		if (_updateAtPos) // 
		{
			Rect box;

			if (!InitBoxSelector::selectBox(_image, box))
				return false;

			_boundingBox = Rect_<double>(static_cast<double>(box.x),
				static_cast<double>(box.y),
				static_cast<double>(box.width),
				static_cast<double>(box.height));

			_updateAtPos = false;

			std::cout << "UpdateAt_: " << _boundingBox << std::endl;	
			_targetOnFrame = (_boundingBox.x < 0 || _boundingBox.y<0 || (_boundingBox.x + _boundingBox.width)>_image.cols || (_boundingBox.y + _boundingBox.height)>_image.rows) ? false : true;
			if (!_targetOnFrame)
				std::cout << "Target not found!" << std::endl;

		}
		else
		{		
			_targetOnFrame = (_boundingBox.x < 0 || _boundingBox.y<0 || (_boundingBox.x + _boundingBox.width)>_image.cols || (_boundingBox.y + _boundingBox.height)>_image.rows) ? false : true;
		}
	}	
	if (debugShowImage)
	{
		Mat hudImage;
		_image.copyTo(hudImage);
		rectangle(hudImage, _boundingBox, Scalar(0, 0, 255), 2);
		Point_<double> center;
		center.x = _boundingBox.x + _boundingBox.width / 2;
		center.y = _boundingBox.y + _boundingBox.height / 2;
		circle(hudImage, center, 3, Scalar(0, 0, 255), 2);

		stringstream ss;
		ss.str("");
		ss.clear();
		ss << "#" << _frameIdx;
		putText(hudImage, ss.str(), Point(hudImage.cols - 60, 20), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0));		

		if (!_targetOnFrame)
		{
			cv::Point_<double> tl = _boundingBox.tl();
			cv::Point_<double> br = _boundingBox.br();

			line(hudImage, tl, br, Scalar(0, 0, 255));
			line(hudImage, cv::Point_<double>(tl.x, br.y),
			cv::Point_<double>(br.x, tl.y), Scalar(0, 0, 255));
		}
		imshow(_windowTitle.c_str(), hudImage);		
	}

	char c = (char)waitKey(10);

	if (c == 27)
	{
		_exit0 = true;
		return false;
	}

	switch (c)
	{
		case 'p':
		_isPaused = !_isPaused;
		break;
		case 'c':
		_isStep = true;
		break;
		case 'r':
		_hasInitBox = false;
		_isTrackerInitialzed = false;
		break;
		case 't':
		_updateAtPos = true;
		break;
		default:
		;
	}
	return true;
}

	