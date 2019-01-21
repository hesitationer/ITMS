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
#include <direct.h> // _getcwd
#include <stdlib.h>
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

int _frameIdx = 0;
bool _isPaused = false;
bool _isStep = false;
bool _exit0 = false;
bool _hasInitBox = false;
bool _isTrackerInitialzed = false;
bool _updateAtPos = false; // delete a point
bool _targetOnFrame = false;
std::vector<Point> _boundingBoxPts;                 // a box bounding points
std::vector<std::vector<cv::Point>> _boundingBoxes; // bunch of boxes for specfic region boundaries

std::string windowTitle = "Road Bounding Point Selector";
std::string _windowTitle(windowTitle);
bool debugShowImage = true;

bool update(cv::Mat &_image);

inline bool existFileTest(const std::string& name);
// parameters
namespace Config
{
	string run_time_dir;
	string video_path;	
}

bool loadConfig(void)
{
	bool fexist = existFileTest("./config/RoadPoint_Config.xml");
	
	//char currentPath[_MAX_PATH];
	//_getcwd(currentPath, _MAX_PATH); // current working dir
	//printf("%s\n", currentPath);	

	CvFileStorage* fs = cvOpenFileStorage("./config/RoadPoint_Config.xml", 0, CV_STORAGE_READ);
	const char *VP = cvReadStringByName(fs, NULL, "VideoPath", NULL);
	//strcpy(Config::VideoPath, VP);	
	Config::video_path = (std::string)VP;	
	cvReleaseFileStorage(&fs);

	return fexist;
}

int main(int argc, char* argv[]) {
#ifdef _sk_Memory_Leakag_Detector
#if _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif	
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	loadConfig(); // get the exe location

	string filename = Config::video_path;//"D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/20180912_112338_cam_0.avi"/* 20180911_113611_cam_0 20180912_192157_cam_0.avi*//*_paras.device*/;
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
	float fscaleFactor = 1.0;

	//std::string imgFinalPath;	
	// print help in the console
	std::cout << "Switch pause with 'p'" << std::endl;
	std::cout << "Step frame with 'c'" << std::endl;
	std::cout << "Select new target with 'd'" << std::endl;
	std::cout << "Add more box points to the previous box(es)  'a'" << std::endl;
	std::cout << "Quit with 'ESC'or 'q'/'Q'" << std::endl;
	
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

		

		char key = (char)waitKey(1); //delay N millis, usually long enough to display and capture input

		switch (key) {
		case 'q':
		case 'Q':
		case 27: //escape key
			if (resultsFile.is_open()) {
				for (int ib = 0; ib<_boundingBoxes.size(); ib++) {
					std::vector<cv::Point> db_boxPts = _boundingBoxes.at(ib);
					for (int i = 0; i < db_boxPts.size(); i++) {						
						resultsFile << db_boxPts.at(i).x << "," << db_boxPts.at(i).y << endl;
					}
					resultsFile << endl;
				}
				resultsFile.close();
			}
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
		
	if (resultsFile.is_open()){
		for (int ib = 0; ib<_boundingBoxes.size(); ib++) {
			std::vector<cv::Point> db_boxPts = _boundingBoxes.at(ib);
			for (int i = 0; i < db_boxPts.size(); i++) {
				resultsFile << db_boxPts.at(i).x << "," << db_boxPts.at(i).y << endl;
			}
			resultsFile << endl;
		}
		resultsFile.close();
	}

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
			_boundingBoxes.clear();
			_boundingBoxPts.clear(); // clear the previous Pts
			std::vector<Point> boxPoints; // road points

			if (!InitPointSelector::selectPoints(_image, boxPoints))
			return false;

			_boundingBoxPts = boxPoints;

			_hasInitBox = true;
		}
		// check the validity		
		bool ptValid = true;
		for(int i=0;i< _boundingBoxPts.size();i++){
			ptValid = (_boundingBoxPts.at(i).x < 0 || _boundingBoxPts.at(i).y<0 || (_boundingBoxPts.at(i).x >=_image.cols) || (_boundingBoxPts.at(i).y >=_image.rows))? false:true;
			if(ptValid) break;
		}
		_targetOnFrame = ptValid;
		if(_targetOnFrame){
			_isTrackerInitialzed = true;
			_boundingBoxes.push_back(_boundingBoxPts); // insert a confirmed points
		}
	}
	else if (_isTrackerInitialzed && (!_isPaused || _isStep))
	{
		_isStep = false;
		// need to check the availability of _boundingBoxes

		if (_updateAtPos) //  add a box points more 
		{
			// clear the right previous points
			_boundingBoxPts.clear();
			// draw the previous points and its line
			cv::Mat debugImg;
			_image.copyTo(debugImg);
			for(int i= 0; i< _boundingBoxes.size(); i++)
			for(int ip=0; ip< _boundingBoxes.at(i).size(); ip++)
				line(debugImg, _boundingBoxes.at(i).at(ip), _boundingBoxes.at(i).at((ip+1)% _boundingBoxes.at(i).size()),Scalar(0, 255, 255), 2,1);

			std::vector<Point> boxPoints; // road points

			if (!InitPointSelector::selectPoints(debugImg, boxPoints))
				return false;

			_boundingBoxPts = boxPoints;

			_updateAtPos = false;

			std::cout << "Added a box At: " << boxPoints << std::endl;
			// check the validity		
			bool ptValid = true;
			for (int i = 0; i< _boundingBoxPts.size(); i++) {
				ptValid = (_boundingBoxPts.at(i).x < 0 || _boundingBoxPts.at(i).y<0 || (_boundingBoxPts.at(i).x >= _image.cols) || (_boundingBoxPts.at(i).y >= _image.rows)) ? false : true;
				if (ptValid) break;
			}
			if(_targetOnFrame = ptValid)
				_boundingBoxes.push_back(_boundingBoxPts); // insert a confirmed points
		}
		else
		{		
			// check the validity		
			bool ptValid = true;
			for (int i = 0; i< _boundingBoxPts.size(); i++) {
				ptValid = (_boundingBoxPts.at(i).x < 0 || _boundingBoxPts.at(i).y<0 || (_boundingBoxPts.at(i).x >= _image.cols) || (_boundingBoxPts.at(i).y >= _image.rows)) ? false : true;
				if (ptValid) break;
			}
			_targetOnFrame = ptValid;
		}
	}	
	if (debugShowImage)
	{
		Mat hudImage;
		_image.copyTo(hudImage);
		for(int ib=0; ib<_boundingBoxes.size(); ib++){
			std::vector<cv::Point> db_boxPts = _boundingBoxes.at(ib);
			for (int i = 0; i < db_boxPts.size(); i++) {
				line(hudImage, db_boxPts.at(i), db_boxPts.at((i + 1) % db_boxPts.size()), Scalar(0, 255, 0), 2);
			}
		}
		Point_<double> center;
		double ptsSumX=0, ptsSumY= 0;
		for (int ib = 0; ib<_boundingBoxes.size(); ib++) {
		    std::vector<cv::Point> db_boxPts = _boundingBoxes.at(ib);
		    for (int i = 0; i < db_boxPts.size(); i++) {
			    ptsSumX += db_boxPts.at(i).x;
			    ptsSumY += db_boxPts.at(i).y;
		    }
		    center.x = ptsSumX/ db_boxPts.size();
		    center.y = ptsSumY/ db_boxPts.size();
		    circle(hudImage, center, 3, Scalar(0, 0, 255), 2);
		}

		stringstream ss;
		ss.str("");
		ss.clear();
		ss << "#" << _frameIdx;
		putText(hudImage, ss.str(), Point(hudImage.cols - 60, 20), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0));		
		cv::namedWindow(windowTitle.c_str(), cv::WINDOW_AUTOSIZE);
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
		case 'd':      // clear all boxes and start a new box
		_hasInitBox = false;
		_isTrackerInitialzed = false;
		break;
		case 'a':       // add a box at a tme
		_updateAtPos = true;
		break;
		default:
		;
	}
	return true;
}

inline bool existFileTest(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}