#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <memory> // unique_prt

#include "fdssttracker.hpp"


#include <windows.h>
//#include <dirent.h> // linux or include this file in windows

//#define _sk_Memory_Leakag_Detector
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

//void onMouse(int event, int x, int y, int flags, void* userdata);
void onMouse(int event, int x, int y, int flags, void* userdata) {
	//Point origin;	//
	
	if (event == EVENT_LBUTTONDOWN)
	{
		select_flag = false;
		origin = Point(x, y);
		showRect = Rect(x, y, 0, 0);
	}
	else if (event == EVENT_LBUTTONUP)
	{
		select_flag = true;
		selected = true;
		if (Rect(origin, Point(x, y)).area() > 10){			
			_bROI_Selected = true; // select targetbox anytime
		}else {
			selected = false;
			_bROI_Selected = false;
		}
	}
	if (select_flag)
	{
		showRect.x = MIN(origin.x, x);
		showRect.y = MIN(origin.y, y);
		showRect.width = abs(x - origin.x);
		showRect.height = abs(y - origin.y);
		showRect &= Rect(0, 0, show_img.cols, show_img.rows);
	}
}

std::vector <cv::Mat> imgVec;

int main(int argc, char* argv[]){
#ifdef _sk_Memory_Leakag_Detector
#if _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif	
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = false;
	bool LAB = true;

	string filename = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/20180912_112338_cam_0.avi"/* 20180911_113611_cam_0 20180912_192157_cam_0.avi*//*_paras.device*/;
	VideoCapture cam(/*0*/filename); //webcam : 0

	/*for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "hog") == 0)
			HOG = true;
		if (strcmp(argv[i], "fixed_window") == 0)
			FIXEDWINDOW = true;
		if (strcmp(argv[i], "singlescale") == 0)
			MULTISCALE = false;
		if (strcmp(argv[i], "show") == 0)
			SILENT = false;
		if (strcmp(argv[i], "lab") == 0) {
			LAB = true;
			HOG = true;
		}
		if (strcmp(argv[i], "gray") == 0)
			HOG = false;
	}*/

	// Create KCFTracker object
	// std::unique_ptr<FDSSTTracker> m_tracker(new FDSSTTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB)); // unique_ptr option 1
	std::unique_ptr<FDSSTTracker> m_tracker;
	m_tracker = std::make_unique<FDSSTTracker>(HOG, FIXEDWINDOW, MULTISCALE, LAB);					 // unique_ptr option 2
	//FDSSTTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	//New window
	string window_name = "video | q or esc to quit";
	cv::namedWindow(window_name, 1);

	//std::string PATH_IMG_TOPCV = argv[1];
	//std::string seq_name = argv[2];

	float time_sum = 0.0;

	//int count = 1;
	//cv::Mat image;
	//char name[9];
	//std::string imgName;
	//std::string imgPath = PATH_IMG_TOPCV + "\\" + seq_name + "\\imgs\\";

	////get init target box params from information file
	//std::ifstream initInfoFile;
	//std::string fileName = imgPath + "groundtruth.txt";
	//initInfoFile.open(fileName);
	//std::string firstLine;
	//std::getline(initInfoFile, firstLine);
	float initX=0, initY=0, initWidth=1, initHegiht=1;
	//char ch;
	//std::istringstream ss(firstLine);
	//ss >> initX, ss >> ch;
	//ss >> initY, ss >> ch;
	//ss >> initWidth, ss >> ch;
	//ss >> initHegiht, ss >> ch;
	
	cv::Rect_<float> initRect = cv::Rect(initX, initY, initWidth, initHegiht);
	

	double duration = 0;
	cv::Rect_<float> _roi = initRect;
	cv::Point2f center_point = cv::Point2f(_roi.x + _roi.width / 2.0, _roi.y + _roi.height / 2.0);
	cv::Size target_size = cv::Size(_roi.width, _roi.height);


	/*std::ifstream imagesFile;

	fileName = imgPath + "images.txt";
	imagesFile.open(fileName);
	std::string text;

	std::vector<std::string> filenames;

	while (getline(imagesFile, text))
	{
		filenames.push_back(text);
	}


	std::ofstream resultsFile;
	resultsFile.open(seq_name + ".txt");*/

	

	// Write Results
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);

	// Frame counter
	int nFrames = 0;
	float fscaleFactor = 0.5;

	/*
	if (padded_w >= padded_h)  //fit to width
	_scale = padded_w / (float)template_size;
	else
	_scale = padded_h / (float)template_size;
	*/



	
	//std::string imgFinalPath;

	//float confidence;
	int count = 1;
	//for (int i = 0; i < filenames.size(); i++)
	while(1)
	{
		//std::string imgFinalPath = imgPath + "\\" + filenames[i];
		//show_img = cv::imread(imgFinalPath, IMREAD_COLOR);

		//processImg = cv::imread(imgFinalPath, CV_LOAD_IMAGE_COLOR);
		cam >> show_img;

		if (show_img.empty())
		{
			std::cout<< "End of File" <<std::endl;
			break;
		}
		resize(show_img, show_img, Size(), fscaleFactor, fscaleFactor);
		
		//set Rect
		setMouseCallback(window_name, onMouse, 0);		
		//Using min and max of X and Y for groundtruth rectangle
		float xMin = showRect.x;
		float yMin = showRect.y;
		float width = showRect.width;
		float height = showRect.height;
		if(selected){
			cv::Rect_<float> initRect = cv::Rect(xMin, yMin, width, height);

			
			if (count == 1 || _bROI_Selected)
			{
				cv::Mat img;
				cv::cvtColor(show_img, img, cv::COLOR_RGB2GRAY);
				m_tracker->init(initRect, img);
				showRect = initRect;
				_bROI_Selected = false;
			}
			else{
#ifndef WINDOWS
				LARGE_INTEGER t1, t2, tc;
				QueryPerformanceFrequency(&tc);
				QueryPerformanceCounter(&t1);
#endif
				cv::Mat img;
				cv::cvtColor(show_img, img, cv::COLOR_RGB2GRAY);
				double tt1 = (double)cv::getTickCount();
				showRect = m_tracker->update(img);
				double tt2 = (double)cv::getTickCount();
				double tt3 = (tt2 - tt1) / (double)getTickFrequency();
				ostringstream os;
				os << float(1/tt3);
				putText(show_img, "FPS: " + os.str(), Point(100, 30), FONT_HERSHEY_SIMPLEX,
					0.75, Scalar(225, 0, 0), 2);
				cout << "Processing time>>  #:" << tt3*1000.0 << "msec, " << 1. / tt3 << "fps \n";

				resultsFile << showRect.x << "," << showRect.y << "," << showRect.width << "," << showRect.height << endl;

#ifndef WINDOWS
				QueryPerformanceCounter(&t2);
				printf("Use Time : %f\n", (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);
				time_sum += ((t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);
#endif
			// printf( "rect (w h): %d %d \n" , showRect.width, showRect.height);
			}
			if(!SILENT){
				cv::rectangle(show_img, showRect, cv::Scalar(0, 255, 0));
				cv::imshow(window_name, show_img);
				cv::waitKey(1);
			}
			count++;
		}
		else {
			if (!SILENT) {
				cv::rectangle(show_img, Point(xMin, yMin), Point(xMin + width, yMin + height), Scalar(0, 255, 255), 1, 8);
				imshow(window_name, show_img);
			}
		}
		char key = (char)waitKey(10); //delay N millis, usually long enough to display and capture input

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
	std::cout << "FPS: " << count/ time_sum  << "\n";

	system("pause");
	if (resultsFile.is_open())
		resultsFile.close();

#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	

	return 0;

}
