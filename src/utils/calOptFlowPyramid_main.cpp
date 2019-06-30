#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
const int KEY_SPACE = 32;
const int KEY_ESC = 27;

namespace Config {
	float scaleFactor=0.5;
};

#define MAX_POINTS 16

//Config::scaleFactor = .5;
Point _pt1, _pt2;
bool _bLeftDownAndMove = false;
bool _bROI_Selected = false;
void onMouse(int mevent, int x, int y, int flags, void* param) {
	switch (mevent)
	{
		case EVENT_LBUTTONDOWN:
			_bLeftDownAndMove = false;
			_pt1 = Point(x,y);
			break;
		case EVENT_MOUSEMOVE:
			
			if (flags == EVENT_FLAG_LBUTTON) {
				_pt2 = Point(x,y);
				_bLeftDownAndMove = true;
			}
			break;
		case EVENT_LBUTTONUP:
			_pt2 = Point(x,y);
			_bROI_Selected = true;
			_bLeftDownAndMove = false;
			break;
	}
}

void DrawTrackingPoints(vector<Point2f> &points, Mat &image) {
	for (int i = 0; i < points.size(); i++) {
		int x = cvRound(points[i].x);
		int y = cvRound(points[i].y);
		circle(image, Point(x,y), 3, Scalar(255,0,0),2);
	}
}

void detect(Mat img);
inline bool existFileTest(const std::string& name);

int main(int argc, char** argv)
{
  std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;
   
  //IplImage  *frame; // opencv 2.x
  Mat frame, dstImg; // opencv 3.x
  Mat curImg, prevImg; // gray image
  int input_resize_percent = 100;
  std::string runtime_data_dir = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/";  
  std::string videoFile = runtime_data_dir + "TrafficVideo/20180911_113611_cam_0.avi"; //20180912_112338_cam_0  // 20180911_113611_cam_0 // 20180911_113611_cam_0
   
  bool bfile = false;
  bfile = existFileTest(videoFile);
  std::cout << " file name: " << videoFile.c_str() << std::endl;
  VideoCapture capture(videoFile);
 
  if (!capture.isOpened()) {
	  std::cout << " the video file does not exist!!\n"; // iostream
	  return 0;
  } 
  namedWindow("dstImg");
  setMouseCallback("dstImg", onMouse, NULL);
  
  // store the result with video file
  int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');
  bool isColor = true;
  int fps = 24;
  Size size = Size((int)capture.get(CAP_PROP_FRAME_WIDTH)*Config::scaleFactor, (int)capture.get(CAP_PROP_FRAME_HEIGHT)*Config::scaleFactor);
  VideoWriter outputVideo("trackingRect.avi", fourcc, fps, size, isColor);
  if (fourcc != -1) { // for waiting read the camera if you use it
	  imshow("dstImg", NULL);
	  waitKey(100); // it is not working because there is no window
  }
  TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 0.01);
  Size winSize(11,11);

  vector<Point2f> prevPts;		// previous points
  vector<Point2f> curPts;		// current points
  vector<Point2f> boundPts;		// boundary points
  int delay = 1000/fps;
  int nFrame = 0;
  while(1)
  {
    //frame1 = cvQueryFrame(capture);
	  capture >> frame;
	  if (frame.empty()) {
		  cout << "frame is not valid or end of video !!" << endl;
		  break;
	  }
	  //cvResize(frame1, frame);
	  resize(frame, frame, Size(), Config::scaleFactor, Config::scaleFactor, 1);
	  frame.copyTo(dstImg);	  
	  imshow("dstImg", dstImg);
	  cvtColor(dstImg, curImg, COLOR_BGR2GRAY);
	  if(prevImg.empty())
	    curImg.copyTo(prevImg);
		
	  GaussianBlur(curImg, curImg, Size(5,5), 0.5);	  
	  

	  if (_bLeftDownAndMove) {
		  rectangle(dstImg, _pt1, _pt2, Scalar(0, 0, 255), 2);
		  outputVideo << dstImg;
		  imshow("dstImg", dstImg);
	  }
	  if (_bROI_Selected) { // if initialized tracking points
		Mat mask(size, CV_8U);
		mask = 0; 
		int w = _pt2.x - _pt1.x + 1;
		int h = _pt2.y - _pt1.y +1;
		mask(Rect(_pt1.x, _pt1.y, w, h)) = 1;

		double qualityLevel = 0.001;
		double minDistance = 10;
		int blockSize = 3;
		prevPts.clear();
		goodFeaturesToTrack(prevImg,prevPts, MAX_POINTS, qualityLevel, minDistance, mask, blockSize, true, 0.04);
		cornerSubPix(prevImg, prevPts, winSize, Size(-1,-1), criteria);
		DrawTrackingPoints(prevPts, dstImg);

		// find minAreaRect, and reset the boundaryPts
		RotatedRect minRect = minAreaRect(prevPts);
		Point2f rectPts[4];
		minRect.points(rectPts);
		for(int i=0; i< 4; i++)
			boundPts.push_back(rectPts[i]);

		outputVideo << dstImg;
		_bROI_Selected = false;	
	  }
	  if (prevPts.size() > 0) {
		  vector<Mat> prevPyr, curPyr;
		  Mat status, err;
		  /*buildOpticalFlowPyramid(prevImg, prevPyr, winSize, 3, true);
		  buildOpticalFlowPyramid(curImg, curPyr, winSize, 3, true);
		  calcOpticalFlowPyrLK(prevPyr, curPyr, prevPts, curPts, status, err, winSize);*/
		  
		  calcOpticalFlowPyrLK(prevImg, curImg, prevPts, curPts, status, err, winSize);

		  for (int i = 0; i < prevPts.size(); i++) {
			  if (!status.at<uchar>(i)) {
				prevPts.erase(prevPts.begin()+i);
				curPts.erase(curPts.begin()+i);
				i--;
			  }
		  }		
		
	  }
	  if (curPts.size() >= 4) {
		  cornerSubPix(curImg, curPts, winSize, Size(-1,-1), criteria);
		  DrawTrackingPoints(curPts, dstImg);

		  //// transform boundPts using M
		  //Mat M = findHomography(prevPts, curPts, RANSAC);
		  //perspectiveTransform(boundPts, boundPts, M);
		  //for (int i = 0; i < 4; i++) {
			 // line(dstImg, boundPts[i], boundPts[(i+1)%4], Scalar(0, 255, 255), 2);
		  //}
		  outputVideo << dstImg;
		  imshow("dstImg", dstImg);
		  //prevPts = curPts;
		  swap(prevPts, curPts);
	  }


	  //curImg.copyTo(prevImg);
	  swap(prevImg, curImg);

	  nFrame++;
	  int key = waitKey(100);

	  if(key == KEY_SPACE)
		  key = cvWaitKey(0);

	  if(key == KEY_ESC)
		  break;

  }  

  return 0;
}


inline bool existFileTest(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

//// opencv example 
//#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"
//#include "opencv2/highgui.hpp"
//#include <iostream>
//#include <ctype.h>
//using namespace cv;
//using namespace std;
//static void help()
//{
//	// print a welcome message, and the OpenCV version
//	cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
//		"Using OpenCV version " << CV_VERSION << endl;
//	cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
//	cout << "\nHot keys: \n"
//		"\tESC - quit the program\n"
//		"\tr - auto-initialize tracking\n"
//		"\tc - delete all the points\n"
//		"\tn - switch the \"night\" mode on/off\n"
//		"To add/remove a feature point click it\n" << endl;
//}
//Point2f point;
//bool addRemovePt = false;
//static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
//{
//	if (event == EVENT_LBUTTONDOWN)
//	{
//		point = Point2f((float)x, (float)y);
//		addRemovePt = true;
//	}
//}
//int main(int argc, char** argv)
//{
//	VideoCapture cap;
//	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
//	Size subPixWinSize(10, 10), winSize(31, 31);
//	const int MAX_COUNT = 500;
//	bool needToInit = false;
//	bool nightMode = false;
//	help();
//	cv::CommandLineParser parser(argc, argv, "{@input|0|}");
//	string input = parser.get<string>("@input");
//	  std::string runtime_data_dir = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/";  
//	  std::string videoFile = runtime_data_dir + "TrafficVideo/20180911_113611_cam_0.avi"; //20180912_112338_cam_0  // 20180911_113611_cam_0 // 20180911_113611_cam_0
//	  input = videoFile;
//	if (input.size() == 1 && isdigit(input[0]))
//		cap.open(input[0] - '0');
//	else
//		cap.open(input);
//	if (!cap.isOpened())
//	{
//		cout << "Could not initialize capturing...\n";
//		return 0;
//	}
//	namedWindow("LK Demo", 1);
//	setMouseCallback("LK Demo", onMouse, 0);
//	Mat gray, prevGray, image, frame;
//	vector<Point2f> points[2];
//	for (;;)
//	{
//		cap >> frame;
//		if (frame.empty())
//			break;
//		frame.copyTo(image);
//		cvtColor(image, gray, COLOR_BGR2GRAY);
//		if (nightMode)
//			image = Scalar::all(0);
//		if (needToInit)
//		{
//			// automatic initialization
//			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
//			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
//			addRemovePt = false;
//		}
//		else if (!points[0].empty())
//		{
//			vector<uchar> status;
//			vector<float> err;
//			if (prevGray.empty())
//				gray.copyTo(prevGray);
//			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
//				3, termcrit, 0, 0.001);
//			size_t i, k;
//			for (i = k = 0; i < points[1].size(); i++)
//			{
//				if (addRemovePt)
//				{
//					if (norm(point - points[1][i]) <= 5)
//					{
//						addRemovePt = false;
//						continue;
//					}
//				}
//				if (!status[i])
//					continue;
//				points[1][k++] = points[1][i];
//				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8); // draw only current points
//			}
//			points[1].resize(k);	// erase and refine the strong points
//		}
//		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
//		{
//			vector<Point2f> tmp;
//			tmp.push_back(point);
//			cornerSubPix(gray, tmp, winSize, Size(-1, -1), termcrit);
//			points[1].push_back(tmp[0]);
//			addRemovePt = false;
//		}
//		needToInit = false;
//		imshow("LK Demo", image);
//		char c = (char)waitKey(10);
//		if (c == 27)
//			break;
//		switch (c)
//		{
//		case 'r':
//			needToInit = true;
//			break;
//		case 'c':
//			points[0].clear();
//			points[1].clear();
//			break;
//		case 'n':
//			nightMode = !nightMode;
//			break;
//		}
//		std::swap(points[1], points[0]);
//		cv::swap(prevGray, gray);
//	}
//	return 0;
//}