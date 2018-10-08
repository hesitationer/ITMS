/*	video stabilization algorithm comparisons 
	implemented by sangkny 
	
	*/

#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
// 
//#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>	// depending : calcOpticalFlowPyrLK, estimateRigidTransform
// 
#include <iostream>
// include itms utilities
#include "itms_utils.h"
using namespace cv;
using namespace std;

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

bool debugGeneral = true;
bool debugImshow = true;
bool flagVideostabilization = true;
bool debugTime = true;

enum VideoStabType {	// video stabilization Types
	VS_PHASE = 0,		// phase correlation
	VS_KALMAN_OPTICALFLOW_LPF = 1,	// kalman and Low Pass Filtering for realtime processing
	VS_OPTICALFLOW = 2	// Optical flow
};
VideoStabType videostabtype = VS_PHASE;

// utility functions for phase correlation
// using namespace cv;
// and whatever header 'abs' requires...

//  offsetImageWithPadding : shift the image to offsetX, offsetY with backrouhdColour paddings
Mat offsetImageWithPadding(const Mat& originalImage, int offsetX, int offsetY, Scalar backgroundColour) {  
  cv::Mat padded = Mat(originalImage.rows + 2 * abs(offsetY), originalImage.cols + 2 * abs(offsetX), originalImage.type(), backgroundColour);
  originalImage.copyTo(padded(Rect(abs(offsetX), abs(offsetY), originalImage.cols, originalImage.rows)));
  return Mat(padded, Rect(abs(offsetX) + offsetX, abs(offsetY) + offsetY, originalImage.cols, originalImage.rows));
}
//example use with black borders along the right hand side and top:
//Mat offsetImage = offsetImageWithPadding(originalImage, -10, 6, Scalar(0, 0, 0));

// class definition for testing optical flow-based approach : Tracker
class Tracker {
	vector<Point2f> trackedFeatures;
	Mat             prevGray;
public:
	bool            freshStart;
	Mat_<float>     rigidTransform;

	Tracker() :freshStart(true)
	{
		rigidTransform = Mat::eye(3, 3, CV_32FC1); //affine 2x3 in a 3x3 matrix
	}

	void processImage(Mat& img)
	{
		Mat gray; cvtColor(img, gray, CV_BGR2GRAY);
		vector<Point2f> corners;
		if (trackedFeatures.size() < 200)
		{
			goodFeaturesToTrack(gray, corners, 300, 0.01, 1);
			cout << "found " << corners.size() << " features\n";
			for (unsigned int i = 0; i < corners.size(); ++i)
			{
				trackedFeatures.push_back(corners[i]);
			}
		}

		if (!prevGray.empty()) {
			vector<uchar> status; vector<float> errors;
			calcOpticalFlowPyrLK(prevGray, gray, trackedFeatures, corners, status, errors, Size(10, 10));

			if (countNonZero(status) < status.size() * 0.8)
			{
				cout << "cataclysmic error \n";
				rigidTransform = Mat::eye(3, 3, CV_32FC1);
				trackedFeatures.clear();
				prevGray.release();
				freshStart = true;
				return;
			}
			else
				freshStart = false;

			Mat_<float> newRigidTransform = estimateRigidTransform(trackedFeatures, corners, false);
			Mat_<float> nrt33 = Mat_<float>::eye(3, 3);
			newRigidTransform.copyTo(nrt33.rowRange(0, 2));
			rigidTransform *= nrt33;

			trackedFeatures.clear();
			for (unsigned int i = 0; i < status.size(); ++i)
			{
				if (status[i]) {
					trackedFeatures.push_back(corners[i]);
				}
			}
		}

		for (unsigned int i = 0; i < trackedFeatures.size(); ++i)
		{
			circle(img, trackedFeatures[i], 3, Scalar(0, 0, 255), CV_FILLED);
		}

		gray.copyTo(prevGray);
	}
};



int main(int, char*[])
{
#ifdef _sk_Memory_Leakag_Detector
#if _DEBUG
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif	
  std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

  // std::string filename = "D:/LectureSSD_rescue/project-related/도로-기상-유고-토페스/code/ITMS/TrafficVideo/20180912_134130_cam_0.avi";
  std::string filename = "D:/LectureSSD_rescue/project-related/도로-기상-유고-토페스/code/ITMS/TrafficVideo/Relaxinghighwaytraffic.mp4";
  //VideoCapture video(0);
  VideoCapture video(filename);
  if (!video.isOpened()) {
    std::cout << " the video file does not exist!!\n"; // iostream
    return 0;
  }

  Mat frame, curr, prev, curr64f, prev64f, hann;
  Mat correctedMat, diffMat;
  char key;
  int frameNumCounter = 0;
  float scalefactor = 1;
  // time measure components
  double t1, t2, t3;  
  
  // for opticalflow
  Tracker tracker;

  do
  {
    video >> frame;

    if (frame.empty()) {
      cout << "the current frame has some problem!! and exit the program(!)\n";
      getchar();

#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
      _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	

      return 0;
    }

    resize(frame, frame, cv::Size(), scalefactor, scalefactor);
    cvtColor(frame, curr, COLOR_RGB2GRAY);

    if (prev.empty())
    {
      prev = curr.clone();
      createHanningWindow(hann, curr.size(), CV_64F);
    }

    prev.convertTo(prev64f, CV_64F);
    curr.convertTo(curr64f, CV_64F);
	if (debugTime) {
		t1 = (double)cvGetTickCount();		
	}
	// video stabilization types
	Point2d shift;
	switch (videostabtype)
	{
		case VS_PHASE:
			shift = phaseCorrelate(prev64f, curr64f, hann);
			break;
		case VS_KALMAN_OPTICALFLOW_LPF:
			break;
		case VS_OPTICALFLOW:
			break;
		default:
			shift = phaseCorrelate(prev64f, curr64f, hann);
			break;
	}
    
	if (debugTime) {
		t2 = (double)cvGetTickCount();
		t3 = (t2 - t1) / (double)getTickFrequency();
		cout << "Processing time>>  #:" << (frameNumCounter - 1) << " --> " << t3*1000.0 << "msec, " << 1. / t3 << "fps \n";
	}
    double radius = std::sqrt(shift.x*shift.x + shift.y*shift.y);    

    if (radius >= 1)
    {
      // draw a circle and line indicating the shift direction...
      Point center(curr.cols >> 1, curr.rows >> 1);
      circle(frame, center, (int)radius, Scalar(0, 255, 0), 3, LINE_AA);
      line(frame, center, Point(center.x + (int)shift.x, center.y + (int)shift.y), Scalar(0, 255, 0), 3, LINE_AA);
      // correct the current image    
      if (flagVideostabilization) {
        correctedMat = offsetImageWithPadding(curr, shift.x, shift.y, Scalar(0, 0, 0));
		if (debugImshow)
			itms::imshowBeforeAndAfter(curr, correctedMat, "before and after", 10);
      }
      if (debugGeneral && debugImshow && flagVideostabilization) {
        cout << " shift (x,y) : (" << shift.x << ", " << shift.y << ")\n";

        cv::absdiff(prev, correctedMat, diffMat);
        cv::Mat thdiffMat= cv::Mat::zeros(diffMat.size(), diffMat.type());
        cv::threshold(diffMat, thdiffMat, 30, 255.0, CV_THRESH_BINARY);        
        imshow(" difference between pre and crtMat(curr)", thdiffMat);
        diffMat = cv::Mat::zeros(diffMat.size(), diffMat.type());
        cv::absdiff(prev, curr, diffMat);
        thdiffMat = cv::Mat::zeros(diffMat.size(), diffMat.type());
        cv::threshold(diffMat, thdiffMat, 30, 255.0, CV_THRESH_BINARY);
        imshow(" difference between pre and curr", thdiffMat);
      }
    }
    if (debugGeneral && debugImshow) {
      imshow("phase shift", frame);      
    }
    key = (char)waitKey(2);

    prev = curr.clone();
    if (debugGeneral) {
      cout << "frame #" << frameNumCounter << endl;      
    }
    frameNumCounter++;
  } while (key != 27); // Esc to exit...

#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	

  return 0;
}