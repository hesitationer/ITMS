#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include "realtimeVideoStabilization.h"
#include "itms_utils.h"

using namespace std;
using namespace cv;


const int HORIZONTAL_BORDER_CROP = 30;
const char ESC_KEY = 27;
bool flagWriteVideo = false;
bool debugTime = true;
bool debugImshows = true;

int main(int argc, char **argv)
{

	//Create a object of stabilization class
	VideoStab stab;

	//Initialize the VideoCapture object
	
	// std::string filename = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/20180912_134130_cam_0.avi";
	std::string filename = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/TrafficVideo/Relaxinghighwaytraffic.mp4";
	//VideoCapture cap(0);
	VideoCapture cap(filename);
	if (!cap.isOpened()) {
		std::cout << " the video file does not exist!!\n"; // iostream
		return 0;
	}

	Mat frame_2, frame2;
	Mat frame_1, frame1;

	cap >> frame_1;
	cvtColor(frame_1, frame1, COLOR_BGR2GRAY);

	Mat smoothedMat(2, 3, CV_64F);

	VideoWriter outputVideo;
	outputVideo.open("com.avi", CV_FOURCC('X', 'V', 'I', 'D'), 30, frame_1.size());
	char chCheckForEscKey = 0;
	double t1, t2, t3;
	double frameCount = 0;
	
	while (true && chCheckForEscKey!= ESC_KEY/* ESC */)
	{

		cap >> frame_2;
		frameCount++;

		if (frame_2.data == NULL)
		{
			break;
		}

		cvtColor(frame_2, frame2, COLOR_BGR2GRAY);

		Mat smoothedFrame;
		if (debugTime) {
			t1 = getTickCount();
		}

		smoothedFrame = stab.stabilize(frame_1, frame_2);
		
		if (debugTime) {
			t2 = (double)cvGetTickCount();
			t3 = (t2 - t1) / (double)getTickFrequency();
			cout << "Processing time>>  #:" << (frameCount - 1) << " --> " << t3*1000.0 << "msec, " << 1. / t3 << "fps \n";
		}
		
		if(flagWriteVideo)
			outputVideo.write(smoothedFrame);

		if (debugImshows) {
			itms::imshowBeforeAndAfter(frame_2, smoothedFrame, "stabilized image", 5);
			imshow("Stabilized Video", smoothedFrame);
		}

		chCheckForEscKey =waitKey(10);

		frame_1 = frame_2.clone();
		frame2.copyTo(frame1);

		
	}
	
	if (outputVideo.isOpened()) // actually we don't need to consider closing process because of destructor of outputVideo 
		outputVideo.release();

	return 0;
}