#ifndef _REALTIMEVIDEOSTABILIZATION_H
#define _REALTIMEVIDEOSTABILIZATION_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/gpu/gpu.hpp"
#include <opencv2/opencv.hpp>
/* 
realtime video stabilization paper
implemented by sangkny on 2018. 10. 05

*/
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>

using namespace cv;
using namespace std;

class VideoStab
{
public:
	VideoStab();
	VideoCapture capture;

	Mat frame2;
	Mat frame1;

	int k;

	const int HORIZONTAL_BORDER_CROP = 20;

	Mat smoothedMat;
	Mat affine;

	Mat smoothedFrame;

	double dx;
	double dy;
	double da;
	double ds_x;
	double ds_y;

	double sx;
	double sy;

	double scaleX;
	double scaleY;
	double thetha;
	double transX;
	double transY;

	double diff_scaleX;
	double diff_scaleY;
	double diff_transX;
	double diff_transY;
	double diff_thetha;

	double errscaleX;
	double errscaleY;
	double errthetha;
	double errtransX;
	double errtransY;

	double Q_scaleX;
	double Q_scaleY;
	double Q_thetha;
	double Q_transX;
	double Q_transY;

	double R_scaleX;
	double R_scaleY;
	double R_thetha;
	double R_transX;
	double R_transY;

	double sum_scaleX;
	double sum_scaleY;
	double sum_thetha;
	double sum_transX;
	double sum_transY;

	Mat stabilize(Mat frame_1, Mat frame_2);
	void Kalman_Filter(double *scaleX, double *scaleY, double *thetha, double *transX, double *transY);
};



#endif // _REALTIMEVIDEOSTABILIZATION_H