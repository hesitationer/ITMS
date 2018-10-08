#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

#include "itms_utils.h"

#define CAM 0

using namespace cv;
using namespace std;

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

bool debugTime = true;
bool debugShowImages = true;

int main()
{
	
	std::string filename = "D:/LectureSSD_rescue/project-related/도로-기상-유고-토페스/code/ITMS/TrafficVideo/Relaxinghighwaytraffic.mp4";
	//VideoCapture vc(CAM);
	VideoCapture vc(filename);
	if(!vc.isOpened()){
		cout << " please check the device(!), it is not opened. \n";
		return 0;
	}

	Mat frame, orig, orig_warped, tmp;

	Tracker tracker;

	cout << "in main" << endl;
	char pressed_key = 0;
	// time measure components
	double t1, t2, t3;
	double frameNumCounter = 0;
	while (1 && pressed_key != 27/* ESC */)
	{
		vc >> frame;
		frameNumCounter++;

		if (frame.empty()) break;
		frame.copyTo(orig);
		if (debugTime)
			t1 = (double)cvGetTickCount();
		// stabilization 
		tracker.processImage(orig);
		Mat invTrans = tracker.rigidTransform.inv(DECOMP_SVD);
		warpAffine(orig, orig_warped, invTrans.rowRange(0, 2), Size());
		// stabilization end
		if (debugTime) {
			t2 = (double)cvGetTickCount();
			t3 = (t2 - t1) / (double)getTickFrequency();
			cout << "Processing time>>  #:" << (frameNumCounter - 1) << " --> " << t3*1000.0 << "msec, " << 1. / t3 << "fps \n";
		}
		if (debugShowImages) {			
			/*Mat canvas = Mat::zeros(orig.rows, orig.cols * 2 + 10, orig.type());
			orig.copyTo(canvas(Range::all(), Range(0, orig_warped.cols)));
			orig_warped.copyTo(canvas(Range::all(), Range(orig_warped.cols + 10, orig_warped.cols * 2 + 10)));
			if (canvas.cols > 1920)
			{
				resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));
			}
			imshow("before and after", canvas);		*/
			itms::imshowBeforeAndAfter(orig, orig_warped, "after warping", 10);
		}
		
		pressed_key = waitKey(10);
	}
	if(vc.isOpened())
		vc.release();

	return 0;
}