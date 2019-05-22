/*	this header implements required misc functions for ITMS 
	implemented by sangkny
	sangkny@gmail.com
	last updated on 2018. 10. 07

*/
#ifndef _ITMS_UTILS_H
#define _ITMS_UTILS_H

#include <iostream>
#include "opencv/cv.hpp"
#include "itms_Blob.h"
#include "detector/BaseDetector.h"
#include "Tracker/Ctracker.h"

using namespace cv;
using namespace std;


// sangkny itms
#ifdef WIN32
#define ITMS_DLL_EXPORT __declspec( dllexport )
#else
#define ITMS_DLL_EXPORT 
#endif


//typedef float track_t;

namespace itms {
	// global variables ///////////////////////////////////////////////////////////////////////////////
	const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
	const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
	const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
	const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
	const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
	const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
	const cv::Scalar SCALAR_MAGENTA = cv::Scalar(255.0, 0.0, 255.0);
	const cv::Scalar SCALAR_CYAN = cv::Scalar(255.0, 255.0, 0.0);

	//// system related  
	inline bool existFileTest(const std::string& name) {
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	}

	//---------------------------------------------------------------------------
	///

	// Object size fitting class
	class ITMSPolyValues {
	public:
		ITMSPolyValues():mPolySize(0) {};
		// put polynomial coefficient from the index 0 to the end
		ITMSPolyValues(std::vector<float> polyCoeffs, int polyCoeffSize) {
			assert(polyCoeffs.size() == (size_t)polyCoeffSize);
			for (size_t i = 0; i < polyCoeffs.size(); i++)
				mPolyCoeffs.push_back(polyCoeffs.at(i));
			mPolySize = mPolyCoeffs.size();
		};
		void setValues(std::vector<float> polyCoeffs, int polyCoeffSize) {
			assert(polyCoeffs.size() == (size_t)polyCoeffSize);
			if (mPolyCoeffs.size())
				mPolyCoeffs.clear();
			for (size_t i = 0; i < polyCoeffs.size(); i++)
				mPolyCoeffs.push_back(polyCoeffs.at(i));
			mPolySize = mPolyCoeffs.size();
		};

		~ITMSPolyValues() {
			mPolyCoeffs.clear();
		};

		// get a poly value 
		double getPolyValue(float fValue) {
			double pvalue = 0;
			if (mPolySize == 0) {
				std::cout << "No polynomial coefficients exist!! in ITMSPolyValues (!)(!)" << std::endl;
				return -1;
			}
			for (int i = 0; i < mPolySize; i++)
				pvalue += (mPolyCoeffs.at(i)*pow(fValue, mPolySize - 1 - i));
			return pvalue;
		};
		int getPolySize(void) { return mPolySize; };
	private:
		std::vector<float> mPolyCoeffs;
		int mPolySize;
	};

	struct Config
	{
	public:

		// debug parameters
		bool debugShowImages = false;
		bool debugShowImagesDetail = false;
		bool debugGeneral = false;
		bool debugGeneralDetail = false;
		bool debugTrace = false;
		bool debugTime = false;

		bool debugSpecial = false;
		bool debugSaveFile = false;
		// 

		// main processing parameters
		float scaleFactor = .5;
		bool zoomBG = true;		// background subtraction after zooming the longdistance region 
		// auto brightness and apply to threshold
		bool isAutoBrightness = true;
		int AutoBrightness_x = 1162;
		int  max_past_frames_autoBrightness = 15;
		cv::Rect AutoBrightness_Rect= cv::Rect(1419, 205, 121, 56);// sangkny 2019. 05. 22 when night vision has some problem in the previous settings
			// cv::Rect(1162, 808, 110, 142);// 1x, default[1162, 808, 110, 142] for darker region,
														  // brighter region [938, 760, 124, 94]; // for a little brighter asphalt
		char VideoPath[512];
		
		char BGImagePath[512];       // background related 
        bool bGenerateBG = true;
		int  intNumBGRefresh = 5 * 30; // 5 seconds * frames/sec
		double dblMOGVariance = 32; // default : 16 in OpenCV
		double dblMOGShadowThr = 50; // default 50 MOG2 Shadow Filtering Thredhold : 200


		double StartX = 0;
		double EndX = 0;
		double StartY = 0;
		double EndY = 0;

		// road configuration related
		float camera_height = 11.0 * 100;				// camera height 11 meter
		float lane_length = 200.0 * 100;				// lane length
		float lane2lane_width = 3.5 * 2 * 100;			// lane width
		// road points settings
		// object distance settings
		float max_obj_distance = 200.0 * 100;          // maximum object distance in computation
		float min_obj_distance = 0.5 * 100;             // minimum object distance in computation

														// object tracking related 
		bool bNoitifyEventOnce = true;                  // event notification flag: True -> notify once in its life, False -> Keep notifying events
		bool bStrictObjEvent = true;                    // strictly determine the object events (serious determination, rare event and more accurate) 
		int minVisibleCount = 3;						// minimum survival consecutive frame for noise removal effect
		int minVisibleCountHuman = 10;                  // minimum visible counts for human detection
		int minConsecutiveFramesForOS = 3;				// minimum consecutive frames for object status determination
		int max_Center_Pts = 5 * 30;					// maximum number of center points (frames), about 5 sec.
		int numberOfTracePoints = 15;					// # of tracking tracer in debug Image
		int maxNumOfConsecutiveInFramesWithoutAMatch = 50; // it is used for track update
		int maxNumOfConsecutiveInvisibleCounts = 100;	// for removing disappeared objects from the screen
		int minDistanceForBackwardMoving = 1000;		// minimum distance for determining correct backward moving of the object
		int movingThresholdInPixels = 0;				// motion threshold in pixels affected by Config::scaleFactor, average point를 이용해야 함..
		int img_dif_th = 15;							// BGS_DIF biranry threshold (10~30) at day through night, 
														// Day/Night and Object Probability
		int nightBrightness_Th = 20;
		float nightObjectProb_Th = 0.8;

		float BlobNCC_Th = 0.5;							// blob NCC threshold < 0.5 means no BG

		int maxNumOfTrackers = 100;                     // number of maximum trackers (blobs in this project), it determines the limits of show Ids and track ids

		bool m_useLocalTracking = false;				// local tracking  capture and detect level
		bool m_externalTrackerForLost = false;			// do fastDSST for lost object
		bool isSubImgTracking = false;					// come with m_externalTrackerForLost to find out the lost object inSubImg or FullImg
		float sTrackingDist = 10000;                    // tracking start distance from the zero point for detecting stoppin in distance
		bool isCorrectTrackerBlob = false;              // correct blob property from tracker's correction, it has to do more works in globall tracking....
		bool useTrackerMatching = false;                // option: use tracker to match a block to the existing blobs, it can be used to verify same object matching !!
		float useTrackerMatchingThres = 0.75;			// use TrackerMatching Threshold 
		float useTrackerAllowedPercentage = 0.25;		// overlapped region percentage against prediected region

		// define FAST DSST
		bool HOG = true;
		bool FIXEDWINDOW = true;
		bool MULTISCALE = true;
		bool SILENT = false;
		bool LAB = false;

		LaneDirection ldirection = LD_NORTH;			// vertical lane
		BgSubType bgsubtype = BgSubType::BGS_DIF;

		// template matching algorithm implementation, demo
		bool use_mask = false;
		int match_method = cv::TM_CCOEFF_NORMED;
		int max_Trackbar = 5;

		// dnn -based approach 
		// Initialize the parameters
		float confThreshold = 0.1;						// Confidence threshold
		float nmsThreshold = 0.4;						// Non-maximum suppression threshold
		int inpWidthorg = 52;
		int inpHeightorg = 37;

		// configuration file loading flag
		bool isLoaded = false; // loading flag not from file 

							   // road map
		bool existroadMapFile = false; //exist and loaded then true;

		std::vector<std::vector<Point>> Road_ROI_Pts; // sidewalks and carlanes
		std::vector<cv::Point> Boundary_ROI_Pts;		// Boundary_ROI Points
														// vehicle ration 
		bool existvehicleRatioFile = false; // exist and loaded then true
		std::vector<std::vector<float>> vehicleRatios;
		ITMSPolyValues polyvalue_sedan_h; // sedan_h
		ITMSPolyValues polyvalue_sedan_w; // sedan_w
		ITMSPolyValues polyvalue_suv_h;   // suv_h
		ITMSPolyValues polyvalue_suv_w;   // suv_w
		ITMSPolyValues polyvalue_truck_h; // truck_h
		ITMSPolyValues polyvalue_truck_w; // truck_w
		ITMSPolyValues polyvalue_human_h; // human_h
		ITMSPolyValues polyvalue_human_w; // human_w

		// road configuration
		// road deskew matrix
		cv::Mat transmtxH;					// this value will be computed in the processing

		// classifier						// need to check its memory alloaction
		cv::CascadeClassifier cascade;
		cv::HOGDescriptor hog;
		int trackid = 0;					// tracking id: it should be controlloed in the main function.
		int maxTrackIds = 1024;

		// object speed limit related
		int lastMinimumPoints = 30;			// minimum frame to determine the object speed 
		int fps = 30;						// frames per second 
		track_t speedLimitForstopping = 2;		// 4 km/hour for human
	};

	inline bool ITMS_DLL_EXPORT _stdcall loadConfig(Config& _conf)
	{
		std::string configFile = "./config/Area.xml";
		std::string roadmapFile = "./config/roadMapPoints.xml";
		std::string vehicleRatioFile = "./config/vehicleRatio_v2.xml"; // update 2019. 04.30
		std::vector<Point> road_roi_pts;
		// cascade related 
		// define the casecade detector  
		std::string cascadexmlFile = "./config/cascade.xml"; // cars.xml with 1 neighbors good, cascade.xml with 5 neighbors, people cascadG.xml(too many PA) with 4 neighbors and size(30,80), size(80,200)
		if (!existFileTest(cascadexmlFile) || !_conf.cascade.load(cascadexmlFile)) {
			std::cout << "Plase check the xml file at " << cascadexmlFile << std::endl;
			return false;
		}
		_conf.hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector()); // use default descriptor, see reference for more detail

																			 // parameters need to be loaded first
		bool efiletest = existFileTest(configFile);
		if (efiletest) {
			CvFileStorage* fs = cvOpenFileStorage(configFile.c_str(), 0, CV_STORAGE_READ);

			// debug parameters		
			_conf.debugShowImages = cvReadIntByName(fs, 0, "debugShowImages", 1);
			_conf.debugShowImagesDetail = cvReadIntByName(fs, 0, "debugShowImagesDetail", 0);
			_conf.debugGeneral = cvReadIntByName(fs, 0, "debugGeneral", 0);
			_conf.debugGeneralDetail = cvReadIntByName(fs, 0, "debugGeneralDetail", 0);
			_conf.debugTrace = cvReadIntByName(fs, 0, "debugTrace", 0);
			_conf.debugTime = cvReadIntByName(fs, 0, "debugTime", 1);
			_conf.debugSpecial = cvReadIntByName(fs, 0, "debugSpecial", 0);
			_conf.debugSaveFile = cvReadIntByName(fs, 0, "debugSaveFile", 0);

			_conf.scaleFactor = (float)cvReadRealByName(fs, 0, "scaleFactor", 0.5); // sangkny 2019. 02. 12 
			_conf.zoomBG = (bool)cvReadIntByName(fs, 0, "zoomBG", 1);				// sangkny 2019. 04. 30
			
			// road configuration 		
			_conf.camera_height = cvReadRealByName(fs, 0, "camera_height", 11 * 100);
			_conf.lane_length = cvReadRealByName(fs, 0, "lane_length", 200 * 100);
			_conf.lane2lane_width = cvReadRealByName(fs, 0, "lane2lane_width", 3.5 * 2 * 100);

			// object distance
			_conf.max_obj_distance = cvReadRealByName(fs, 0, "max_obj_distance", 20000);
			_conf.min_obj_distance = cvReadIntByName(fs, 0, "min_obj_distance", 50);

			_conf.StartX = cvReadRealByName(fs, 0, "StartX", 0);
			_conf.EndX = cvReadRealByName(fs, 0, "EndX", 0);
			_conf.StartY = cvReadRealByName(fs, 0, "StartY", 0);
			_conf.EndY = cvReadRealByName(fs, 0, "EndY", 0);
			
			_conf.isAutoBrightness = cvReadIntByName(fs, 0, "isAutoBrightness", 1);
			_conf.AutoBrightness_Rect.x = (cvReadIntByName(fs, 0, "AutoBrightness_x", 1162 * _conf.scaleFactor))*_conf.scaleFactor;
			_conf.AutoBrightness_Rect.y = (cvReadIntByName(fs, 0, "AutoBrightness_y", 808 * _conf.scaleFactor))*_conf.scaleFactor;
			_conf.AutoBrightness_Rect.width = (cvReadIntByName(fs, 0, "AutoBrightness_width", 110 * _conf.scaleFactor))*_conf.scaleFactor;
			_conf.AutoBrightness_Rect.height = (cvReadIntByName(fs, 0, "AutoBrightness_heigh", 142 * _conf.scaleFactor))*_conf.scaleFactor;

			_conf.max_past_frames_autoBrightness = cvReadIntByName(fs, 0, "max_past_frames_autoBrightness", 15);

			_conf.nightBrightness_Th = cvReadIntByName(fs, 0, "nightBrightness_Th", 20);
			_conf.nightObjectProb_Th = cvReadRealByName(fs, 0, "nightObjectProb_Th", 0.8);

			// detection related
			_conf.bGenerateBG = cvReadIntByName(fs, 0, "bGenerateBG", 1);
			_conf.intNumBGRefresh = cvReadIntByName(fs, 0, "intNumBGRefresh", 150);
			_conf.dblMOGVariance = cvReadRealByName(fs, 0, "dblMOGVariance", 32);
			_conf.dblMOGShadowThr = cvReadRealByName(fs, 0, "dblMOGShadowThr", 50); 

			_conf.ldirection = LaneDirection(cvReadIntByName(fs, 0, "ldirection", LaneDirection::LD_NORTH));
			_conf.bgsubtype = BgSubType(cvReadIntByName(fs, 0, "bgsubtype", BgSubType::BGS_DIF));
			_conf.use_mask = cvReadIntByName(fs, 0, "use_mask", 0);
			_conf.match_method = cvReadIntByName(fs, 0, "match_method", cv::TM_CCOEFF_NORMED);
			_conf.max_Trackbar = cvReadIntByName(fs, 0, "max_Trackbar", 5);
			_conf.confThreshold = cvReadRealByName(fs, 0, "confThreshold", 0.1);
			_conf.nmsThreshold = cvReadRealByName(fs, 0, "nmsThreshold", 0.4);
			_conf.inpWidthorg = cvReadIntByName(fs, 0, "inpWidthorg", 52);
			_conf.inpHeightorg = cvReadIntByName(fs, 0, "inpHeightorg", 37);

			// file loading
			const char *VP = cvReadStringByName(fs, NULL, "VideoPath", NULL);
			const char *BGP = cvReadStringByName(fs, NULL, "BGImagePath", NULL);
			if (VP)
				strcpy_s(_conf.VideoPath, VP);
			if (BGP)
				strcpy_s(_conf.BGImagePath, BGP);

			// Object Tracking Related
			_conf.bNoitifyEventOnce = cvReadIntByName(fs, 0, "bNoitifyEventOnce", true);
			_conf.bStrictObjEvent = cvReadIntByName(fs, 0, "bStrictObjEvent", true);
			_conf.maxNumOfTrackers = cvReadIntByName(fs, 0, "maxNumOfTrackers", 100);
			_conf.minVisibleCount = cvReadIntByName(fs, 0, "minVisibleCound", 3);
			_conf.minVisibleCountHuman = cvReadIntByName(fs, 0, "minVisibleCountHuman", 10);
			_conf.minConsecutiveFramesForOS = cvReadIntByName(fs, 0, "minConsecutiveFramesForOS", 3);
			_conf.max_Center_Pts = cvReadIntByName(fs, 0, "max_Center_Pts", 150);
			_conf.numberOfTracePoints = cvReadIntByName(fs, 0, "numberOfTracePoints", 15);
			_conf.maxNumOfConsecutiveInFramesWithoutAMatch = cvReadIntByName(fs, 0, "maxNumOfConsecutiveInFramesWithoutAMatch", 50);
			_conf.maxNumOfConsecutiveInvisibleCounts = cvReadIntByName(fs, 0, "maxNumOfConsecutiveInvisibleCounts", 100);
			_conf.minDistanceForBackwardMoving = cvReadIntByName(fs, 0, "minDistanceForBackwardMoving", 1000);
			_conf.movingThresholdInPixels = cvReadIntByName(fs, 0, "movingThresholdInPixels", 0);

			// Object Speed Limitation

			_conf.img_dif_th = cvReadIntByName(fs, 0, "img_dif_th", 20);

			_conf.BlobNCC_Th = cvReadRealByName(fs, 0, "BlobNCC_Th", 0.5);

			_conf.m_useLocalTracking = cvReadIntByName(fs, 0, "m_useLocalTracking", 0);
			_conf.m_externalTrackerForLost = cvReadIntByName(fs, 0, "m_externalTrackerForLost", 0);
			_conf.isSubImgTracking = cvReadIntByName(fs, 0, "isSubImgTracking", 0);
			_conf.sTrackingDist = cvReadRealByName(fs, 0, "sTrackingDist", 10000);
			_conf.isCorrectTrackerBlob = cvReadIntByName(fs, 0, "isCorrectTrackerBlob", 0);
			_conf.useTrackerMatching = cvReadIntByName(fs, 0, "useTrackerMatching", 0);
			_conf.useTrackerMatchingThres = cvReadRealByName(fs, 0, "useTrackerMatchingThres", 0.75);
			_conf.useTrackerAllowedPercentage = cvReadRealByName(fs, 0, "useTrackerAllowedPercentage", 0.25);
			_conf.HOG = cvReadIntByName(fs, 0, "HOG", 1);
			_conf.FIXEDWINDOW = cvReadIntByName(fs, 0, "FIXEDWINDOW", 1);
			_conf.MULTISCALE = cvReadIntByName(fs, 0, "MULTISCALE", 1);
			_conf.SILENT = cvReadIntByName(fs, 0, "SILENT", 0);
			_conf.LAB = cvReadIntByName(fs, 0, "LAB", 0);

			// Object Speed Limitation
			_conf.lastMinimumPoints = cvReadIntByName(fs, 0, "lastMinimumPoints", 30);
			_conf.fps = cvReadIntByName(fs, 0, "fps", 30);
			_conf.speedLimitForstopping = cvReadRealByName(fs, 0, "speedLimitForstopping", 2);

			cvReleaseFileStorage(&fs);

			_conf.isLoaded = true;
		}
		else {
			_conf.isLoaded = false;
		}
		_conf.Road_ROI_Pts.clear();
		if (existFileTest(roadmapFile)) {  // try to load        
			FileStorage fr(roadmapFile, FileStorage::READ);
			if (fr.isOpened()) {
				Mat aMat;
				int countlabel = 0;
				while (1) {
					road_roi_pts.clear();
					stringstream ss;
					ss << countlabel;
					string str = "Road" + ss.str();
					cout << str << endl;
					fr[str] >> aMat;
					if (fr[str].isNone() == 1) {
						break;
					}
					for (int i = 0; i<aMat.rows; i++) {
						cout << i << ": " << aMat.at<Vec2i>(i) << endl;
						road_roi_pts.push_back(Point(aMat.at<Vec2i>(i))*_conf.scaleFactor); // The file data should be measured with original size.
					}
					_conf.Road_ROI_Pts.push_back(road_roi_pts);
					countlabel++;
				}
				road_roi_pts.clear();
				fr.release();

				if (_conf.debugGeneralDetail)
					for (unsigned j = 0; j < _conf.Road_ROI_Pts.size(); j++) {
						cout << "Vec:" << _conf.Road_ROI_Pts.at(j) << endl;
					}
				_conf.existroadMapFile = true;
			}
			else {
				_conf.existroadMapFile = false;
			}
		}
		if (!_conf.existroadMapFile) {
			if (_conf.debugGeneralDetail) {
				cout << " Road Map file is not exist at" << roadmapFile << endl;
				cout << " the defualt map is used now." << endl;
			}
			// use default
			//20180912_112338
			// side walk1			
			road_roi_pts.push_back(Point(932.75, 100.25)*_conf.scaleFactor);
			road_roi_pts.push_back(Point(952.25, 106.25)*_conf.scaleFactor);
			road_roi_pts.push_back(Point(434.75, 1055.75)*_conf.scaleFactor);
			road_roi_pts.push_back(Point(235.25, 1054.25)*_conf.scaleFactor);
			_conf.Road_ROI_Pts.push_back(road_roi_pts);
			road_roi_pts.clear();
			// car lane
			road_roi_pts.push_back(Point(949.25, 104.75)*_conf.scaleFactor);
			road_roi_pts.push_back(Point(1015.25, 103.25)*_conf.scaleFactor);
			road_roi_pts.push_back(Point(1105.25, 1048.25)*_conf.scaleFactor);
			road_roi_pts.push_back(Point(416.75, 1057.25)*_conf.scaleFactor);
			_conf.Road_ROI_Pts.push_back(road_roi_pts);
			road_roi_pts.clear();
			// side walk2
			road_roi_pts.push_back(Point(1009.25, 101.75)*_conf.scaleFactor);
			road_roi_pts.push_back(Point(1045.25, 98.75)*_conf.scaleFactor);
			road_roi_pts.push_back(Point(1397.75, 1052.75)*_conf.scaleFactor);
			road_roi_pts.push_back(Point(1087.25, 1049.75)*_conf.scaleFactor);
			_conf.Road_ROI_Pts.push_back(road_roi_pts);
			road_roi_pts.clear();
		}

		// generate the boundary ROI points from Road_ROI_Pts.
		int interval = 1; // 10 pixel
		_conf.Boundary_ROI_Pts.clear();
		_conf.Boundary_ROI_Pts.push_back(cv::Point(_conf.Road_ROI_Pts.at(0).at(0).x + interval, _conf.Road_ROI_Pts.at(0).at(0).y + interval));
		_conf.Boundary_ROI_Pts.push_back(cv::Point(_conf.Road_ROI_Pts.at(2).at(1).x - interval, _conf.Road_ROI_Pts.at(2).at(1).y + interval));
		_conf.Boundary_ROI_Pts.push_back(cv::Point(_conf.Road_ROI_Pts.at(2).at(2).x - interval, _conf.Road_ROI_Pts.at(2).at(2).y - std::min(100, 12 * interval)));
		_conf.Boundary_ROI_Pts.push_back(cv::Point(_conf.Road_ROI_Pts.at(0).at(3).x + interval, _conf.Road_ROI_Pts.at(0).at(3).y - std::min(100, 12 * interval)));

		//absolute coordinator unit( pixel to centimeters) using Homography pp = H*p  
		//float camera_height = 11.0 * 100; // camera height 11 meter
		//float lane_length = 200.0 * 100;  // lane length
		//float lane2lane_width = 3.5 * 2* 100; // lane width
		std::vector<cv::Point2f> srcPts; // skewed ROI source points
		std::vector<cv::Point2f> tgtPts; // deskewed reference ROI points (rectangular. Top-to-bottom representation but, should be bottom-to-top measure in practice

		srcPts.push_back(static_cast<Point2f>(_conf.Road_ROI_Pts.at(1).at(0))); // detect region left-top p0
		srcPts.push_back(static_cast<Point2f>(_conf.Road_ROI_Pts.at(1).at(1))); // detect region right-top p1
		srcPts.push_back(static_cast<Point2f>(_conf.Road_ROI_Pts.at(1).at(2))); // detect region right-bottom p2 
		srcPts.push_back(static_cast<Point2f>(_conf.Road_ROI_Pts.at(1).at(3))); // detect region left-bottom  p3

		tgtPts.push_back(Point2f(0, 0));                        // pp0
		tgtPts.push_back(Point2f(_conf.lane2lane_width, 0));           // pp1
		tgtPts.push_back(Point2f(_conf.lane2lane_width, _conf.lane_length));// pp2
		tgtPts.push_back(Point2f(0, _conf.lane_length));              // pp3

		_conf.transmtxH = cv::getPerspectiveTransform(srcPts, tgtPts); // homography


		if (existFileTest(vehicleRatioFile)) {
			FileStorage fv(vehicleRatioFile, FileStorage::READ);
			if (fv.isOpened()) {
				Mat aMat;
				int countlabel = 0;
				std::vector<float> vehiclePts;
				while (1) {
					vehiclePts.clear();
					stringstream ss;
					ss << countlabel;
					string str = "Vehicle" + ss.str();
					cout << str << endl;
					fv[str] >> aMat;
					if (fv[str].isNone() == 1) {
						break;
					}
					for (int i = 0; i < aMat.rows; i++) {
						cout << i << ": " << aMat.at<float>(i)*_conf.scaleFactor << endl;
						vehiclePts.push_back(aMat.at<float>(i)*_conf.scaleFactor); // The file data should be measured with original size.
					}
					_conf.vehicleRatios.push_back(vehiclePts);
					countlabel++;
				}
				fv.release();
				_conf.existvehicleRatioFile = true;
			}
		}
		if (!_conf.existvehicleRatioFile) { // used default		
			//std::vector<float> sedan_h = { -0.00004382882987f*2.0, 0.0173377779098603f*2.0, -2.28169958272933f*2.0, 112.308488612836f*2.0 }; // scale factor 0.5 need to go header
			//std::vector<float> sedan_w = { -0.00003795583601f*2.0, 0.0146676803999458f*2.0, -1.91394276754689f*2.0, 98.7860907208733f*2.0 }; // 20190430
			std::vector<float> sedan_h = { -0.000039987980491413f*2.0, 0.0161278690798142f*2.0, -2.15112517789863f*2.0, 109.244494635799f*2.0 }; // scale factor 0.5 need to go header
			std::vector<float> sedan_w = { -0.0000383449290771945f*2.0, 0.0148587047328466f*2.0, -1.91739512923723f*2.0, 99.1472802559759f*2.0 }; // 20190506
			std::vector<float> suv_h = { -0.0000573893011f*2.0, 0.02198602567f*2.0, -2.786735669f*2.0, 138.9535103f*2.0 };
			std::vector<float> suv_w = { -0.00004930887826f*2.0, 0.0190299365f*2.0, -2.436554248f*2.0, 122.0330322f*2.0 };
			std::vector<float> truck_h = { -0.00006180993767f*2.0, 0.02390822247f*2.0, -3.076351259f*2.0, 149.7855261f*2.0 };
			std::vector<float> truck_w = { -0.00003778247771f*2.0, 0.015239317f*2.0, -2.091105041f*2.0, 110.7544702f*2.0 };
			std::vector<float> human_h = { -0.000003756096433f*2.0, 0.002062517955f*2.0, -0.4861445445f*2.0, 48.88594015f*2.0 };
			std::vector<float> human_w = { -0.000006119547882f*2.0, 0.002164848881f*2.0, -0.3171686628f*2.0, 27.98164879f*2.0 };

			_conf.vehicleRatios.clear();
			float myconstant{ _conf.scaleFactor };
			std::transform(sedan_h.begin(), sedan_h.end(), sedan_h.begin(), [myconstant](auto& c) {return c*myconstant; }); // multiply my constant
			_conf.vehicleRatios.push_back(sedan_h);

			std::transform(sedan_w.begin(), sedan_w.end(), sedan_w.begin(), [myconstant](auto& c) {return c*myconstant; });
			_conf.vehicleRatios.push_back(sedan_w);

			std::transform(suv_h.begin(), suv_h.end(), suv_h.begin(), [myconstant](auto& c) {return c*myconstant; });
			_conf.vehicleRatios.push_back(suv_h);

			std::transform(suv_w.begin(), suv_w.end(), suv_w.begin(), [myconstant](auto& c) {return c*myconstant; });
			_conf.vehicleRatios.push_back(suv_w);

			std::transform(truck_h.begin(), truck_h.end(), truck_h.begin(), [myconstant](auto& c) {return c*myconstant; });
			_conf.vehicleRatios.push_back(truck_h);

			std::transform(truck_w.begin(), truck_w.end(), truck_w.begin(), [myconstant](auto& c) {return c*myconstant; });
			_conf.vehicleRatios.push_back(truck_w);

			std::transform(human_h.begin(), human_h.end(), human_h.begin(), [myconstant](auto& c) {return c*myconstant; });
			_conf.vehicleRatios.push_back(human_h);

			std::transform(human_w.begin(), human_w.end(), human_w.begin(), [myconstant](auto& c) {return c*myconstant; });
			_conf.vehicleRatios.push_back(human_w);
		}
		// set each vehicle properties // 2019. 01. 31
		_conf.polyvalue_sedan_h.setValues(_conf.vehicleRatios.at(0), _conf.vehicleRatios.at(0).size()); // sedan_h
		_conf.polyvalue_sedan_w.setValues(_conf.vehicleRatios.at(1), _conf.vehicleRatios.at(1).size()); // sedan_w
		_conf.polyvalue_suv_h.setValues(_conf.vehicleRatios.at(2), _conf.vehicleRatios.at(2).size());   // suv_h
		_conf.polyvalue_suv_w.setValues(_conf.vehicleRatios.at(3), _conf.vehicleRatios.at(3).size());   // suv_w
		_conf.polyvalue_truck_h.setValues(_conf.vehicleRatios.at(4), _conf.vehicleRatios.at(4).size()); // truck_h
		_conf.polyvalue_truck_w.setValues(_conf.vehicleRatios.at(5), _conf.vehicleRatios.at(5).size()); // truck_w
		_conf.polyvalue_human_h.setValues(_conf.vehicleRatios.at(6), _conf.vehicleRatios.at(6).size()); // human_h
		_conf.polyvalue_human_w.setValues(_conf.vehicleRatios.at(7), _conf.vehicleRatios.at(7).size()); // human_w

		return true;
	}


	class ITMSResult {
	public:
		ITMSResult() {};
		~ITMSResult(){};
		
		std::vector<std::pair<int, int>> objStatus; // blob id, status
		std::vector<std::pair<int, int>> objClass; // blob id, object class;
		std::vector<cv::Rect> objRect; // blob rect
		std::vector<track_t> objSpeed;	// object speed
		
		void reset(void) { objStatus.clear(); objClass.clear(); objRect.clear(); objSpeed.clear(); };
	};

	/*class CRegion
	{
	public:
		CRegion()
			: m_type(""), m_confidence(-1)
		{
		}

		CRegion(const cv::Rect& rect)
			: m_rect(rect)
		{

		}

		CRegion(const cv::Rect& rect, const std::string& type, float confidence)
			: m_rect(rect), m_type(type), m_confidence(confidence)
		{

		}

		cv::Rect m_rect;
		std::vector<cv::Point2f> m_points;

		std::string m_type;
		float m_confidence;
	};*/

	//typedef std::vector<CRegion> regions_t;

	

	////// simple functions 

  void imshowBeforeAndAfter(cv::Mat &before_, cv::Mat &after_, std::string windowtitle, int gabbetweenimages);
  Rect expandRect(Rect original, int expandXPixels, int expandYPixels, int maxX, int maxY); // 
  Rect maxSqRect(Rect& original, int maxX, int maxY); // make squre with max length
  Rect maxSqExpandRect(Rect& original, float floatScalefactor, int maxX, int maxY); // combine both above with scalefactor
	
																					// returns the value according to the tp position according to starting point sP and ending point eP;
  // + : left/bottom(below), 0: on the line, -: right/top(above)
  bool isPointBelowLine(cv::Point sP, cv::Point eP, cv::Point tP);  
  
  /// vector related
  /*  example to use the pop_front
  if (vec.size() > max_past_frames)
  {
  vec.pop_front(vec.size() - max_past_frames);
  }
  */
  ///
  /// \brief pop_front
  /// \param vec : input vector
  /// \param count
  ///
  template<typename T>
  void pop_front(std::vector<T>& vec, size_t count)
  {
	  assert(count >= 0 && !vec.empty());
	  if (count < vec.size())
	  {
		  vec.erase(vec.begin(), vec.begin() + count);
	  }
	  else
	  {
		  vec.clear(); // delete all elemnts
	  }
  }

  ///
  /// \brief pop_front: delete the first element
  /// \param vec  
  ///
  template<typename T>
  void pop_front(std::vector<T>& vec)
  {
	  assert(!vec.empty());
	  vec.erase(vec.begin());
  }
  ///// get weight values   

  template<typename T>
  T weightFnc(std::vector<T>& _vec, T _tlevel1 = 30/* transition level 1*/, T _tlevel2 = 128, T _maxTh = 30, T _minTh = 10)
  {
	  assert(_tlevel2 > _tlevel1 && _maxTh > _minTh);

	  T m = mean(_vec)[0];	  T res = 0;
	  res = (m <= _tlevel1) ? _maxTh : (m >= _tlevel2 ? _minTh : ((_minTh - _maxTh)*(m - _tlevel1) / (_tlevel2 - _tlevel1) + _maxTh));	  
	  return T(res);
  }  
  template<typename T>
  T weightFnc_x(T _x = 130, T _tlevel1 = 30/* transition level 1*/, T _tlevel2 = 128, T _maxTh = 30, T _minTh = 10)
  {
	  assert(_tlevel2 > _tlevel1 && _maxTh > _minTh);
	  T m = _x;	  T res = 0;
	  return (res = (m <= _tlevel1) ? _maxTh : (m >= _tlevel2 ? _minTh : ((_minTh - _maxTh)*(m - _tlevel1) / (_tlevel2 - _tlevel1) + _maxTh)));
  }
    
  ///// classes
						
  // Wrapper over OpenCV cv::VideoWriter class, with option write or not to write to file.
  // ITMSVideoWriter(bool writeToFile, const char* filename, int codec, double fps, Size frameSize)
  class ITMSVideoWriter {
  public:
    ITMSVideoWriter(bool writeToFile, const char* filename, int codec, double fps, Size frameSize, bool color = true);
    void write(Mat& frame);
    ~ITMSVideoWriter() {
      if (writeToFile)
        if (writer.isOpened())
          writer.release();
    }
  private:
    VideoWriter writer;
    bool writeToFile; // defines whether we need to write frames to file or not (for easy debugging and readability)
  };

 
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // itms main functions
  // function prototypes ////////////////////////////////////////////////////////////////////////////
  // utils
  // get distance in Meters from the image locations according to the predefined ROI 
  float getDistanceInMeterFromPixels(const std::vector<cv::Point2f> &srcPx, const cv::Mat &transmtx /* 3x3*/, const float _laneLength = 20000, const bool flagLaneDirectionTop2Bottom = false);
  // convert camera coordinate to real world coordinate (not 3D)
  cv::Point2f cvtPx2RealPx(const cv::Point2f &srcPx, const cv::Mat &transmtx /* 3x3*/);
  // srcPx: pixel location in the image
  // transmtx: H matrix
  // _laneLength: lane distance 0 to the end of roi
  // flagLaneDirectionTop2Bottom : true -> image coord direction is same with distance measure direction, false: opposite

  // getNCC gets NCC value between background image and a foreground image
  float getNCC(itms::Config& _conf, cv::Mat &bgimg, cv::Mat &fgtempl, cv::Mat &fgmask, int match_method/* cv::TM_CCOEFF_NORMED*/, bool use_mask/*false*/);

  // type2srt returns the type of cv::Mat
  string type2str(int type); // get Math type()
  int InterSectionRect(cv::Rect &rect1, cv::Rect &rect2);

  //////////////////////////////////////////////////////////////////////////////////

  // general : blob image processing (blob_imp)
  void mergeBlobsInCurrentFrameBlobs(itms::Config& _conf, std::vector<Blob> &currentFrameBlobs);
  void mergeBlobsInCurrentFrameBlobsWithPredictedBlobs(std::vector<Blob>& currentFrameBlobs, std::vector<Blob> &predBlobs);
  void matchCurrentFrameBlobsToExistingBlobs(itms::Config& _conf, const cv::Mat& orgImg, cv::Mat& preImg, const cv::Mat& srcImg, std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, int& id);
  void matchExistingBlobsToCurrentFrameBlobs(itms::Config& _conf, cv::Mat& preImg, const cv::Mat& srcImg, std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, int &id);
  void addBlobToExistingBlobs(itms::Config& _conf, Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
  void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &id);
  double distanceBetweenPoints(cv::Point point1, cv::Point point2);
  float angleBetweenPoints(Point p1, Point p2);
  double distanceBetweenBlobs(const itms::Blob& _blob1, const itms::Blob& _blob2);
  ObjectStatus getObjectStatusFromBlobCenters(Config& config, Blob &blob, const LaneDirection &lanedirection, int movingThresholdInPixels, int minTotalVisibleCount = 3);
  ObjectStatus getObjStatusUsingLinearRegression(Config& config, Blob &blob, const LaneDirection &lanedirection, const int movingThresholdInPixels, const int minTotalVisibleCount = 3);
  void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName, const cv::Scalar& _color=SCALAR_WHITE);
  void drawAndShowContours(itms::Config& _conf, cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
  bool checkIfBlobsCrossedTheLine(itms::Config& _conf, std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
  bool checkIfBlobsCrossedTheLine(itms::Config& _conf, std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy, cv::Point Pt1, cv::Point Pt2, int &carCount, int &truckCount, int &bikeCount);
  bool checkIfBlobsCrossedTheBoundary(itms::Config& _conf, std::vector<Blob> &blobs,/* cv::Mat &imgFrame2Copy,*/ itms::LaneDirection _laneDirection, std::vector<cv::Point> &_tboundaryPts);
  bool checkIfPointInBoundary(const itms::Config& _conf, const cv::Point& p1, const std::vector<cv::Point> &_tboundaryPts);
  bool checkIfBlobInBoundaryAndDistance(const itms::Config& _conf, const itms::Blob& _blob, const std::vector<cv::Point> &_tboundaryPts, float& _realDistance);
  void drawBlobInfoOnImage(itms::Config& _conf, std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
  void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);
  void drawRoadRoiOnImage(std::vector<std::vector<cv::Point>> &_roadROIPts, cv::Mat &_srcImg);
  void updateBlobProperties(const Config& _conf, itms::Blob &updateBlob, itms::ObjectStatus &curStatus, const double _speed = 0); // update simple blob properties including os counters
  ObjectStatus computeObjectStatusProbability(const itms::Blob &srcBlob); // compute probability and returns object status 
  																	  // classificy an object with distance and its size
  void classifyObjectWithDistanceRatio(itms::Config& _conf, Blob &srcBlob, float distFromZero/* distance from the starting point*/, ObjectClass & objClass, float& fprobability);
  //bool checkObjectStatus(const itms::Config& _conf, const cv::Mat& _curImg, std::vector<Blob>& _Blobs, itms::ITMSResult& _itmsRes);				// check the event true if exists, false otherwise

  // cascade detector related
  void detectCascadeRoi(itms::Config& _conf, cv::Mat img, cv::Rect& rect);
  void detectCascadeRoiVehicle(itms::Config& _conf, /* put config file */const cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _cars);
  void detectCascadeRoiHuman(const itms::Config& _conf, /* put config file */const cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _people);
  // cascade detector related ends

  //// road configuration
  //// road deskew matrix
  //cv::Mat transmtxH;

  // tracking related blob								
  void collectPointsInBlobs(std::vector<Blob> &_blobs, bool _collectPoints); // collectPoints with a number of blobs
  void collectPointsInBlob(Blob &_blob);
  void getCollectPoints(Blob& _blob, std::vector<Point2f> &_collectedPts);	// get points inside blob, if exists in a blob, otherwise, compute it.

																			// get predicted blobs from the existing blobs
  void predictBlobs(itms::Config& _conf, std::vector<Blob>& tracks/* existing blobs */, cv::UMat prevFrame, cv::UMat curFrame, std::vector<Blob>& predBlobs);

  std::vector<cv::Point> getBlobUnderRect(const Config &_conf, const cv::Mat& _curImg, const cv::Rect& _prect, const itms::Blob& _curBlob);  // sangkny 20190404
  // get the new blob information including contour and etc under the new rect on the given image
  bool doubleCheckStopObject(const Config& _conf, itms::Blob& _curBlob); // check the stop conditions
  bool doubleCheckBackwardMoving(const Config& _conf, itms::Blob& _curBlob);
  bool trackNewLocationFromPrevBlob(const itms::Config& _conf, const cv::Mat& _preImg, const cv::Mat& _srcImg, itms::Blob& _ref, cv::Rect& _new_rect, const int _expandY = 2);
  bool trackNewLocation(const itms::Config& _conf, const cv::Mat& _preImg, const cv::Mat& _srcImg, itms::Blob& _ref, cv::Rect& _new_rect, const int _expandY = 2);  
  
  // sangkny FDSSTTracker m_tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB); // initialze and update !!
  // sort contour related
  bool compareContourAreasDes(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2);
  bool compareContourAreasAsc(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2);
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  
  // for API class
  class ITMS_DLL_EXPORT itmsFunctions {
  public:
	  itmsFunctions() {};
	  itmsFunctions(Config* config);
	  bool Init(void);
	  bool process(const cv::Mat& curImg1, ITMSResult& _itmsRes);
	  
	  ~itmsFunctions() {};
	  
	  Config* _config;
	  Ptr<BackgroundSubtractorMOG2> pBgSub;
	  Ptr<BackgroundSubtractorMOG2> pBgOrgSub; // original size image
	  // parameters
	  bool isInitialized = false;
	  bool isConfigFileLoaded = false;
	  bool m_collectPoints = false; 
	  bool blnFirstFrame = true;
	  // functions
	  cv::Mat preImg;
	  int mCarCount = 0;
	  int maxCarCount = 1024;
	  // file write
	  ofstream out_object_class;

  protected:
	  cv::Point2f zPmin, zPmax; // zoomBG region
	  cv::Point2f ozPmin, ozPmax; // original coordinate for the above coords.
  private:
	  std::vector<Blob> blobs;
	  std::vector<int> pastBrightnessLevels; // past brightness checking and adjust the threshold
	  cv::Rect brightnessRoi; // brightness ROI  = conf.AutoBrightness_Rect;
	  cv::Mat BGImage; // background image
	  cv::Mat accmImage; // accumulated Image for background model
	  cv::Mat road_mask;
	  cv::Mat orgImage; // original image of current image
	  cv::Mat orgPreImage;

	  // 
	  cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	  cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	  cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	  cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

	  cv::Mat getBGImage(void) { return BGImage; };
	  void setBGImage(const cv::Mat& _bgImg) { BGImage = _bgImg; };
	  bool checkObjectStatus(itms::Config& _conf, const cv::Mat& _curImg, std::vector<Blob>& _Blobs, itms::ITMSResult& _itmsRes);				// check the event true if exists, false otherwise
	  float getNCC(itms::Config& _conf, cv::Mat &bgimg, cv::Mat &fgtempl, cv::Mat &fgmask, int match_method/* cv::TM_CCOEFF_NORMED*/, bool use_mask/*false*/);	  
  };
  // line segment class
  class ITMS_DLL_EXPORT LineSegment
  {

  public:
	  cv::Point p1, p2;
	  float slope;
	  float length;
	  float angle;

	  // LineSegment(Point point1, Point point2);
	  LineSegment();
	  LineSegment(int x1, int y1, int x2, int y2);
	  LineSegment(cv::Point p1, cv::Point p2);

	  void init(int x1, int y1, int x2, int y2);

	  bool isPointBelowLine(cv::Point tp);

	  float getPointAt(float x);
	  float getXPointAt(float y);

	  cv::Point closestPointOnSegmentTo(cv::Point p);

	  cv::Point intersection(LineSegment line);

	  LineSegment getParallelLine(float distance);

	  cv::Point midpoint();

	  inline std::string str()
	  {
		  std::stringstream ss;
		  ss << "(" << p1.x << ", " << p1.y << ") : (" << p2.x << ", " << p2.y << ")";
		  return ss.str();
	  }

  };

  // car counting-related classes
  class ITMS_DLL_EXPORT RoadLine
  {
  public:
	  ///
	  /// \brief RoadLine
	  ///
	  RoadLine()
	  {
	  }
	  RoadLine(const cv::Point2f& pt1, const cv::Point2f& pt2, unsigned int uid)
		  :
		  m_pt1(pt1), m_pt2(pt2), m_uid(uid)
	  {
	  }

	  cv::Point2f m_pt1;
	  cv::Point2f m_pt2;

	  unsigned int m_uid = 0;

	  int m_intersect1 = 0;
	  int m_intersect2 = 0;

	  ///
	  /// \brief operator ==
	  /// \param line
	  /// \return
	  ///
	  bool operator==(const RoadLine &line) const
	  {
		  return line.m_uid == m_uid;
	  }

	  ///
	  /// \brief Draw
	  /// \param frame
	  ///
	  void Draw(cv::Mat frame) const
	  {
		  auto Ptf2i = [&](cv::Point2f pt) -> cv::Point
		  {
			  return cv::Point(cvRound(frame.cols * pt.x), cvRound(frame.rows * pt.y));
		  };

		  cv::line(frame, Ptf2i(m_pt1), Ptf2i(m_pt2), cv::Scalar(0, 255, 255), 1, cv::LINE_8, 0);

		  std::string label = "Line " + std::to_string(m_uid) + ": " + std::to_string(m_intersect1) + "/" + std::to_string(m_intersect2);
		  //int baseLine = 0;
		  //cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		  cv::putText(frame, label, Ptf2i(0.5f * (m_pt1 + m_pt2)), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 0));
	  }

	  ///
	  /// \brief IsIntersect
	  /// \param pt1
	  /// \param pt2
	  /// \return
	  ///
	  int IsIntersect(cv::Point2f pt1, cv::Point2f pt2)
	  {
		  bool isIntersect = CheckIntersection(pt1, pt2);
		  int direction = 0;

		  if (isIntersect)
		  {
			  cv::Point2f pt;
			  if ((m_pt1.x <= m_pt2.x) && (m_pt1.y > m_pt1.y))
			  {
				  pt.x = (m_pt1.x + m_pt2.x) / 2.f - 0.01f;
				  pt.y = (m_pt1.y + m_pt1.y) / 2.f - 0.01f;
			  }
			  else
			  {
				  if ((m_pt1.x <= m_pt2.x) && (m_pt1.y <= m_pt1.y))
				  {
					  pt.x = (m_pt1.x + m_pt2.x) / 2.f + 0.01f;
					  pt.y = (m_pt1.y + m_pt1.y) / 2.f - 0.01f;
				  }
				  else
				  {
					  if ((m_pt1.x > m_pt2.x) && (m_pt1.y > m_pt1.y))
					  {
						  pt.x = (m_pt1.x + m_pt2.x) / 2.f - 0.01f;
						  pt.y = (m_pt1.y + m_pt1.y) / 2.f + 0.01f;
					  }
					  else
					  {
						  if ((m_pt1.x > m_pt2.x) && (m_pt1.y <= m_pt1.y))
						  {
							  pt.x = (m_pt1.x + m_pt2.x) / 2.f + 0.01f;
							  pt.y = (m_pt1.y + m_pt1.y) / 2.f + 0.01f;
						  }
					  }
				  }
			  }
			  if (CheckIntersection(pt1, pt))
			  {
				  direction = 1;
				  ++m_intersect1;
			  }
			  else
			  {
				  direction = 2;
				  ++m_intersect2;
			  }
		  }

		  return direction;
	  }

  private:

	  ///
	  /// \brief CheckIntersection
	  /// \param pt1
	  /// \param pt2
	  /// \return
	  ///
	  bool CheckIntersection(cv::Point2f pt1, cv::Point2f pt2) const
	  {
		  const float eps = 0.00001f; // Epsilon for equal comparing

									  // First line equation
		  float a1 = 0;
		  float b1 = 0;
		  bool trivial1 = false; // Is first line is perpendicular with OX

		  if (fabs(m_pt1.x - m_pt2.x) < eps)
		  {
			  trivial1 = true;
		  }
		  else
		  {
			  a1 = (m_pt2.y - m_pt1.y) / (m_pt2.x - m_pt1.x);
			  b1 = (m_pt2.x * m_pt1.y - m_pt1.x * m_pt2.y) / (m_pt2.x - m_pt1.x);
		  }

		  // Second line equation
		  float a2 = 0;
		  float b2 = 0;
		  bool trivial2 = false; // Is second line is perpendicular with OX

		  if (fabs(pt1.x - pt2.x) < eps)
		  {
			  trivial2 = true;
		  }
		  else
		  {
			  a2 = (pt2.y - pt1.y) / (pt2.x - pt1.x);
			  b2 = (pt2.x * pt1.y - pt1.x * pt2.y) / (pt2.x - pt1.x);
		  }

		  // define intersection point
		  cv::Point2f intersectPt;

		  bool isIntersect = true;
		  if (trivial1)
		  {
			  if (trivial2)
			  {
				  isIntersect = (fabs(m_pt1.x - pt1.x) < eps);
			  }
			  else
			  {
				  intersectPt.x = m_pt1.x;
			  }
			  intersectPt.y = a2 * intersectPt.x + b2;
		  }
		  else
		  {
			  if (trivial2)
			  {
				  intersectPt.x = pt1.x;
			  }
			  else
			  {
				  if (fabs(a2 - a1) > eps)
				  {
					  intersectPt.x = (b1 - b2) / (a2 - a1);
				  }
				  else
				  {
					  isIntersect = false;
				  }
			  }
			  intersectPt.y = a1 * intersectPt.x + b1;
		  }

		  if (isIntersect)
		  {
			  auto InRange = [](float val, float minVal, float  maxVal) -> bool
			  {
				  return (val >= minVal) && (val <= maxVal);
			  };

			  isIntersect = InRange(intersectPt.x, std::min(m_pt1.x, m_pt2.x), std::max(m_pt1.x, m_pt2.x) + eps) &&
				  InRange(intersectPt.x, std::min(pt1.x, pt2.x), std::max(pt1.x, pt2.x) + eps) &&
				  InRange(intersectPt.y, std::min(m_pt1.y, m_pt2.y), std::max(m_pt1.y, m_pt2.y) + eps) &&
				  InRange(intersectPt.y, std::min(pt1.y, pt2.y), std::max(pt1.y, pt2.y) + eps);
		  }

		  return isIntersect;
	  }
  };
  // ----------------------------------------------------------------------

  ///
  /// \brief The CarsCounting class
  ///
  class ITMS_DLL_EXPORT CarsCounting
  {
  public:
	  CarsCounting(const cv::CommandLineParser& parser);
	  CarsCounting(Config* config);
	  virtual ~CarsCounting();

	  bool Init(void); // from ITMSFunctions
	  void Process();  // full looping : all frames processes
	  bool process(const cv::Mat& colorFrame, ITMSResult& _itmsRes); // single loop : one frame process
	  Config* _config;

	  // parameters
	  bool isInitialized = false;
	  bool isConfigFileLoaded = false;
	  bool m_collectPoints = false;
	  bool blnFirstFrame = true;
	  
	  // functions
	  cv::Mat preImg;
	  int mCarCount = 0;
	  int maxCarCount = 1024;

	  // Lines API
	  void AddLine(const RoadLine& newLine);
	  bool GetLine(unsigned int lineUid, RoadLine& line);
	  bool RemoveLine(unsigned int lineUid);

  private:
	  //std::vector<Blob> blobs;
	  std::vector<int> pastBrightnessLevels; // past brightness checking and adjust the threshold
	  cv::Rect brightnessRoi; // brightness ROI  = conf.AutoBrightness_Rect;
	  cv::Mat BGImage; // background image
	  cv::Mat accmImage; // accumulated Image for background model
	  cv::Mat road_mask;
	  cv::Mat orgImage; // original image of current image


  protected:
	  std::unique_ptr<BaseDetector> m_detector;
	  std::unique_ptr<CTracker> m_tracker;

	  bool m_showLogs = true;
	  float m_fps = 0;
	  bool m_useLocalTracking = true;

	  virtual bool GrayProcessing() const;

	  virtual bool InitTracker(cv::UMat frame);

	  void Detection(cv::Mat frame, cv::UMat grayFrame, regions_t& regions);
	  void Tracking(cv::Mat frame, cv::UMat grayFrame, const regions_t& regions);

	  virtual void DrawData(cv::Mat frame, int framesCounter, int currTime);

	  void DrawTrack(cv::Mat frame,
		  int resizeCoeff,
		  const CTrack& track,
		  bool drawTrajectory = true,
		  bool isStatic = false);

  private:

	  bool m_isTrackerInitialized = false;
	  std::string m_inFile;
	  std::string m_outFile;
	  int m_startFrame = 0;
	  int m_endFrame = 0;
	  int m_finishDelay = 0;
	  std::vector<cv::Scalar> m_colors;

	  int m_minObjWidth = 5;

	  // Road lines
	  std::deque<RoadLine> m_lines;
	  void CheckLinesIntersection(const CTrack& track, float xMax, float yMax, std::set<size_t>& currIntersections);
	  std::set<size_t> m_lastIntersections;
  };

  // ITMS API Native Class 
  class ITMS_DLL_EXPORT ITMSAPINativeClass
{
public:
	ITMSAPINativeClass();
	~ITMSAPINativeClass();
	int Init();
	int ResetAndProcessFrame(int iCh, unsigned char * pImage, int lSize); // reset and process
	int ResetAndProcessFrame(const cv::Mat& curImg1);
	//std::unique_ptr<ITMSResult> getResult(void);							// it has some problem because of direct calling even with move
	std::vector<std::pair<int, int>> getObjectStatus(void);
	std::vector<std::pair<int, int>> getObjectClass(void);
	std::vector<cv::Rect> getObjectRect(void);
	std::vector<track_t> getObjectSpeed(void);

	Config conf;
	Mat pFrame;

	bool isInitialized;

protected:
	std::unique_ptr<itmsFunctions> itmsFncs;				// itms main class	
	std::unique_ptr<ITMSResult> itmsres;                     // itms result structure		
};

}
#endif // _ITMS_UTILS_H

