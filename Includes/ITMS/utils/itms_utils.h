/*	this header implements required misc functions for ITMS 
	implemented by sangkny
	sangkny@gmail.com
	last updated on 2018. 10. 07

*/
#ifndef _ITMS_UTILS_H
#define _ITMS_UTILS_H

#include <iostream>
#include "opencv/cv.hpp"
#include "./itms_Blob.h"

using namespace cv;
using namespace std;


// sangkny itms
#ifdef WIN32
#define ITMS_DLL_EXPORT __declspec( dllexport )
#else
#define ITMS_DLL_EXPORT 
#endif


typedef float track_t;

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
		bool debugShowImages = true;
		bool debugShowImagesDetail = true;
		bool debugGeneral = true;
		bool debugGeneralDetail = true;
		bool debugTrace = true;
		bool debugTime = true;
		// 

		// main processing parameters
		float scaleFactor = .5;
		// auto brightness and apply to threshold
		bool isAutoBrightness = true;
		int AutoBrightness_x = 1162;
		int  max_past_frames_autoBrightness = 15;
		cv::Rect AutoBrightness_Rect=cv::Rect(1162, 808, 110, 142);// 1x, default[1162, 808, 110, 142] for darker region,
														  // brighter region [938, 760, 124, 94]; // for a little brighter asphalt
		char VideoPath[512];
		
		char BGImagePath[512];       // background related 
        bool bGenerateBG = true;
		int  intNumBGRefresh = 5 * 30; // 5 seconds * frames/sec


		double StartX = 0;
		double EndX = 0;
		double StartY = 0;
		double EndY = 0;

		// road configuration related
		float camera_height = 11.0 * 100;				// camera height 11 meter
		float lane_length = 200.0 * 100;				// lane length
		float lane2lane_width = 3.5 * 2 * 100;			// lane width

														// road points settings

														// object tracking related 
		bool bNoitifyEventOnce = true;                  // event notification flag: True -> notify once in its life, False -> Keep notifying events
		bool bStrictObjEvent = true;                    // strictly determine the object events (serious determination, rare event and more accurate) 
		int minVisibleCount = 3;						// minimum survival consecutive frame for noise removal effect
		int minConsecutiveFramesForOS = 3;				// minimum consecutive frames for object status determination
		int max_Center_Pts = 5 * 30;					// maximum number of center points (frames), about 5 sec.
		int numberOfTracePoints = 15;					// # of tracking tracer in debug Image
		int maxNumOfConsecutiveInFramesWithoutAMatch = 50; // it is used for track update
		int maxNumOfConsecutiveInvisibleCounts = 100;	// for removing disappeared objects from the screen
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

	inline bool ITMS_DLL_EXPORT loadConfig(Config& _conf)
	{
		std::string configFile = "./config/Area.xml";
		std::string roadmapFile = "./config/roadMapPoints.xml";
		std::string vehicleRatioFile = "./config/vehicleRatio.xml";
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

			// road configuration 		
			_conf.camera_height = cvReadRealByName(fs, 0, "camera_height", 11 * 100);
			_conf.lane_length = cvReadRealByName(fs, 0, "lane_length", 200 * 100);
			_conf.lane2lane_width = cvReadRealByName(fs, 0, "lane2lane_width", 3.5 * 2 * 100);

			_conf.StartX = cvReadRealByName(fs, 0, "StartX", 0);
			_conf.EndX = cvReadRealByName(fs, 0, "EndX", 0);
			_conf.StartY = cvReadRealByName(fs, 0, "StartY", 0);
			_conf.EndY = cvReadRealByName(fs, 0, "EndY", 0);

			_conf.scaleFactor = min(1.0, max(0.1, cvReadRealByName(fs, 0, "scaleFactor", 0.5))); // sangkny 2019. 02. 12 
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
			_conf.minConsecutiveFramesForOS = cvReadIntByName(fs, 0, "minConsecutiveFramesForOS", 3);
			_conf.max_Center_Pts = cvReadIntByName(fs, 0, "max_Center_Pts", 150);
			_conf.numberOfTracePoints = cvReadIntByName(fs, 0, "numberOfTracePoints", 15);
			_conf.maxNumOfConsecutiveInFramesWithoutAMatch = cvReadIntByName(fs, 0, "maxNumOfConsecutiveInFramesWithoutAMatch", 50);
			_conf.maxNumOfConsecutiveInvisibleCounts = cvReadIntByName(fs, 0, "maxNumOfConsecutiveInvisibleCounts", 100);
			_conf.movingThresholdInPixels = cvReadIntByName(fs, 0, "movingThresholdInPixels", 0);

			// Object Speed Limitation

			_conf.img_dif_th = cvReadIntByName(fs, 0, "img_dif_th", 20);

			_conf.BlobNCC_Th = cvReadRealByName(fs, 0, "BlobNCC_Th", 0.5);

			_conf.m_useLocalTracking = cvReadIntByName(fs, 0, "m_useLocalTracking", 0);
			_conf.m_externalTrackerForLost = cvReadIntByName(fs, 0, "m_externalTrackerForLost", 0);
			_conf.isSubImgTracking = cvReadIntByName(fs, 0, "isSubImgTracking", 0);
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
		int interval = 6; // 10 pixel
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
			std::vector<float> sedan_h = { -0.00004444328872f*2.0, 0.01751602326f*2.0, -2.293443176f*2.0, 112.527668f*2.0 }; // scale factor 0.5 need to go header
			std::vector<float> sedan_w = { -0.00003734137716f*2.0, 0.01448943505f*2.0, -1.902199174f*2.0, 98.56691135f*2.0 };
			std::vector<float> suv_h = { -0.00005815785621f*2.0, 0.02216859672f*2.0, -2.797603666f*2.0, 139.0638999f*2.0 };
			std::vector<float> suv_w = { -0.00004854032314f*2.0, 0.01884736545f*2.0, -2.425686251f*2.0, 121.9226426f*2.0 };
			std::vector<float> truck_h = { -0.00006123592908f*2.0, 0.02373661426f*2.0, -3.064585294f*2.0, 149.6535855f*2.0 };
			std::vector<float> truck_w = { -0.00003778247771f*2.0, 0.015239317f*2.0, -2.091105041f*2.0, 110.7544702f*2.0 };
			std::vector<float> human_h = { -0.000002473245036f*2.0, 0.001813179193f*2.0, -0.5058008988f*2.0, 49.27950311f*2.0 };
			std::vector<float> human_w = { -0.000003459461125f*2.0, 0.001590306464f*2.0, -0.3208648543f*2.0, 28.23621306f*2.0 };

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

	class CRegion
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
	};

	typedef std::vector<CRegion> regions_t;

	

	////// simple functions 

  void imshowBeforeAndAfter(cv::Mat &before, cv::Mat &after, std::string windowtitle, int gabbetweenimages);
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
  void matchCurrentFrameBlobsToExistingBlobs(itms::Config& _conf, cv::Mat& preImg, const cv::Mat& srcImg, std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, int& id);
  void addBlobToExistingBlobs(itms::Config& _conf, Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
  void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &id);
  double distanceBetweenPoints(cv::Point point1, cv::Point point2);
  ObjectStatus getObjectStatusFromBlobCenters(Config& config, Blob &blob, const LaneDirection &lanedirection, int movingThresholdInPixels, int minTotalVisibleCount = 3);
  ObjectStatus getObjStatusUsingLinearRegression(Config& config, Blob &blob, const LaneDirection &lanedirection, const int movingThresholdInPixels, const int minTotalVisibleCount = 3);
  void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName, const cv::Scalar& _color=SCALAR_WHITE);
  void drawAndShowContours(itms::Config& _conf, cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
  bool checkIfBlobsCrossedTheLine(itms::Config& _conf, std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
  bool checkIfBlobsCrossedTheLine(itms::Config& _conf, std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy, cv::Point Pt1, cv::Point Pt2, int &carCount, int &truckCount, int &bikeCount);
  bool checkIfBlobsCrossedTheBoundary(itms::Config& _conf, std::vector<Blob> &blobs,/* cv::Mat &imgFrame2Copy,*/ itms::LaneDirection _laneDirection, std::vector<cv::Point> &_tboundaryPts);
  bool checkIfPointInBoundary(const itms::Config& _conf, const cv::Point& p1, const std::vector<cv::Point> &_tboundaryPts);
  void drawBlobInfoOnImage(itms::Config& _conf, std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
  void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);
  void drawRoadRoiOnImage(std::vector<std::vector<cv::Point>> &_roadROIPts, cv::Mat &_srcImg);
  void updateBlobProperties(const Config& _conf, itms::Blob &updateBlob, itms::ObjectStatus &curStatus, const double _speed = 0); // update simple blob properties including os counters
  ObjectStatus computeObjectStatusProbability(const itms::Blob &srcBlob); // compute probability and returns object status 
  																	  // classificy an object with distance and its size
  void classifyObjectWithDistanceRatio(itms::Config& _conf, Blob &srcBlob, float distFromZero/* distance from the starting point*/, ObjectClass & objClass, float& fprobability);
  bool checkObjectStatus(const itms::Config& _conf, const cv::Mat& _curImg, std::vector<Blob>& _Blobs, itms::ITMSResult& _itmsRes);				// check the event true if exists, false otherwise

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

  // sangkny FDSSTTracker m_tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB); // initialze and update !!
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  
  // for API class
  class ITMS_DLL_EXPORT itmsFunctions {
  public:
	  itmsFunctions() {};
	  itmsFunctions(Config* config);
	  bool Init(void);
	  bool process(const cv::Mat& curImg, ITMSResult& _itmsRes);
	  
	  ~itmsFunctions() {};
	  
	  Config* _config;
	  Ptr<BackgroundSubtractor> pBgSub;
	  // parameters
	  bool isInitialized = false;
	  bool isConfigFileLoaded = false;
	  bool m_collectPoints = false; 
	  bool blnFirstFrame = true;
	  // functions
	  cv::Mat preImg;
	  int mCarCount = 0;
	  int maxCarCount = 1024;
  private:
	  std::vector<Blob> blobs;
	  std::vector<int> pastBrightnessLevels; // past brightness checking and adjust the threshold
	  cv::Rect brightnessRoi; // brightness ROI  = conf.AutoBrightness_Rect;
	  cv::Mat BGImage; // background image
	  cv::Mat accmImage; // accumulated Image for background model
	  cv::Mat road_mask;

	  // 
	  cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	  cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	  cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	  cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

	  cv::Mat getBGImage(void) { return BGImage; };
	  void setBGImage(const cv::Mat& _bgImg) { BGImage = _bgImg; };

  };

}



#endif // _ITMS_UTILS_H

