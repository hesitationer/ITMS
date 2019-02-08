// main.cpp


#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include<iostream>			// cout etc
#include<fstream>			// file stream (i/ofstream) etc



#include <sstream>
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows
#include <time.h>

#include "itms_Blob.h"
#include "bgsubcnt.h"

#include "./utils/itms_utils.h"
// DSST
//#include <memory> // for std::unique_ptr 
//#include <algorithm>
//#include "./src/fastdsst/fdssttracker.hpp"


#define SHOW_STEPS            // un-comment or comment this line to show steps or not
//#define HAVE_OPENCV_CONTRIB
#ifdef HAVE_OPENCV_CONTRIB
#include <opencv2/video/background_segm.hpp>
using namespace cv::bgsegm;
#endif
using namespace cv;
using namespace std;
using namespace itms;


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


Config conf;


bool loadConfig(itms::Config& _conf)
{
	std::string configFile = "./config/Area.xml";
	std::string roadmapFile = "./config/roadMapPoints0.xml";
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

		_conf.scaleFactor = cvReadRealByName(fs, 0, "scaleFactor", 0.5);
		_conf.isAutoBrightness = cvReadIntByName(fs, 0, "isAutoBrightness", 1);
		_conf.AutoBrightness_Rect.x = (cvReadIntByName(fs, 0, "AutoBrightness_x", 1162 * _conf.scaleFactor))*_conf.scaleFactor;
		_conf.AutoBrightness_Rect.y = (cvReadIntByName(fs, 0, "AutoBrightness_y", 808 * _conf.scaleFactor))*_conf.scaleFactor;
		_conf.AutoBrightness_Rect.width = (cvReadIntByName(fs, 0, "AutoBrightness_width", 110 * _conf.scaleFactor))*_conf.scaleFactor;
		_conf.AutoBrightness_Rect.height = (cvReadIntByName(fs, 0, "AutoBrightness_heigh", 142 * _conf.scaleFactor))*_conf.scaleFactor;

		_conf.max_past_frames_autoBrightness = cvReadIntByName(fs, 0, "max_past_frames_autoBrightness", 15);
				
		_conf.nightBrightness_Th = cvReadIntByName(fs, 0, "nightBrightness_Th", 20);
		_conf.nightObjectProb_Th = cvReadRealByName(fs, 0, "nightObjectProb_Th", 0.8);

		// detection related
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
			strcpy(_conf.VideoPath, VP);
		if (BGP)
			strcpy(_conf.BGImagePath, BGP);

		// Object Tracking Related
		_conf.minVisibleCount = cvReadIntByName(fs, 0, "minVisibleCound", 3);
		_conf.max_Center_Pts = cvReadIntByName(fs, 0, "max_Center_Pts", 150);
		_conf.numberOfTracePoints = cvReadIntByName(fs, 0, "numberOfTracePoints", 15);
		_conf.maxNumOfConsecutiveInFramesWithoutAMatch = cvReadIntByName(fs, 0, "maxNumOfConsecutiveInFramesWithoutAMatch", 50);
		_conf.maxNumOfConsecutiveInvisibleCounts = cvReadIntByName(fs, 0, "maxNumOfConsecutiveInvisibleCounts", 100);
		_conf.movingThresholdInPixels = cvReadIntByName(fs, 0, "movingThresholdInPixels", 0);

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

		cvReleaseFileStorage(&fs);

		_conf.isLoaded = true;
	}
	else {
		_conf.isLoaded = false;
	}

	if(existFileTest(roadmapFile)){  // try to load        
        FileStorage fr(roadmapFile, FileStorage::READ);        
        if(fr.isOpened()){
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
                for (int i = 0; i<aMat.rows; i++){
                    cout<< i<<": "<< aMat.at<Vec2i>(i)<<endl;
                    road_roi_pts.push_back(Point(aMat.at<Vec2i>(i))*_conf.scaleFactor); // The file data should be measured with original size.
                }
                _conf.Road_ROI_Pts.push_back(road_roi_pts);                
                countlabel++;
            }
            road_roi_pts.clear();
            fr.release();

            if(_conf.debugGeneralDetail)
                for (unsigned j = 0; j < _conf.Road_ROI_Pts.size(); j++) {
                    cout << "Vec:" << _conf.Road_ROI_Pts.at(j) << endl;
               }            
			_conf.existroadMapFile = true;
        }else{
			_conf.existroadMapFile = false;
        }
	}	
    if(!_conf.existroadMapFile){
        if(_conf.debugGeneralDetail){
            cout << " Road Map file is not exist at" << roadmapFile << endl;
            cout<< " the defualt map is used now." << endl;
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
	_conf.Boundary_ROI_Pts.push_back(cv::Point(_conf.Road_ROI_Pts.at(0).at(0).x+interval, _conf.Road_ROI_Pts.at(0).at(0).y+interval));
	_conf.Boundary_ROI_Pts.push_back(cv::Point(_conf.Road_ROI_Pts.at(2).at(1).x-interval, _conf.Road_ROI_Pts.at(2).at(1).y+interval));
	_conf.Boundary_ROI_Pts.push_back(cv::Point(_conf.Road_ROI_Pts.at(2).at(2).x-interval, _conf.Road_ROI_Pts.at(2).at(2).y-std::min(100, 12*interval)));
	_conf.Boundary_ROI_Pts.push_back(cv::Point(_conf.Road_ROI_Pts.at(0).at(3).x+interval, _conf.Road_ROI_Pts.at(0).at(3).y-std::min(100, 12*interval)));

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
		float myconstant{_conf.scaleFactor};
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
	_conf.polyvalue_sedan_h.setValues(conf.vehicleRatios.at(0), conf.vehicleRatios.at(0).size()); // sedan_h
	_conf.polyvalue_sedan_w.setValues(conf.vehicleRatios.at(1), conf.vehicleRatios.at(1).size()); // sedan_w
	_conf.polyvalue_suv_h.setValues(conf.vehicleRatios.at(2), conf.vehicleRatios.at(2).size());   // suv_h
	_conf.polyvalue_suv_w.setValues(conf.vehicleRatios.at(3), conf.vehicleRatios.at(3).size());   // suv_w
	_conf.polyvalue_truck_h.setValues(conf.vehicleRatios.at(4), conf.vehicleRatios.at(4).size()); // truck_h
	_conf.polyvalue_truck_w.setValues(conf.vehicleRatios.at(5), conf.vehicleRatios.at(5).size()); // truck_w
	_conf.polyvalue_human_h.setValues(conf.vehicleRatios.at(6), conf.vehicleRatios.at(6).size()); // human_h
	_conf.polyvalue_human_w.setValues(conf.vehicleRatios.at(7), conf.vehicleRatios.at(7).size()); // human_w

	return true;
}

bool isWriteToFile = false;

// end tracking

int main(void) {
#ifdef _sk_Memory_Leakag_Detector
#if _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif	
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	int trackId = 0;          // unique object id
	int showId = 0;           //     
	std::cout << ".... configurating ...\n";
	loadConfig(conf);
	std::cout << " configarion is done !!\n\n";
	std::unique_ptr<itmsFunctions> itmsFncs;
	itmsFncs= std::make_unique<itmsFunctions>(&conf); // create instance and initialize
	

	cv::VideoCapture capVideo;

	cv::Mat imgFrame1; // previous frame
	cv::Mat imgFrame2; // current frame

  	int carCount = 0;
	int truckCount = 0;
	int bikeCount = 0;
	int humanCount = 0;
	int videoLength = 0;  
	
	bool b = capVideo.open(conf.VideoPath);  
      
	if (!capVideo.isOpened()) {                                                 // if unable to open video file
		std::cout << "reading video file error (!)" << std::endl << std::endl;      // show error message
		_getch();                   // it may be necessary to change or remove this line if not using Windows
		return(0);                                                              // and exit program
	}

  int max_frames = capVideo.get(CV_CAP_PROP_FRAME_COUNT);

	if ( max_frames< 2) {
		std::cout << "error: video file must have at least two frames" << std::endl;
		_getch();                   // it may be necessary to change or remove this line if not using Windows
		return(0);
	}
	/* Event Notice */
	std::cout << "Press 'ESC' to quit..." << std::endl;

	// video information
	int fps = 15;
	bool hasFile = true;  
	if (hasFile)
	{
		fps = int(capVideo.get(CAP_PROP_FPS));
		cout << "Video FPS: " << fps << endl;
	}
	
    capVideo.read(imgFrame1);    
    capVideo.read(imgFrame2);
    if (imgFrame1.empty() || imgFrame2.empty())
      return 0; 


    char chCheckForEscKey = 0;

    bool blnFirstFrame = true;
	int m_startFrame = 0;	 // 240
    int frameCount = m_startFrame + 1;

    // Video save start
	capVideo.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);
    int frame_width = capVideo.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = capVideo.get(CV_CAP_PROP_FRAME_HEIGHT);

    time_t _tm = time(NULL);
    struct tm * curtime = localtime(&_tm);
    std::string yr = std::to_string(1900 + curtime->tm_year);
    std::string mo = std::to_string(1 + curtime->tm_mon);
    std::string da = std::to_string(curtime->tm_mday);
    std::string hr = std::to_string(1 + curtime->tm_hour);
    std::string mi = std::to_string(curtime->tm_min);
    std::string se = std::to_string(curtime->tm_sec);

    std::string videoFilename = "d:\\sangkny\\dataset";

    videoFilename.append(yr);
    videoFilename.append(".");
    videoFilename.append(mo);
    videoFilename.append(".");
    videoFilename.append(da);
    videoFilename.append(".");
    videoFilename.append(hr);
    videoFilename.append(".");
    videoFilename.append(mi);
    videoFilename.append(".avi");

    cv::VideoWriter video(videoFilename, CV_FOURCC('D', 'I', 'V', 'X'), 30, cv::Size(frame_width, frame_height), true);
    // Video save end   
	
    while (capVideo.isOpened() && chCheckForEscKey != 27) {

		double t1 = (double)cvGetTickCount();   

        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();

		itmsFncs->process(imgFrame2); // with current Frame 
  
       // now we prepare for the next iteration
        imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
            capVideo.read(imgFrame2);
			//frameCount++;
			//capVideo.read(imgFrame2);
            ////resize(imgFrame2, imgFrame2, Size(), conf.scaleFactor, conf.scaleFactor);
            if (imgFrame2.empty()) {
              std::cout << "The input image is empty!! Please check the video file!!" << std::endl;
              _getch();
              break;
            }
        } else {
            std::cout << "end of video (!) \n";
            break;
        }

        blnFirstFrame = false;
        frameCount++;
        chCheckForEscKey = cv::waitKey(1);
		if (conf.debugTime) {
			double t2 = (double)cvGetTickCount();
			double t3 = (t2 - t1) / (double)getTickFrequency();
			cout << "Processing time>>  #:" << (frameCount - 1) <<"/("<< max_frames<<")"<< " --> " << t3*1000.0<<"msec, "<< 1./t3 << "fps \n";
		}
    }

    if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
        cv::waitKey(5000);                         // hold the windows open to allow the "end of video" message to show
    }
    // note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows
#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	
    return(0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
