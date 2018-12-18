// main.cpp


#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include<iostream>			// cout etc
#include<fstream>			// file stream (i/ofstream) etc


// dnn: deep neural network
#include <opencv2/dnn.hpp>
#include <sstream>
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows
#include <time.h>

#include "../src/itms_Blob.h"
#include "../src/bgsubcnt.h"

#include "utils/itms_utils.h"


#define SHOW_STEPS            // un-comment or comment this line to show steps or not
//#define HAVE_OPENCV_CONTRIB
#ifdef HAVE_OPENCV_CONTRIB
#include <opencv2/video/background_segm.hpp>
using namespace cv::bgsegm;
#endif
using namespace cv;
using namespace std;
using namespace dnn;
using namespace itms;
using namespace dnn;

cv::CascadeClassifier cascade;
cv::HOGDescriptor hog;

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

// #define _CASCADE_HUMAN 

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
const cv::Scalar SCALAR_MAGENTA = cv::Scalar(255.0, 0.0, 255.0);
const cv::Scalar SCALAR_CYAN = cv::Scalar(255.0, 255.0, 0.0);


// function prototypes ////////////////////////////////////////////////////////////////////////////
// utils
// get distance in Meters from the image locations according to the predefined ROI 
float getDistanceInMeterFromPixels(std::vector<cv::Point2f> &srcPx, cv::Mat &transmtx /* 3x3*/, float _laneLength=20000, bool flagLaneDirectionTop2Bottom=false);
// srcPx: pixel location in the image
// transmtx: H matrix
// _laneLength: lane distance 0 to the end of roi
// flagLaneDirectionTop2Bottom : true -> image coord direction is same with distance measure direction, false: opposite

// getNCC gets NCC value between background image and a foreground image
float getNCC(cv::Mat &bgimg, cv::Mat &fgtempl, cv::Mat &fgmask, int match_method/* cv::TM_CCOEFF_NORMED*/, bool use_mask/*false*/);

// type2srt returns the type of cv::Mat
string type2str(int type); // get Math type()
int InterSectionRect(cv::Rect &rect1, cv::Rect &rect2);


///////////// callback function --------------------   ///////////////////////////
//void MatchingMethod(int, void*);

//////////////////////////////////////////////////////////////////////////////////

// general : blob image processing (blob_imp)
void mergeBlobsInCurrentFrameBlobs(std::vector<Blob> &currentFrameBlobs);
void matchCurrentFrameBlobsToExistingBlobs(cv::Mat& srcImg, std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, int& id);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs,int &id);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
ObjectStatus getObjectStatusFromBlobCenters(Blob &blob, const LaneDirection &lanedirection, int movingThresholdInPixels, int minTotalVisibleCount=3);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy, cv::Point Pt1, cv::Point Pt2, int &carCount, int &truckCount, int &bikeCount);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);
void updateBlobProperties(itms::Blob &updateBlob, itms::ObjectStatus &curStatus); // update simple blob properties including os counters
ObjectStatus computeObjectStatusProbability(const itms::Blob &srcBlob); // compute probability and returns object status 

// classificy an object with distance and its size
void classifyObjectWithDistanceRatio(Blob &srcBlob, float distFromZero/* distance from the starting point*/, ObjectClass & objClass, float& fprobability);

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

//-------------------- dnn related -------------------------------/
///////////////////////////////////////////////////////////////////////////////////////////////////
namespace FAV1
{
  int carArea = 0;
  int truckArea = 0;
  int bikeArea =  0;
  int humanArea = 0;

  char VideoPath[512];
  char BGImagePath[512];
  double StartX = 0;
  double EndX = 0;
  double StartY = 0;
  double EndY = 0;
}

void loadConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/Area.xml", 0, CV_STORAGE_READ);
  FAV1::carArea =   cvReadIntByName(fs, 0, "AllVehicleArea", 0);
  FAV1::truckArea = cvReadIntByName(fs, 0, "HeavyVehicleArea", 0);
  FAV1::bikeArea =  cvReadIntByName(fs, 0, "BikeArea", 0);
  FAV1::humanArea = cvReadIntByName(fs, 0, "HumanArea", 0);
  FAV1::StartX =    cvReadRealByName(fs, 0, "StartX", 0);
  FAV1::EndX =      cvReadRealByName(fs, 0, "EndX", 0);
  FAV1::StartY =    cvReadRealByName(fs, 0, "StartY", 0);
  FAV1::EndY =      cvReadRealByName(fs, 0, "EndY", 0);

  const char *VP = cvReadStringByName(fs, NULL, "VideoPath", NULL);
  const char *BGP = cvReadStringByName(fs, NULL, "BGImagePath", NULL);
  strcpy(FAV1::VideoPath, VP);
  strcpy(FAV1::BGImagePath, BGP);
  cvReleaseFileStorage(&fs);
}
enum BgSubType { // background substractor type
	BGS_DIF = 0, // difference
	BGS_CNT = 1, // COUNTER
	BGS_ACC = 2  // ACCUMULATER
};
// parameters
bool debugShowImages = true;
bool debugShowImagesDetail = true;
bool debugGeneral = true;
bool debugGeneralDetail = false;
bool debugTrace = false;
bool debugTime = true;
int numberOfTracePoints = 15;	// # of tracking tracer in debug Image
int minVisibleCount = 3;		  // minimum survival consecutive frame for noise removal effect
int maxCenterPts = 300;			  // maximum number of center points (frames), about 10 sec.
int maxNumOfConsecutiveInFramesWithoutAMatch = 50; // it is used for track update
int maxNumOfConsecutiveInvisibleCounts = 100; // for removing disappeared objects from the screen
int movingThresholdInPixels = 0;              // motion threshold in pixels affected by scaleFactor, average point를 이용해야 함..
int img_dif_th = 10;                          // BGS_DIF biranry threshold (10~30) at day, 
float BlobNCC_Th = 0.5;                       // blob NCC threshold <0.5 means no BG

bool isWriteToFile = false;

LaneDirection ldirection = LD_NORTH; // vertical lane
BgSubType bgsubtype = BGS_DIF;

// template matching algorithm implementation, demo

bool use_mask = false;
int match_method = cv::TM_CCOEFF_NORMED;
int max_Trackbar = 5;

// dnn -based approach 
// Initialize the parameters
float confThreshold = 0.1; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidthorg = 52;
int inpHeightorg = 37;
int inpWidth = (inpWidthorg) % 32 == 0 ? inpWidthorg : inpWidthorg + (32 - (inpWidthorg % 32));// 1280 / 4;  //min(416, int(1280. / 720.*64.*3.) + 1);// 160; //1920 / 2;// 416;  // Width of network's input image
int inpHeight = (inpHeightorg) % 32 == 0 ? inpHeightorg : inpHeightorg + (32 - (inpHeightorg % 32));// 720 / 4;  //64;// 160;// 1080 / 2;// 416; // Height of network's input image
vector<string> classes;

void getPredicInfo(const vector<Mat>& outs, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes);
regions_t DetectInCrop(Net& net, cv::Mat& colorMat, cv::Size crop, vector<Mat>& outs);
cv::Size adjustNetworkInputSize(Size inSize);
// dnn-based approach ends
// cascade detector related
void detectCascadeRoi(cv::Mat img, cv::Rect& rect);
void detectCascadeRoiVehicle(/* put config file */cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _cars);
void detectCascadeRoiHuman(/* put config file */cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _people);
// cascade detector related ends
// raod configuration related
float camera_height = 11.0 * 100; // camera height 11 meter
float lane_length = 200.0 * 100;  // lane length
float lane2lane_width = 3.5 * 3 * 100; // lane width
cv::Mat transmtxH;

int main(void) {
#ifdef _sk_Memory_Leakag_Detector
#if _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif	
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	int trackId = 0;          // unique object id
	int showId = 0;           //     


	cv::VideoCapture capVideo;

	cv::Mat imgFrame1;
	cv::Mat imgFrame2;

  float scaleFactor = .5;


	std::vector<Blob> blobs;

	cv::Point crossingLine[2];

	int carCount = 0;
	int truckCount = 0;
	int bikeCount = 0;
	int humanCount = 0;
	int videoLength = 0;
  
	loadConfig();
	bool b = capVideo.open(FAV1::VideoPath);  
  // load background image	
  cv::Mat BGImage = imread(FAV1::BGImagePath);
  
	//std::vector<Point> Road_ROI_Pts;
	//// relaxinghighwaytraffic.mp4
	//Road_ROI_Pts.push_back(Point(380, 194)*scaleFactor);
	//Road_ROI_Pts.push_back(Point(433, 194)*scaleFactor);
	//Road_ROI_Pts.push_back(Point(344, 423)*scaleFactor);
	//Road_ROI_Pts.push_back(Point(89, 423)*scaleFactor);

  std::vector<Point> road_roi_pts;
  std::vector<std::vector<Point>> Road_ROI_Pts; // sidewalks and carlanes
  // relaxinghighwaytraffic.mp4 for new one
  //// side walk1
  //road_roi_pts.push_back(Point(387.00,  189.00)*scaleFactor);
  //road_roi_pts.push_back(Point(392.00,  189.00)*scaleFactor);
  //road_roi_pts.push_back(Point(14.00,   479.00)*scaleFactor);
  //road_roi_pts.push_back(Point(4.00,    378.00)*scaleFactor);
  //Road_ROI_Pts.push_back(road_roi_pts);
  //road_roi_pts.clear();
  //// car lane
  //road_roi_pts.push_back(Point(391.00, 189.00)*scaleFactor);
  //road_roi_pts.push_back(Point(483.00, 187.00)*scaleFactor);
  //road_roi_pts.push_back(Point(797.00, 478.00)*scaleFactor);
  //road_roi_pts.push_back(Point(13.00, 478.00)*scaleFactor);
  //Road_ROI_Pts.push_back(road_roi_pts);
  //road_roi_pts.clear();
  //// side walk2
  //road_roi_pts.push_back(Point(480.00, 187.00)*scaleFactor);
  //road_roi_pts.push_back(Point(496.00, 187.00)*scaleFactor);
  //road_roi_pts.push_back(Point(853.00, 419.00)*scaleFactor);
  //road_roi_pts.push_back(Point(791.00, 478.00)*scaleFactor);
  //Road_ROI_Pts.push_back(road_roi_pts);
  //road_roi_pts.clear();

  ////20180911_172930_
  //// side walk1
  //road_roi_pts.push_back(Point(1057.25, 83.75)*scaleFactor);
  //road_roi_pts.push_back(Point(1069.25, 92.75)*scaleFactor);
  //road_roi_pts.push_back(Point(289.25, 1036.25)*scaleFactor);
  //road_roi_pts.push_back(Point(103.25, 1009.25)*scaleFactor);
  //Road_ROI_Pts.push_back(road_roi_pts);
  //road_roi_pts.clear();
  //// car lane
  //road_roi_pts.push_back(Point(1063.25, 89.75)*scaleFactor);
  //road_roi_pts.push_back(Point(1135.25, 92.75)*scaleFactor);
  //road_roi_pts.push_back(Point(955.25, 1033.25)*scaleFactor);
  //road_roi_pts.push_back(Point(253.25, 1039.25)*scaleFactor);
  //Road_ROI_Pts.push_back(road_roi_pts);
  //road_roi_pts.clear();
  //// side walk2
  //road_roi_pts.push_back(Point(1129.25, 91.25)*scaleFactor);
  //road_roi_pts.push_back(Point(1157.75, 95.75)*scaleFactor);
  //road_roi_pts.push_back(Point(1231.25, 1034.75)*scaleFactor);
  //road_roi_pts.push_back(Point(941.75, 1031.75)*scaleFactor);
  //Road_ROI_Pts.push_back(road_roi_pts);
  //road_roi_pts.clear();

  //20180912_112338
  // side walk1
  road_roi_pts.push_back(Point(932.75, 100.25)*scaleFactor);
  road_roi_pts.push_back(Point(952.25, 106.25)*scaleFactor);
  road_roi_pts.push_back(Point(434.75, 1055.75)*scaleFactor);
  road_roi_pts.push_back(Point(235.25, 1054.25)*scaleFactor);
  Road_ROI_Pts.push_back(road_roi_pts);
  road_roi_pts.clear();
  // car lane
  road_roi_pts.push_back(Point(949.25, 104.75)*scaleFactor);
  road_roi_pts.push_back(Point(1015.25, 103.25)*scaleFactor);
  road_roi_pts.push_back(Point(1105.25, 1048.25)*scaleFactor);
  road_roi_pts.push_back(Point(416.75, 1057.25)*scaleFactor);
  Road_ROI_Pts.push_back(road_roi_pts);
  road_roi_pts.clear();
  // side walk2
  road_roi_pts.push_back(Point(1009.25, 101.75)*scaleFactor);
  road_roi_pts.push_back(Point(1045.25, 98.75)*scaleFactor);
  road_roi_pts.push_back(Point(1397.75, 1052.75)*scaleFactor);
  road_roi_pts.push_back(Point(1087.25, 1049.75)*scaleFactor);
  Road_ROI_Pts.push_back(road_roi_pts);
  road_roi_pts.clear();
  // object size LUT config
  // sedan w
  // seda h 
  vector<float> sedan_h = { -0.00004444328872f, 0.01751602326f, -2.293443176f, 112.527668f }; // scale factor 0.5
  vector<float> sedan_w = { -0.00003734137716f, 0.01448943505f, -1.902199174f, 98.56691135f };
  vector<float> suv_h = { -0.00005815785621f, 0.02216859672f, -2.797603666f, 139.0638999f };
  vector<float> suv_w = { -0.00004854032314f, 0.01884736545f, -2.425686251f, 121.9226426f };
  vector<float> truck_h = { -0.00006123592908f, 0.02373661426f, -3.064585294f, 149.6535855f };
  vector<float> truck_w = { -0.00003778247771f, 0.015239317f, -2.091105041f, 110.7544702f };
  vector<float> human_h = { -0.000002473245036f, 0.001813179193f, -0.5058008988f, 49.27950311f };
  vector<float> human_w = { -0.000003459461125f, 0.001590306464f, -0.3208648543f, 28.23621306f };
  ITMSPolyValues polyvalue_sedan_h(sedan_h, sedan_h.size());
  ITMSPolyValues polyvalue_sedan_w(sedan_w, sedan_w.size());
  ITMSPolyValues polyvalue_suv_h(suv_h, suv_h.size());
  ITMSPolyValues polyvalue_suv_w(suv_w, suv_w.size());
  ITMSPolyValues polyvalue_truck_h(truck_h, truck_h.size());
  ITMSPolyValues polyvalue_truck_w(truck_w, truck_w.size());
  ITMSPolyValues polyvalue_human_h(human_h, human_h.size());
  ITMSPolyValues polyvalue_human_w(human_w, human_w.size());
  float value = polyvalue_sedan_w.getPolyValue(10.5);

  //absolute coordinator unit( pixel to centimeters) using Homography pp = H*p  
  //float camera_height = 11.0 * 100; // camera height 11 meter
  //float lane_length = 200.0 * 100;  // lane length
  //float lane2lane_width = 3.5 * 3* 100; // lane width
  std::vector<cv::Point2f> srcPts; // skewed ROI source points
  std::vector<cv::Point2f> tgtPts; // deskewed reference ROI points (rectangular. Top-to-bottom representation but, should be bottom-to-top measure in practice

  srcPts.push_back(Point2f(949.25, 104.75)*scaleFactor); // detect region left-top p0
  srcPts.push_back(Point2f(1045.25, 98.75)*scaleFactor); // detect region right-top p1
  srcPts.push_back(Point2f(1397.75, 1052.75)*scaleFactor); // detect region right-bottom p2 
  srcPts.push_back(Point2f(416.75, 1057.25)*scaleFactor); // detect region left-bottom  p3

  tgtPts.push_back(Point2f(0, 0));                        // pp0
  tgtPts.push_back(Point2f(lane2lane_width,0));           // pp1
  tgtPts.push_back(Point2f(lane2lane_width, lane_length));// pp2
  tgtPts.push_back(Point2f(0, lane_length));              // pp3

  transmtxH = cv::getPerspectiveTransform(srcPts, tgtPts); // homography
  
  // --------------  confirmation test  -----------------------------------------------------------------
  // get distance in Meters from the image locations according to the predefined ROI 
  // get float distanceInMeterFromPixels(std::vector<cv::Point2f> &srcPx, HtestPx, cv::Mat &trasmtx /* 3x3*/) 
  
  if (debugGeneralDetail) {
	  cout << " transfromation matrix H: " << transmtxH << endl;	  
	  std::vector<cv::Point2f> testPx, HtestPx;
	  testPx.push_back(Point2f(1000, 125)*scaleFactor);	  
	  cv::perspectiveTransform(testPx, HtestPx, transmtxH);
	  // test part if the given concept is working or not
	  // one point mapping to find the real distance 
	  cout << " transformed point from " << testPx << " to " << cv::Point(HtestPx.back()) << endl;
	  cout << " the distance (meter) from the start point (bottom line): " << (lane_length - round(HtestPx.back().y)) / 100. << " meters" << endl;
	  // multi point mapping test
	  cv::perspectiveTransform(srcPts, tgtPts, transmtxH);
	  cout << " src Points:\n" << srcPts << endl;
	  cout << " to : \n" << tgtPts << endl;
	  // ----------------------------- confirmation test completed --------------------------------------
  }
  // sangkny test : distance
  /*float distance = 0;
  std::vector<cv::Point2f> testPx;
  testPx.push_back(Point2f(1000, 125)*scaleFactor);
  distance = getDistanceInMeterFromPixels(testPx, transmtxH, lane_length, false);
  cout << " distance: " << distance / 100 << " meters from the starting point.\n";*/
  // define the case cade detector
  std::string runtime_data_dir1 = "D:/LectureSSD_rescue/project-related/도로-기상-유고-토페스/code/ITMS/";
  std::string xmlFile = runtime_data_dir1 + "config/cascade.xml"; // cars.xml with 1 neighbors good, cascade.xml with 5 neighbors, people cascadG.xml(too many PA) with 4 neighbors and size(30,80), size(80,200)
  if (!cascade.load(xmlFile)) {
	  std::cout << "Plase check the xml file in the given location !!(!)\n";
	  std::cout << xmlFile << std::endl;
	  return 0;
  }  
  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
  

	if (!capVideo.isOpened()) {                                                 // if unable to open video file
		std::cout << "error reading video file" << std::endl << std::endl;      // show error message
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
	/* Fast BSA */ // by sangkny
	//if(bgsubtype == BGS_CNT){ // type selection
		Ptr<BackgroundSubtractor> pBgSub;
		//pBgSub = cv::bgsubcnt::createBackgroundSubtractorCNT(fps, true, fps * 60);
		pBgSub = createBackgroundSubtractorMOG2();
	//}

    capVideo.read(imgFrame1);    
    capVideo.read(imgFrame2);
    if (imgFrame1.empty() || imgFrame2.empty())
      return 0;
    resize(imgFrame1, imgFrame1, Size(), scaleFactor, scaleFactor);
    resize(imgFrame2, imgFrame2, Size(), scaleFactor, scaleFactor);
    
    // background image // gray
    if (!BGImage.empty()) {
      resize(BGImage, BGImage, Size(), scaleFactor, scaleFactor);
      if (BGImage.channels() > 1)
        cv::cvtColor(BGImage, BGImage, CV_BGR2GRAY);
	}
	else {
		cout << "Background image is not selected. Please check this out (!)(!)\n";
		BGImage = imgFrame1.clone();
		//resize(BGImage, BGImage, Size(), scaleFactor, scaleFactor);
		if (BGImage.channels() > 1)
			cv::cvtColor(BGImage, BGImage, CV_BGR2GRAY);
	}
    if (debugShowImages && debugShowImagesDetail) {
      imshow("BGImage", BGImage);      
    }


    /*int intHorizontalLinePosition = (int)std::round((double)imgFrame1.rows * 0.5);
    crossingLine[0].x = 0;
    crossingLine[0].y = intHorizontalLinePosition;

    crossingLine[1].x = imgFrame1.cols - 1;
    crossingLine[1].y = intHorizontalLinePosition;*/

    int intHorizontalLinePosition = (int)std::round((double)imgFrame1.rows * 0.5);

    crossingLine[0].x = imgFrame1.cols * FAV1::StartX;
    crossingLine[0].y = imgFrame1.rows * FAV1::StartY;

    crossingLine[1].x = imgFrame1.cols * FAV1::EndX;
    crossingLine[1].y = imgFrame1.rows * FAV1::EndY;


    char chCheckForEscKey = 0;

    bool blnFirstFrame = true;

    int frameCount = 2;

    // Video save start
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

    cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

    // road mask generation
    cv::Mat road_mask = cv::Mat::zeros(imgFrame1.size(), imgFrame1.type());
    for(int ir=0; ir<Road_ROI_Pts.size(); ir++)
      fillConvexPoly(road_mask, Road_ROI_Pts.at(ir).data(), Road_ROI_Pts.at(ir).size(), Scalar(255, 255, 255), 8);
    
    if (road_mask.channels() > 1)
      cvtColor(road_mask, road_mask, CV_BGR2GRAY);
    if (0&& debugShowImages && debugShowImagesDetail) {
      imshow("road mask", road_mask);
      waitKey(1);
    }
	
	////// template matching algorithm implementation, test code
	//
	//cv::Rect roiRect(200, 200, 50, 50);
	//cv::Mat img = imgFrame1(roiRect);
	//
	//cvtColor(img, img, CV_BGR2GRAY);
	//cv::Mat templ = img.clone()(Rect(0,0, 50,50));

	//float NCC = getNCC(img, templ, Mat(), match_method, use_mask);
	//
	//// end template matching algorithm

  // Deep learning based Detection and Classification //
    std::string runtime_data_dir = "D:/LectureSSD_rescue/project-related/도로-기상-유고-토페스/code/Multitarget-tracker-master/data/";
    string classesFile = runtime_data_dir + "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

	// dnn-based approach
    // Give the configuration and weight files for the model
    String modelConfiguration = runtime_data_dir + "yolov3-tiny.cfg"; // was yolov3.cfg
    String modelWeights = runtime_data_dir + "yolov3-tiny.weights";   // was ylov3.weights
                                                                      //String modelConfiguration = runtime_data_dir + "tiny-yolo.cfg"; // was 
                                                                      //String modelWeights = runtime_data_dir + "tiny-yolo.weights";   // was 
                                                                      // Load the network
	// sangkny YOLO test
    /*
	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
	*/
	
    while (capVideo.isOpened() && chCheckForEscKey != 27) {

		double t1 = (double)cvGetTickCount();
        std::vector<Blob> currentFrameBlobs;

        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();

        cv::Mat imgDifference;
        cv::Mat imgThresh;

        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0); // was 5x5
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);
		if (bgsubtype == BGS_CNT) {
			pBgSub->apply(imgFrame2Copy, imgDifference);
		  if(debugShowImages && debugShowImagesDetail){
			Mat bgImage = Mat::zeros(imgFrame2Copy.size(), imgFrame2Copy.type());
			pBgSub->getBackgroundImage(bgImage);
			cv::imshow("backgroundImage", bgImage);
			if (isWriteToFile && frameCount == 200) {
			  string filename = FAV1::VideoPath;
			  filename.append("_"+to_string(scaleFactor)+"x.jpg");
			  cv::imwrite(filename, bgImage);
			  std::cout << " background image has been generated (!!)\n";
			}
		  }
			// only shadow part
			//cv::threshold(imgDifference, imgDifference, 125, 255, cv::THRESH_BINARY);
		}
		else {
			cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);
		}
    // roi applied
    if (!road_mask.empty()) {
      bitwise_and(road_mask, imgDifference, imgDifference);
    }
        cv::threshold(imgDifference, imgThresh, img_dif_th, 255.0, CV_THRESH_BINARY);
		if (debugShowImages && debugShowImagesDetail) {
			cv::imshow("imgThresh", imgThresh);
			cv::waitKey(1);
		}        

        for (unsigned int i = 0; i < 1; i++) {
			if(bgsubtype == BgSubType::BGS_CNT)
				cv::erode(imgThresh, imgThresh, structuringElement3x3);
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
			if (bgsubtype == BgSubType::BGS_DIF)
				cv::erode(imgThresh, imgThresh, structuringElement5x5);      
          //cv::morphologyEx(imgThresh, imgThresh, CV_MOP_CLOSE, structuringElement7x7);
        }

        cv::Mat imgThreshCopy = imgThresh.clone();

        std::vector<std::vector<cv::Point> > contours;

        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		if (debugShowImages && debugShowImagesDetail) {
			drawAndShowContours(imgThresh.size(), contours, "imgContours");
		}
        
		std::vector<std::vector<cv::Point> > convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

		if (debugShowImages && debugShowImagesDetail) {
			drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");
		}

        for (auto &convexHull : convexHulls) {
            Blob possibleBlob(convexHull);
            
            if (possibleBlob.currentBoundingRect.area() > 10 &&
              possibleBlob.dblCurrentAspectRatio > 0.2 &&
              possibleBlob.dblCurrentAspectRatio < 6.0 &&
              possibleBlob.currentBoundingRect.width > 6 &&
              possibleBlob.currentBoundingRect.height > 6 &&
              possibleBlob.dblCurrentDiagonalSize > 3.0 &&
              (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
              //  new approach according to 
              // 1. distance, 2. correlation within certain range
				  std::vector<cv::Point2f> blob_ntPts;
				  blob_ntPts.push_back(Point2f(possibleBlob.centerPositions.back()));
				  float realDistance = getDistanceInMeterFromPixels(blob_ntPts, transmtxH, lane_length, false);
				  cv::Rect roi_rect = possibleBlob.currentBoundingRect;
				  float blobncc = 0;
				  if (debugGeneral) {
					cout << "Candidate object:" << blob_ntPts.back() << "(W,H)" << cv::Size(roi_rect.width, roi_rect.height) << " is in(" << to_string(realDistance / 100.) << ") Meters ~(**)\n";
				  }
				  // bg image
				  // currnt image

				  /*imshow("bgimage", BGImage(roi_rect));
				  imshow("blob image_roi", imgFrame2Copy(roi_rect));
				  waitKey(0);*/
				  blobncc = getNCC(BGImage(roi_rect), imgFrame2Copy(roi_rect), Mat(), match_method, use_mask);
				  double d3 = matchShapes(BGImage(roi_rect), imgFrame2Copy(roi_rect), CONTOURS_MATCH_I3, 0);
				  if (realDistance >= 100 && realDistance <= 19900/* distance constraint */ && blobncc <= abs(BlobNCC_Th)) {// check the correlation with bgground, object detection/classification
		//            regions_t tempRegion;
		//            vector<Mat> outMat;                
					/*float scaleRect = 1.5;
					Rect expRect = expandRect(roi_rect, scaleRect*roi_rect.width, scaleRect*roi_rect.height, imgFrame2.cols, imgFrame2.rows);*/
					//Rect expRect = maxSqExpandRect(roi_rect, scaleRect, imgFrame2Copy.cols, imgFrame2Copy.rows);
		//            /*tempRegion = DetectInCrop(net, imgFrame2(expRect), adjustNetworkInputSize(Size(max(416, min(416, expRect.width*2)), max(416, min(416, expRect.height*2)))), outMat);
		//            if (tempRegion.size() > 0) {
		//              for (int tr = 0; tr < tempRegion.size(); tr++)
		//                cout << "=========> (!)(!) class:" << tempRegion[tr].m_type << ", prob:" << tempRegion[tr].m_confidence << endl;
		//            }*/
					////detectCascadeRoi(imgFrame2Copy, expRect); // detect both cars and humans
					//vector<Rect> Cars, Humans;
					//detectCascadeRoiVehicle(imgFrame2Copy, expRect, Cars); // detect cars only
					//detectCascadeRoiHuman(imgFrame2Copy, expRect, Humans); // detect people only
					//if(debugGeneralDetail && Cars.size())
					//	cout<< " ==>>>> Car is detected !!"<<endl;
					//if(debugGeneralDetail && Humans.size())
					//	cout << " ==>>>> Human is detected !!" << endl;
					// classify and put it to the blobls
					ObjectClass objclass;
					float classProb = 0.f;
					classifyObjectWithDistanceRatio(possibleBlob, realDistance / 100, objclass, classProb);
					// update the blob info and add to the existing blobs according to the classifyObjectWithDistanceRatio function output
					// verify the object with cascade object detection
					if(classProb > 0.99 /* 1.0 */){
						currentFrameBlobs.push_back(possibleBlob);
					}
					else if (classProb>0.5f) {
						// check with a ML-based approach
						float scaleRect = 1.5;
						Rect expRect = expandRect(roi_rect, scaleRect*roi_rect.width, scaleRect*roi_rect.height, imgFrame2Copy.cols, imgFrame2Copy.rows);
					
						if (possibleBlob.oc == itms::ObjectClass::OC_VEHICLE) {
							// verify it
							std::vector<cv::Rect> cars;
							detectCascadeRoiVehicle(imgFrame2Copy, expRect, cars);
							if (cars.size())
								possibleBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
							else
								continue;
						}
						else if (possibleBlob.oc == itms::ObjectClass::OC_HUMAN) {
							// verify it
							std::vector<cv::Rect> people;
							detectCascadeRoiHuman(imgFrame2Copy, expRect, people);
							if (people.size())
								possibleBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
							else
								continue;
						}
						else {// should not com in this loop (OC_OTHER)
							int kkk = 0;
						}
						currentFrameBlobs.push_back(possibleBlob);
					}
            
				}
			}
        }

		if (debugShowImages && debugShowImagesDetail) {
			// all of the currentFrameBlobs at this stage have 1 visible count yet. 
			drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");
		}
		// merge assuming
		// blobs are in the ROI because of ROI map
		// 남북 이동시는 가로가 세로보다 커야 한다.
		//     
		mergeBlobsInCurrentFrameBlobs(currentFrameBlobs); // need to consider the distance
		if (debugShowImages && debugShowImagesDetail) {
		  drawAndShowContours(imgThresh.size(), currentFrameBlobs, "after merging currentFrameBlobs");
		  waitKey(1);
		}
        if (blnFirstFrame == true) {
            for (auto &currentFrameBlob : currentFrameBlobs) {
                blobs.push_back(currentFrameBlob);
            }
        } else {
            matchCurrentFrameBlobsToExistingBlobs(imgFrame2Copy, blobs, currentFrameBlobs, trackId);
        }
		imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above
		if (debugShowImages ) {
			if(debugShowImagesDetail)
				drawAndShowContours(imgThresh.size(), blobs, "All imgBlobs");
		
			drawBlobInfoOnImage(blobs, imgFrame2Copy);  // blob(tracked) information
		}    
        //bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, intHorizontalLinePosition, carCount);
        bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, imgFrame2Copy, crossingLine[0], crossingLine[1],carCount, truckCount, bikeCount);

        if (blnAtLeastOneBlobCrossedTheLine == true) {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
        } else {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
        }
		if (debugShowImages) {
			drawCarCountOnImage(carCount, imgFrame2Copy);
			cv::imshow("imgFrame2Copy", imgFrame2Copy);
			cv::waitKey(1);
		}        

        // now we prepare for the next iteration

        currentFrameBlobs.clear();

        imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
            capVideo.read(imgFrame2);            
            resize(imgFrame2, imgFrame2, Size(), scaleFactor, scaleFactor);
            if (imgFrame2.empty()) {
              std::cout << "The input image is empty!! Please check the video file!!" << std::endl;
              _getch();
              break;
            }
        } else {
            std::cout << "end of video\n";
            break;
        }

        blnFirstFrame = false;
        frameCount++;
        chCheckForEscKey = cv::waitKey(1);
		if (debugTime) {
			double t2 = (double)cvGetTickCount();
			double t3 = (t2 - t1) / (double)getTickFrequency();
			cout << "Processing time>>  #:" << (frameCount - 1) <<"/("<< max_frames<<")"<< " --> " << t3*1000.0<<"msec, "<< 1./t3 << "fps \n";
		}
    }

    if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
        cv::waitKey(10000);                         // hold the windows open to allow the "end of video" message to show
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
void copyBlob2Blob(Blob &srcBlob, Blob &tgtBlob) {

  tgtBlob.currentContour.clear();
  for (int i = 0; i < srcBlob.currentContour.size(); i++)
    tgtBlob.currentContour.push_back(srcBlob.currentContour.at(i));

  tgtBlob.currentBoundingRect = srcBlob.currentBoundingRect;
  
  tgtBlob.centerPositions.clear();
  for(int i=0; i<srcBlob.centerPositions.size(); i++)
	tgtBlob.centerPositions.push_back(srcBlob.centerPositions.at(i));

  
  tgtBlob.dblCurrentDiagonalSize = srcBlob.dblCurrentDiagonalSize;

  tgtBlob.dblCurrentAspectRatio = srcBlob.dblCurrentAspectRatio;

  tgtBlob.blnStillBeingTracked = srcBlob.blnStillBeingTracked;
  tgtBlob.blnCurrentMatchFoundOrNewBlob = srcBlob.blnCurrentMatchFoundOrNewBlob;

  tgtBlob.intNumOfConsecutiveFramesWithoutAMatch = srcBlob.intNumOfConsecutiveFramesWithoutAMatch;

  tgtBlob.age = srcBlob.age;
  tgtBlob.totalVisibleCount = srcBlob.totalVisibleCount;
  tgtBlob.showId = srcBlob.showId;
  // object status information
  tgtBlob.oc = srcBlob.oc;
  tgtBlob.os = srcBlob.os;
  tgtBlob.od = srcBlob.od; // lane direction will affect the result, and the lane direction will be given
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void mergeBlobsInCurrentFrameBlobs(std::vector<Blob> &currentFrameBlobs) {
  std::vector<Blob>::iterator currentBlob = currentFrameBlobs.begin();
  while (currentBlob != currentFrameBlobs.end()) {
    int intIndexOfLeastDistance = -1;
    double dblLeastDistance = 100000.0;
    for (unsigned int i = 0; i < currentFrameBlobs.size(); i++) {     

      double dblDistance = distanceBetweenPoints(currentBlob->centerPositions.back(), currentFrameBlobs[i].centerPositions.back());      
      
      if (dblDistance > 1/* same object */ &&  dblDistance < dblLeastDistance) { // center locations should be in the range
        dblLeastDistance = dblDistance;
        intIndexOfLeastDistance = i;
      }
    }
    // at this point we have nearest countours
    if (intIndexOfLeastDistance < 0) {
      ++currentBlob;
      continue;
    }
    
    // check the conditions
    if (dblLeastDistance < currentBlob->dblCurrentDiagonalSize*1.25/*should be car size */) {
      cv::Rect cFBrect = currentFrameBlobs[intIndexOfLeastDistance].currentBoundingRect;
      Point cB = currentBlob->centerPositions.back();
      bool flagMerge = false;
      if (ldirection == LD_EAST || ldirection == LD_WEST /* horizontal*/) {
        if ((cB.y >= cFBrect.y - (round)((float)(cFBrect.height) / 2.)) && (cB.y <= cFBrect.y + (round)((float)(cFBrect.height) / 2.)))
          flagMerge = true;
      }
      else { // other lane direction only considers width and its center point
        if ((cB.x >= cFBrect.x - (round)((float)(cFBrect.width) / 2.)) && (cB.x <= cFBrect.x + (round)((float)(cFBrect.width) / 2.)))
          flagMerge = true;
      }
      // merge and erase index blob
      if (flagMerge) {
        if(debugGeneral)
          cout << "mergeing with " << to_string(intIndexOfLeastDistance)<<" in blob" <<currentBlob->centerPositions.back() << endl;

        // countour merging
        std::vector<cv::Point> points, contour;
        points.insert(points.end(), currentBlob->currentContour.begin(), currentBlob->currentContour.end());
        points.insert(points.end(), currentFrameBlobs[intIndexOfLeastDistance].currentContour.begin(), currentFrameBlobs[intIndexOfLeastDistance].currentContour.end());
        convexHull(cv::Mat(points), contour);
        itms::Blob blob(contour);
        *currentBlob = blob;      // class Blob = operator overloading
        //copyBlob2Blob(blob, *currentBlob);
        std::vector<Blob>::iterator tempBlob = currentFrameBlobs.begin();
		currentBlob = currentFrameBlobs.erase(tempBlob+intIndexOfLeastDistance);       
		// do something after real merge
		continue;
      }    
      
    }    
      ++currentBlob;    
  }
  
 
}
///////////////////////////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(cv::Mat& srcImg, std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, int &id){
	std::vector<Blob>::iterator existingBlob = existingBlobs.begin();
	while ( existingBlob != existingBlobs.end()) {
			// check if a block is too old after disappeared in the screen
		if (existingBlob->blnStillBeingTracked == false 
				&& existingBlob->intNumOfConsecutiveFramesWithoutAMatch>=maxNumOfConsecutiveInvisibleCounts) { 
				// removing a blob from the list of existingBlobs						
			  // overlapping test if overlapped or background, we will erase.
			  std::vector<pair<int, int>> overlappedBobPair; // first:blob index, second:overlapped type
			  int intIndexOfOverlapped = -1;
			  double dblLeastDistance = 100000.0;
			  for (unsigned int i = 0; i < existingBlobs.size(); i++) {
				int intSect = InterSectionRect(existingBlob->currentBoundingRect, existingBlobs[i].currentBoundingRect);

				if (intSect >= 1 && existingBlobs[i].blnStillBeingTracked) { // deserted blob will be eliminated, 1, 2, 3 indicate the inclusion of one block in another
				  overlappedBobPair.push_back(std::pair<int, int>(i, intSect));
				}
			  }
			// erase the blob
			if (overlappedBobPair.size() > 0) { // how about itself ? 
			  if (debugTrace /*&& debugGeneralDetail*/) {
				cout << " (!)==> Old and deserted blob id: " << existingBlob->id << " is eliminated at " << existingBlob->centerPositions.back() << Size(existingBlob->currentBoundingRect.width, existingBlob->currentBoundingRect.height)<<"\n(blobs Capacity: " << existingBlobs.capacity() << ")" << endl;
				cout << "cpt size: " << existingBlob->centerPositions.size() << endl;
				cout << "age: " << existingBlob->age << endl;
				cout << "totalVisible #: " << existingBlob->totalVisibleCount << endl;
			  }
			  existingBlob = existingBlobs.erase(existingBlob);
			}else{ // partial(0) or no overlapped (-1)
			  // conditional elimination
			  if (debugTrace /* && debugGeneralDetail*/) {
				cout << " (!)! Old blob id: " << existingBlob->id << " is conditionally eliminated at " << existingBlob->centerPositions.back() << Size(existingBlob->currentBoundingRect.width, existingBlob->currentBoundingRect.height) << "\n(blobs Capacity: " << existingBlobs.capacity() << ")" << endl;
				cout << "cpt size: " << existingBlob->centerPositions.size() << endl;
				cout << "age: " << existingBlob->age << endl;
				cout << "totalVisible #: " << existingBlob->totalVisibleCount << endl;
			  }
			  if (existingBlob->centerPositions.size() > maxNumOfConsecutiveInvisibleCounts / 2) { // current Blob was moving but now stopped.          
				// this blob needs to declare stopped object and erase : 전에 움직임이 있다가 현재는 없는 object임.
				// 아직은 center 값을 늘렸음....
				if (existingBlob->centerPositions.size() > maxNumOfConsecutiveInvisibleCounts) {
				  cout << "\n\n\n\n -------It--was moving--and --> stopped : object eliminated \n\n\n\n";
				  existingBlob = existingBlobs.erase(existingBlob);
				  continue;
				  //waitKey(0);
				} 
				// -------------------------------------------------------------
				// check this out : sangkny on 2018/12/17
				existingBlob->blnCurrentMatchFoundOrNewBlob = false;
				//existingBlob->centerPositions.push_back(existingBlob->centerPositions.back());
				existingBlob->predictNextPosition(); // can be removed if the above line is commented
				////existingBlob->os = getObjectStatusFromBlobCenters(*existingBlob, ldirection, movingThresholdInPixels, minVisibleCount); // 벡터로 넣을지 생각해 볼 것, update로 이전 2018. 10.25
				++existingBlob;
				continue; // can be removed this
				// ---------------------------------------------------------------
			
			  }else {
				existingBlob = existingBlobs.erase(existingBlob);
			  }
			}
		}else {
		  existingBlob->blnCurrentMatchFoundOrNewBlob = false;
		  existingBlob->predictNextPosition();
		  //existingBlob->os = getObjectStatusFromBlobCenters(*existingBlob, ldirection, movingThresholdInPixels, minVisibleCount); 
		  // 벡터로 넣을지 생각해 볼 것, update로 이전 2018. 10.25
		  ++existingBlob;
		}
	} // end while ( existingBlob != existingBlobs.end())
	/*for (auto &existingBlob : existingBlobs) {	
		existingBlob.blnCurrentMatchFoundOrNewBlob = false;
		existingBlob.predictNextPosition();
	}*/

  // candidate search only with distances between centers of currentFrameBlobs and existing blobs.
  // add more property including area and h/w ratio
  // serch around the nearest neighbor blob for tracking 
  // for searching larger area with more accuracy, we need to increase the search range (CurrentDiagonalSize) or to particle filter
  // with data, kalman or other tracking will be more accurate
	for (auto &currentFrameBlob : currentFrameBlobs) {
		int intIndexOfLeastDistance = 0;
		//int intIndexOfHighestScore = 0;
        double dblLeastDistance = 100000.0;
		double totalScore = 100.0, cutTotalScore = 20.0, maxTotalScore = 0.0;
		float allowedPercentage = 0.25; // 20%
		float minArea = currentFrameBlob.currentBoundingRect.width *(1.0f - allowedPercentage); // 편차가 너무 크므로 
		float MaxArea = currentFrameBlob.currentBoundingRect.width *(1.0f + allowedPercentage); // width 와 height로 구성
		float minDiagonal = currentFrameBlob.currentBoundingRect.height * (1.0f - allowedPercentage); // huMoment를 이용하는 방법 모색
		float maxDiagonal = currentFrameBlob.currentBoundingRect.height * (1.0f + allowedPercentage);

        for (unsigned int i = 0; i < existingBlobs.size(); i++) {
            if (existingBlobs[i].blnStillBeingTracked == true) { // find assigned tracks
                // it can be replaced with the tracking algorithm or assignment algorithm like KALMAN or Hungrian Assignment algorithm 
                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);
				totalScore -= dblDistance;
				totalScore -= (existingBlobs[i].currentBoundingRect.height < minDiagonal || existingBlobs[i].currentBoundingRect.height>maxDiagonal) ? 10 : 0;
				totalScore -= (existingBlobs[i].currentBoundingRect.width < minArea || existingBlobs[i].currentBoundingRect.width > MaxArea) ? 10 : 0;
                totalScore -= (abs(existingBlobs[i].currentBoundingRect.area() - currentFrameBlob.currentBoundingRect.area())/max(existingBlobs[i].currentBoundingRect.width, currentFrameBlob.currentBoundingRect.width));

				if (dblDistance < dblLeastDistance /* && (existingBlobs[i].oc == currentFrameBlob.oc)*/) {
					dblLeastDistance = dblDistance;
					intIndexOfLeastDistance = i;
				}
				/*if (maxTotalScore < totalScore) {
					maxTotalScore = totalScore;
					intIndexOfHighestScore = i;
                    dblLeastDistance = dblDistance;
				}*/
            }
            else { // existingBlobs[i].bInStillBeingTracked == false;
              /* do something for unassinged tracks */
              int temp = 0; // no meaning 
            }
        }
        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize ) { // 충분히 클수록 좋다.
			    addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
		    //  if(maxTotalScore>=cutTotalScore && dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5){
		      //	addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfHighestScore);
        }
        else { // this routine contains new and unassigned track(blob)s
          // add new blob
          vector<Point2f> blobCenterPxs;
          blobCenterPxs.push_back(currentFrameBlob.centerPositions.back());
          float distance = getDistanceInMeterFromPixels(blobCenterPxs, transmtxH, lane_length, false);
          if (debugGeneral)
            cout << " distance: " << distance / 100 << " meters from the starting point.\n";
          // do the inside
          if (distance >= 100.00/* 1m */ && distance < 19900/*199m*/) {// between 1 meter and 199 meters
            ObjectClass objclass;
            float classProb = 0.f;
            classifyObjectWithDistanceRatio(currentFrameBlob, distance / 100, objclass, classProb);
            // update the blob info and add to the existing blobs according to the classifyObjectWithDistanceRatio function output
			// verify the object with cascade object detection

            if(classProb>0.5f){
			  // check with a ML-based approach
				float scaleRect = 1.5;										// put it to the config parameters
				cv::Rect roi_rect(currentFrameBlob.currentBoundingRect);	// copy the current Rect and expand it
				cv::Rect expRect = expandRect(roi_rect, scaleRect*roi_rect.width, scaleRect*roi_rect.height, srcImg.cols, srcImg.rows);
				if (currentFrameBlob.oc == itms::ObjectClass::OC_VEHICLE) {
					// verify it
					std::vector<cv::Rect> cars;					
					detectCascadeRoiVehicle(srcImg, expRect, cars);
					if(cars.size())
						currentFrameBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
					else
						continue;
				}
				else if (currentFrameBlob.oc == itms::ObjectClass::OC_HUMAN) {
					// verify it
					std::vector<cv::Rect> people;
					detectCascadeRoiHuman(srcImg, expRect, people);
					if (people.size())
						currentFrameBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
					else
						continue;
				}
				else {// should not com in this loop (OC_OTHER)
				;
				}			  
				addNewBlob(currentFrameBlob, existingBlobs, id);
			  }
          }
            
        }

    }

    // update tracks 
    // 2018. 10. 25 getObjectStatusFromBlobCenters 을 전반부에서 이동.. 그리고, 각종 object status object classification을 여기서 함..

    for (auto &existingBlob : existingBlobs) { // update track routine

        if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) { // unassigned tracks
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
            existingBlob.age++;
        }
        else { // update the assigned (matched) tracks
          existingBlob.intNumOfConsecutiveFramesWithoutAMatch = 0; // reset because of appearance
          existingBlob.age++;
          existingBlob.totalVisibleCount++;
          //existingBlob.blnStillBeingTracked = (existingBlob.totalVisibleCount >= 5 )?  true: false;
        }

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= maxNumOfConsecutiveInFramesWithoutAMatch/* 1sec. it should be a predefined threshold */) {
            existingBlob.blnStillBeingTracked = false; /* still in the list of blobs */			
        }
        // object status, class update routine starts
        // object status
        existingBlob.os = getObjectStatusFromBlobCenters(existingBlob, ldirection, movingThresholdInPixels, minVisibleCount); // 벡터로 넣을지 생각해 볼 것
        // object classfication according to distance and width/height ratio, area 

        // object status, class update routine ends
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {
// update the status or correcte the status
	// here, we need to control the gradual change of object except the center point	
	float allowedPercentage = 1.0; // 20%
	float minArea = existingBlobs[intIndex].currentBoundingRect.area() *(1.0f - allowedPercentage);
	float MaxArea = existingBlobs[intIndex].currentBoundingRect.area() *(1.0f + allowedPercentage);
	float minDiagonal = existingBlobs[intIndex].dblCurrentDiagonalSize * (1.0f - allowedPercentage);
	float maxDiagonal = existingBlobs[intIndex].dblCurrentDiagonalSize * (1.0f + allowedPercentage);
	if (0&&(currentFrameBlob.currentBoundingRect.area() < minArea ||
		currentFrameBlob.currentBoundingRect.area() > MaxArea || 
		currentFrameBlob.dblCurrentDiagonalSize < minDiagonal ||
		currentFrameBlob.dblCurrentDiagonalSize > maxDiagonal)) {
		
		// if the given current frame blob's size is not propriate, we just move the center point of the existing blob
		// 2018. 10. 27
		// change the center point only to currentFrameBlob
		std::vector<cv::Point> newContourPts = existingBlobs[intIndex].currentContour;
		cv::Point cFBlobctrPt = currentFrameBlob.centerPositions.back();
		cv::Point extBlobctrPt = existingBlobs[intIndex].centerPositions.back();
		for(int i= 0; i<newContourPts.size(); i++)
			newContourPts[i] += (cFBlobctrPt - extBlobctrPt); // center point movement to existing Blob

		existingBlobs[intIndex].currentContour = newContourPts;
		//existingBlobs[intIndex].currentBoundingRect = cv::boundingRect(newContourPts); // actually it is same as existingBlobs[intIndex]

		existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

		//existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize; 
		//existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;
	}
	else {		
		// if the given current frame blob's size is not propriate, we just move the center point of the existing blob
		existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
		existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

		existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

		existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
		existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;
		// sangkny 2018. 12. 19
		// check this out one more time 
		existingBlobs[intIndex].oc = currentFrameBlob.oc;
	}

    //if (existingBlobs[intIndex].totalVisibleCount >= 8 /* it should be a predefined threshold */)
      existingBlobs[intIndex].blnStillBeingTracked = true; /* it is easy to be exposed to noise, so it put constraints to this */
   // else
   //   int kkk = 0;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
    // update the object class
    if (existingBlobs[intIndex].oc_prob < 0.5 || existingBlobs[intIndex].oc == ObjectClass::OC_OTHER) {
      // srcBlob-based classfication 2018. 12. 10
      vector<Point2f> blobCenterPxs;
      blobCenterPxs.push_back(currentFrameBlob.centerPositions.back());
      float distance = getDistanceInMeterFromPixels(blobCenterPxs, transmtxH, lane_length, false);
      if (debugGeneral)
        cout << " distance: " << distance / 100 << " meters from the starting point.\n";
      
      ObjectClass objclass;
      float classProb = 0.f;
      classifyObjectWithDistanceRatio(currentFrameBlob, distance / 100, objclass, classProb);
      // update the blob info 
      if (existingBlobs[intIndex].oc_prob < currentFrameBlob.oc_prob) {
        existingBlobs[intIndex].oc_prob = currentFrameBlob.oc_prob;
        existingBlobs[intIndex].oc = currentFrameBlob.oc;
      }

    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &id) {

  if (currentFrameBlob.totalVisibleCount > 1)
    int temp = 0;
    // check the size according to the distance from the starting point because NCC is already performed.

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;    
	  id = (id > 2048) ? 0 : ++id; // reset id according to the max number of type (int) or time (day or week)
    currentFrameBlob.id = id;    
    assert(currentFrameBlob.startPoint == currentFrameBlob.centerPositions.back()); // always true
    existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// it should inspect after predicting the next position
// it determines the status away at most 5 frame distance from past locations
// it shoud comes from the average but for efficiency, does come from the several past location. 
ObjectStatus getObjectStatusFromBlobCenters( Blob &blob, const LaneDirection &lanedirection, int movingThresholdInPixels, int minTotalVisibleCount) {
  ObjectStatus objectstatus=ObjectStatus::OS_NOTDETERMINED; 
  if (blob.totalVisibleCount < minTotalVisibleCount) // !! parameter
	  return objectstatus;
  
  int numPositions = (int)blob.centerPositions.size();
  //int maxNumPosition = 5;
  int bweightedAvg = -1; // -1: info from the past far away, 0: false (uniform average), 1: true (weighted average)
  Blob tmpBlob = blob;  
  // it will affect the speed because of const Blob declaration in parameters !!!!
  int deltaX;// = blob.predictedNextPosition.x - blob.centerPositions.back().x;
  int deltaY;// = blob.predictedNextPosition.y - blob.centerPositions.back().y; // have to use moving average after applying media filtering    
  cv::Point wpa = tmpBlob.weightedPositionAverage(bweightedAvg);
  deltaX = blob.predictedNextPosition.x - wpa.x;
  deltaY = blob.predictedNextPosition.y - wpa.y;
  
  switch (lanedirection)  { 
  case LD_SOUTH:
  case LD_NORTH:
    if (abs(deltaY) <= movingThresholdInPixels ) 
      objectstatus = OS_STOPPED;
    else { // moving anyway
      objectstatus = (lanedirection == LD_SOUTH)? (deltaY > 0 ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : (deltaY > 0 ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
    }
    break;

  case LD_EAST:
  case LD_WEST:
    if (abs(deltaX) <= movingThresholdInPixels) // 
      objectstatus = OS_STOPPED;
    else { // moving anyway
      objectstatus = (lanedirection == LD_EAST) ? (deltaX > 0 ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : (deltaX > 0 ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
    }
    break;

  case LD_NORTHEAST:
  case LD_SOUTHWEST:    
    if (abs(deltaX) + abs(deltaY) <= movingThresholdInPixels) // 
      objectstatus = OS_STOPPED;
    else { // moving anyway
      objectstatus = (lanedirection == LD_NORTHEAST) ? ((deltaX > 0 || deltaY < 0 )? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : ((deltaX > 0 || deltaY <0)? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
    }
    break;

  case LD_SOUTHEAST:
  case LD_NORTHWEST:  
    if (abs(deltaX) + abs(deltaY) <= movingThresholdInPixels) // 
      objectstatus = OS_STOPPED;
    else { // moving anyway
      objectstatus = (lanedirection == LD_SOUTHEAST) ? ((deltaX > 0 || deltaY > 0) ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : ((deltaX > 0 || deltaY >0) ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
    }
    break;

  default:
    objectstatus = OS_NOTDETERMINED;
    break;
  }

  // update object state and return the current estimated state (redundant because we already update the os status and update again later outside the function
  // 2018. 10. 25
  // 0. current os status will be previous status after update
 // weight policy : current status consecutive counter, others, 1
  // 2018. 10. 26 -> replaced with below functions
  
  itms::Blob orgBlob = blob; // backup 
  updateBlobProperties(blob, objectstatus); // update the blob properties including the os_prob
  //itms::ObjectStatus tmpOS = computeObjectStatusProbability(blob); // get the moving status according to the probability
  //if (tmpOS != objectstatus) {    
  //  objectstatus = tmpOS;
  //  // go back to original blob and update again correctly
  //  blob = orgBlob;
  //  updateBlobProperties(blob, objectstatus);
  //}
  

  return objectstatus;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

    int intX = (point1.x - point2.x);
    int intY = (point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
    cv::waitKey(1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    std::vector<std::vector<cv::Point> > contours, contours_bg;

    for (auto &blob : blobs) {
        if (blob.blnStillBeingTracked == true /*&& blob.totalVisibleCount>= minVisibleCount*/) {
            contours.push_back(blob.currentContour);
        }
		else{
			contours_bg.push_back(blob.currentContour);
		}
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);
	cv::drawContours(image, contours_bg, -1, SCALAR_RED, 2,8);

    cv::imshow(strImageName, image);
    cv::waitKey(1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount) {
    bool blnAtLeastOneBlobCrossedTheLine = false;

    for (auto blob : blobs) {

        if (blob.blnStillBeingTracked == true && blob.totalVisibleCount >= minVisibleCount && blob.centerPositions.size() >= 2) {
            int prevFrameIndex = (int)blob.centerPositions.size() - 2;
            int currFrameIndex = (int)blob.centerPositions.size() - 1;

            if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition) {
                carCount++;
                blnAtLeastOneBlobCrossedTheLine = true;
            }
        }

    }

    return blnAtLeastOneBlobCrossedTheLine;
}
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy, cv::Point Pt1, cv::Point Pt2, int &carCount, int &truckCount, int &bikeCount) {
  bool blnAtLeastOneBlobCrossedTheLine = false;

  for (auto blob : blobs) {

    if (blob.blnStillBeingTracked == true && blob.totalVisibleCount >= minVisibleCount && blob.centerPositions.size() >= 2) {
      int prevFrameIndex = (int)blob.centerPositions.size() - 2;
      int currFrameIndex = (int)blob.centerPositions.size() - 1;

      // Horizontal Line
      if (blob.centerPositions[currFrameIndex].x > Pt1.x  
        && blob.centerPositions[currFrameIndex].x < Pt2.x  
        &&  blob.centerPositions[prevFrameIndex].y < std::max(Pt2.y,Pt1.y) 
        && blob.centerPositions[currFrameIndex].y >= std::min(Pt1.y,Pt2.y)) 
      {
        carCount++;

#ifdef SHOW_STEPS
          cv::Mat crop = Mat::zeros(Size(blob.currentBoundingRect.width, blob.currentBoundingRect.height), imgFrame2Copy.type());
          crop = imgFrame2Copy(blob.currentBoundingRect).clone();
          cv::imwrite("D:\\sangkny\\dataset\\test.png", crop);
          cv::imshow("cropImage", crop);
          cv::waitKey(1);
          cout << "blob track id: " << blob.id << " is crossing the line." << endl;
          cout << "blob infor: (Age, totalSurvivalFrames, ShowId)-(" <<blob.age<<", "<< blob.totalVisibleCount << ", " << blob.showId << ")" << endl;
#endif
        if (cv::contourArea(blob.currentContour) > FAV1::truckArea) {
          truckCount++;
        }else if (cv::contourArea(blob.currentContour) > FAV1::bikeArea) {
          bikeCount++;
        }
        // implement later 
        //else if (cv::contourArea(blob.currentContour) > FAV1::humanArea) {
        //  humanCount++;
        //}

        blnAtLeastOneBlobCrossedTheLine = true;
      }
    }

  }

  return blnAtLeastOneBlobCrossedTheLine;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {

    for (unsigned int i = 0; i < blobs.size(); i++) {

        if (blobs[i].blnStillBeingTracked == true && blobs[i].totalVisibleCount >1) {
            cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_GREEN, 2);

            int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
            double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
            int intFontThickness = (int)std::round(dblFontScale * 1.0);
            string infostr, status;
            if (blobs[i].os == OS_STOPPED) {
              status = " STOP";
              cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_BLUE, 2);
            }
            else if (blobs[i].os == OS_MOVING_FORWARD)
              status = " MV";
            else if (blobs[i].os == OS_MOVING_BACKWARD) {
              status = " WWR"; // wrong way on a road
              cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);
            }
            else
              status = " ND"; // not determined

            infostr = std::to_string(blobs[i].id) + status + blobs[i].getBlobClass();// std::to_string(blobs[i].oc);
            cv::putText(imgFrame2Copy, infostr/*std::to_string(blobs[i].id)*/, blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
            if (debugTrace) {
              // draw the trace of object
              std::vector<cv::Point> centroids2= blobs[i].centerPositions;
              for (std::vector<cv::Point>::iterator it3 = centroids2.end()-1; it3 != centroids2.begin(); --it3)
              {
                cv::circle(imgFrame2Copy, cv::Point((*it3).x, (*it3).y), 3, SCALAR_YELLOW, -1); // draw the trace of the object with Yellow                                
                if (centroids2.end()-it3 > numberOfTracePoints)
                  break;
              }
              
            }

        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {

    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
    double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
    int intFontThickness = (int)std::round(dblFontScale * 1.5);

    cv::Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);

    cv::Point ptTextTopRightPosition;

    ptTextTopRightPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25);
    ptTextTopRightPosition.y = (int)((double)textSize.height * 1.25);

    cv::putText(imgFrame2Copy, std::to_string(carCount), ptTextTopRightPosition, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);

}

// utils
float getDistanceInMeterFromPixels(std::vector<cv::Point2f> &srcPx, cv::Mat &transmtx /* 3x3*/, float _laneLength, bool flagLaneDirectionTop2Bottom) {
	assert(transmtx.size() == cv::Size(3, 3));
	std::vector<cv::Point2f> H_Px;
	bool flagLaneDirection = flagLaneDirectionTop2Bottom;
	float laneLength = _laneLength, distance = 0;

	cv::perspectiveTransform(srcPx, H_Px, transmtx);
	distance = (flagLaneDirection)? round(H_Px.back().y): laneLength - round(H_Px.back().y);
	
	return distance;
}

float getNCC(cv::Mat &bgimg, cv::Mat &fgtempl, cv::Mat &fgmask, int match_method/* cv::TM_CCOEFF_NORMED*/, bool use_mask/*false*/) {
	//// template matching algorithm implementation, demo	
  if (debugGeneralDetail) {
    string ty = type2str(bgimg.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), bgimg.cols, bgimg.rows);
    ty = type2str(fgtempl.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), fgtempl.cols, fgtempl.rows);
  }
	assert(bgimg.type() == fgtempl.type());
	
	cv::Mat bgimg_gray,fgtempl_gray,res;
	float ncc;

	if (bgimg.channels() > 1) {
		cvtColor(bgimg, bgimg_gray, CV_BGR2GRAY);
		cvtColor(fgtempl, fgtempl_gray, CV_BGR2GRAY);
	} {
		bgimg_gray = bgimg.clone();
		fgtempl_gray = fgtempl.clone();
	}

	bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
	if (0&& debugShowImages && debugShowImagesDetail) {
		imshow("img", bgimg_gray);
		imshow("template image", fgtempl_gray);
		waitKey(1);
	}
	if (use_mask && method_accepts_mask)
	{
		matchTemplate(bgimg_gray, fgtempl_gray, res, match_method, fgmask);
	}
	else
	{
		matchTemplate(bgimg_gray, fgtempl_gray, res, match_method);
	}
	ncc = res.at<float>(0, 0); // should be float type [-1 1]

	if (debugGeneral) {
		cout << " NCC value is : " << res << endl;
		cout << " variable ncc: " << ncc << endl;
	}

	return ncc;
}

// call back functions 
//// template matching algorithm implementation, demo
//bool use_mask = false;
//Mat img; Mat templ; Mat result; Mat mask;
//char* image_window = "Source Image";
//char* result_window = "Result window";
//int match_method = cv::TM_CCOEFF_NORMED;
//int max_Trackbar = 5;
//void MatchingMethod(int, void*)
//{
//	Mat img_display;
//	img.copyTo(img_display);
//	int result_cols = img.cols - templ.cols + 1;
//	int result_rows = img.rows - templ.rows + 1;
//	result.create(result_rows, result_cols, CV_32FC1);
//	bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
//	if (use_mask && method_accepts_mask)
//	{
//		matchTemplate(img, templ, result, match_method, mask);
//	}
//	else
//	{
//		matchTemplate(img, templ, result, match_method);
//	}
//	
//	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
//	double minVal; double maxVal; Point minLoc; Point maxLoc;
//	Point matchLoc;
//	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
//	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
//	{
//		matchLoc = minLoc;
//	}
//	else
//	{
//		matchLoc = maxLoc;
//	}
//	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
//	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
//	imshow(image_window, img_display);
//	imshow(result_window, result);
//	return;
//}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
  case CV_8U:  r = "8U"; break;
  case CV_8S:  r = "8S"; break;
  case CV_16U: r = "16U"; break;
  case CV_16S: r = "16S"; break;
  case CV_32S: r = "32S"; break;
  case CV_32F: r = "32F"; break;
  case CV_64F: r = "64F"; break;
  default:     r = "User"; break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}
int InterSectionRect(cv::Rect &rect1, cv::Rect &rect2) {
  // -------------------------------------
  // returns intersection status when one is included to the other one
  // no intersection -1,
  // exist intersect 0 with a sub_region
  // rect1 includes rect2 1
  // rect2 includes rect1 2
  // rect1 == rect 2  3
  // -------------------------------------
  int retvalue = -1;
  
  cv::Rect intRect = (rect1 & rect2);
  bool intersects = (intRect.area() > 0); // intersection 
  if (intersects) {
    retvalue = 0;
	if (rect1.area() == rect2.area())
		retvalue = 3;
    else if (rect1.area() > rect2.area()) {
      if (rect2.area() == intRect.area())
        retvalue = 1;
    }
    else { // rect1 < rect2
      if (rect1.area() == intRect.area())
        retvalue = 2;
    }
  }    
  
  return retvalue;
}

// class Blob image processing blob_imp

void updateBlobProperties(itms::Blob &updateBlob, itms::ObjectStatus &curStatus) {
  itms::ObjectStatus prevOS = updateBlob.os;  // previous object status
  switch (curStatus) { // at this point, blob.os is the previous status !!!
  case OS_NOTDETERMINED:
    updateBlob.os_notdetermined_cnter++;
    updateBlob.os_NumOfConsecutiveStopped_cnter = 0;  // reset the consecutive counter
    updateBlob.os_NumOfConsecutivemvForward_cnter = 0;
    updateBlob.os_NumOfConsecutivemvBackward_cnter = 0;

    break;

  case OS_STOPPED:
    updateBlob.os_stopped_cnter++;
    updateBlob.os_NumOfConsecutiveStopped_cnter = (prevOS == OS_STOPPED) ? updateBlob.os_NumOfConsecutiveStopped_cnter + 1 : 1;
    //blob.os_NumOfConsecutiveStopped_cnter = 1;  // reset the consecutive counter
    updateBlob.os_NumOfConsecutivemvForward_cnter = 0;
    updateBlob.os_NumOfConsecutivemvBackward_cnter = 0;
    break;

  case OS_MOVING_FORWARD:
    updateBlob.os_mvForward_cnter++;
    updateBlob.os_NumOfConsecutivemvForward_cnter = (prevOS == OS_MOVING_FORWARD) ? updateBlob.os_NumOfConsecutivemvForward_cnter + 1 : 1;
    updateBlob.os_NumOfConsecutiveStopped_cnter = 0;  // reset the consecutive counter
                                                //blob.os_NumOfConsecutivemvForward_cnter = 1;
    updateBlob.os_NumOfConsecutivemvBackward_cnter = 0;
    break;

  case OS_MOVING_BACKWARD:
    updateBlob.os_mvBackward_cnter++;
    updateBlob.os_NumOfConsecutivemvBackward_cnter = (prevOS == OS_MOVING_BACKWARD) ? updateBlob.os_NumOfConsecutivemvBackward_cnter + 1 : 1;
    updateBlob.os_NumOfConsecutiveStopped_cnter = 0;  // reset the consecutive counter
    updateBlob.os_NumOfConsecutivemvForward_cnter = 0;
    //blob.os_NumOfConsecutivemvBackward_cnter = 1;
    break;

  defualt:
    // no nothing...
    cout << " object status is not correct!! inside updateBlobProperties \n";
    break;
  }
  // update the os_prob, it needs to improve
  updateBlob.os_pro = (float)updateBlob.totalVisibleCount / (float)max(1, updateBlob.age);

}
itms::ObjectStatus computeObjectStatusProbability(const itms::Blob &srcBlob) {
   //determine the object status with probability computations
  // weight policy : current status consecutive counter, others, 1
  float total_NumOfConsecutiveCounter = (1 + 1 + srcBlob.os_NumOfConsecutivemvBackward_cnter + srcBlob.os_NumOfConsecutivemvForward_cnter + srcBlob.os_NumOfConsecutiveStopped_cnter);
  float stop_prob = ((srcBlob.os_NumOfConsecutiveStopped_cnter+1)*srcBlob.os_stopped_cnter);
  float forward_prob = ((srcBlob.os_NumOfConsecutivemvForward_cnter+1)*srcBlob.os_mvForward_cnter);
  float backward_prob = ((srcBlob.os_NumOfConsecutivemvBackward_cnter+1)*srcBlob.os_mvBackward_cnter);
  float nondetermined_prob = 0; // dummy
  // find max vale
  float os_max = -1;
  int max_index = 0;
  vector<float> os_vector; // put the same order with ObjectStatus
  os_vector.push_back(stop_prob);
  os_vector.push_back(forward_prob);
  os_vector.push_back(backward_prob);
  os_vector.push_back(nondetermined_prob); // dummy
  assert((int)(ObjectStatus::OS_NOTDETERMINED+1) == (int)os_vector.size());  // size check!
  for (int i = 0; i < os_vector.size(); i++) {
   if (os_vector.at(i) > os_max) {
    os_max = os_vector.at(i);
    max_index = i;
   }
  }
  return  ObjectStatus(max_index);  
}

// dnn-based approach
// detect in crop
regions_t DetectInCrop(Net& net, cv::Mat& colorMat, cv::Size crop, vector<Mat>& outs) {
	cv::Mat blob;
	vector<int>  classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> indices;
	regions_t tmpregions;
	blobFromImage(colorMat, blob, 1 / 255.0, crop, Scalar(0, 0, 0), false, true);
	vector<cv::Mat > Blobs;
	imagesFromBlob(blob, Blobs);
  imshow("original blob", colorMat);
	for (int ii = 0; ii < Blobs.size(); ii++) {
		imshow(format("blob image: %d", ii), Blobs[ii]);
		waitKey(1);
	}
	//Sets the input to the network
	net.setInput(blob);
	// Runs the forward pass to get output of the output layers	
  try
  {
    net.forward(outs, getOutputsNames(net)); // forward pass for network layers	 
  }
  catch (...)
  {
    /*for (int ii = 0; ii < Blobs.size(); ii++) {
      imshow(format("blob image: %d", ii), Blobs[ii]);
      waitKey(0);
    }*/
    cout<<"cout not pass forward to the network \n";
    return tmpregions;
  }
	

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * colorMat.cols);
				int centerY = (int)(data[1] * colorMat.rows);
				int width = (int)(data[2] * colorMat.cols);
				int height = (int)(data[3] * colorMat.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences	
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		tmpregions.push_back(CRegion(box, classes[classIds[idx]], confidences[idx]));
	}
	return tmpregions;
}

void getPredicInfo(const vector<Mat>& outs, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes) {

}


// DNN -realted functions
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
  vector<int> classIds;
  vector<float> confidences;
  vector<Rect> boxes;

  for (size_t i = 0; i < outs.size(); ++i)
  {
    // Scan through all the bounding boxes output from the network and keep only the
    // ones with high confidence scores. Assign the box's class label as the class
    // with the highest score for the box.
    float* data = (float*)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
    {
      Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      Point classIdPoint;
      double confidence;
      // Get the value and location of the maximum score
      minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > confThreshold)
      {
        int centerX = (int)(data[0] * frame.cols);
        int centerY = (int)(data[1] * frame.rows);
        int width = (int)(data[2] * frame.cols);
        int height = (int)(data[3] * frame.rows);
        int left = centerX - width / 2;
        int top = centerY - height / 2;

        classIds.push_back(classIdPoint.x);
        confidences.push_back((float)confidence);
        boxes.push_back(Rect(left, top, width, height));
      }
    }
  }

  // Perform non maximum suppression to eliminate redundant overlapping boxes with
  // lower confidences
  vector<int> indices;
  NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
  for (size_t i = 0; i < indices.size(); ++i)
  {
    int idx = indices[i];
    Rect box = boxes[idx];
    /* drawPred(classIds[idx], confidences[idx], box.x, box.y,
    box.x + box.width, box.y + box.height, frame);*/
    drawPred(classIds[idx], confidences[idx], box.x - box.width / 2, box.y - box.height / 2,
      box.x + box.width / 2, box.y + box.height / 2, frame);
  }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
  //Draw a rectangle displaying the bounding box
  rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

  //Get the label for the class name and its confidence
  string label = format("%.2f", conf);
  if (!classes.empty())
  {
    CV_Assert(classId < (int)classes.size());
    label = classes[classId] + ":" + label;
  }

  //Display the label at the top of the bounding box
  int baseLine;
  Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = max(top, labelSize.height);
  rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
  putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
  static vector<String> names;
  if (names.empty())
  {
    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    vector<int> outLayers = net.getUnconnectedOutLayers();

    //get the names of all the layers in the network
    vector<String> layersNames = net.getLayerNames();

    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
      names[i] = layersNames[outLayers[i] - 1];
  }
  return names;
}
/*
int inpWidthorg = 52;
int inpHeightorg = 37;
int inpWidth = (inpWidthorg) % 32 == 0 ? inpWidthorg : inpWidthorg + (32 - (inpWidthorg % 32));// 1280 / 4;  //min(416, int(1280. / 720.*64.*3.) + 1);// 160; //1920 / 2;// 416;  // Width of network's input image
int inpHeight = (inpHeightorg) % 32 == 0 ? inpHeightorg : inpHeightorg + (32 - (inpHeightorg % 32));// 720 / 4;  //64;// 160;// 1080 / 2;// 416; // Height of network's input image
*/
cv::Size adjustNetworkInputSize(Size inSize) {  
  int inpWidth = (inSize.width) % 13 == 0 ? inSize.width : inSize.width + (13 - (inSize.width % 13));// we make the input mutiples of 32
  int inpHeight = (inSize.height) % 13 == 0 ? inSize.height : inSize.height + (13 - (inSize.height % 13));// it depends the network architecture
  if (inpHeight > inpWidth)
    inpWidth = inpHeight;
  
  return Size(inpWidth, inpHeight);
}
// ----------------------- DNN related functions  end ----------------------------------

auto cmp = [](std::pair<string, float > const & a, std::pair<string, float> const & b)
{
	return a.second > b.second; // descending order
};

// std::sort(items.begin(), items.end(), cmp);
// classificy an object with distance and its size
void classifyObjectWithDistanceRatio(Blob &srcBlob, float distFromZero/* distance from the starting point*/, ObjectClass & objClass, float& fprobability)
{  
  // --- algorithm ---------------------------------------------------------------------
  // get the pair infors for object, 
  // 0. get the current object information of width and height
  // 1. get the width and height of the reference class object from the given distance
  // 2. compute the probability for each class, vector<pair<enum, float >>
  // 3. sort with probabilty in descending order
  // 4. determine the class for the given object
  // ------------------------------------------------------------------------------------
  float fmininum_class_prob = 0.5;  // minimum probability for declaring the class type
	std::vector<std::pair<std::string, float>> objClassProbs; // object class with probabilities
	float fdistance = distFromZero, prob=0.f, perc_Thres = 0.25; // 25% error range
	float fWidthHeightWeightRatio_Width = 0.7; // width 0.7 height 0.3
	
	int tgtWidth, tgtHeight, refWidth, refHeight; // target, reference infors
    float tgtWidthHeightRatio, tgtCredit = 1.1;   // credit 10 %
	tgtWidth = srcBlob.currentBoundingRect.width;
	tgtHeight = srcBlob.currentBoundingRect.height;
    tgtWidthHeightRatio = (float)tgtHeight / (float)tgtWidth; // give the more credit according to the shape for vehicle or human 10 %
  vector<float> objWidth;     // for panelty against distance
  vector<float> objHeight; 
	// configuration 
  vector<float> sedan_h = { -0.00004444328872f, 0.01751602326f, -2.293443176f, 112.527668f }; // scale factor 0.5
  vector<float> sedan_w = { -0.00003734137716f, 0.01448943505f, -1.902199174f, 98.56691135f };
  vector<float> suv_h = { -0.00005815785621f, 0.02216859672f, -2.797603666f, 139.0638999f };
  vector<float> suv_w = { -0.00004854032314f, 0.01884736545f, -2.425686251f, 121.9226426f };
  vector<float> truck_h = { -0.00006123592908f, 0.02373661426f, -3.064585294f, 149.6535855f };
  vector<float> truck_w = { -0.00003778247771f, 0.015239317f, -2.091105041f, 110.7544702f };
  vector<float> human_h = { -0.000002473245036f, 0.001813179193f, -0.5058008988f, 49.27950311f };
  vector<float> human_w = { -0.000003459461125f, 0.001590306464f, -0.3208648543f, 28.23621306f };
	ITMSPolyValues polyvalue_sedan_h(sedan_h, sedan_h.size());
	ITMSPolyValues polyvalue_sedan_w(sedan_w, sedan_w.size());
	ITMSPolyValues polyvalue_suv_h(suv_h, suv_h.size());
	ITMSPolyValues polyvalue_suv_w(suv_w, suv_w.size());
	ITMSPolyValues polyvalue_truck_h(truck_h, truck_h.size());
	ITMSPolyValues polyvalue_truck_w(truck_w, truck_w.size());
	ITMSPolyValues polyvalue_human_h(human_h, human_h.size());
	ITMSPolyValues polyvalue_human_w(human_w, human_w.size());
  // 
  objWidth.push_back(polyvalue_sedan_w.getPolyValue(fdistance));
  objWidth.push_back(polyvalue_suv_w.getPolyValue(fdistance));
  objWidth.push_back(polyvalue_truck_w.getPolyValue(fdistance));
  objWidth.push_back(polyvalue_human_w.getPolyValue(fdistance));
  sort(objWidth.begin(), objWidth.end(), greater<float>()); // descending order
  float minRefWidth = objWidth.at(objWidth.size() - 1), maxRefWidth = objWidth.at(0);

  objHeight.push_back(polyvalue_sedan_h.getPolyValue(fdistance));
  objHeight.push_back(polyvalue_suv_h.getPolyValue(fdistance));
  objHeight.push_back(polyvalue_truck_h.getPolyValue(fdistance));
  objHeight.push_back(polyvalue_human_h.getPolyValue(fdistance));
  sort(objHeight.begin(), objHeight.end(), greater<float>());
  float minRefHeight = objWidth.at(objWidth.size() - 1), maxRefHeight = objWidth.at(0);

  // size constraints
  
  if (tgtWidth > maxRefWidth*(1 + perc_Thres) || tgtWidth < minRefWidth*(1 - perc_Thres))
    tgtWidth = 0;
  if (tgtHeight > maxRefHeight*(1 + perc_Thres) || tgtHeight < minRefHeight*(1 - perc_Thres))
    tgtHeight = 0;

	// sedan
	refHeight = polyvalue_sedan_h.getPolyValue(fdistance);
	refWidth = polyvalue_sedan_w.getPolyValue(fdistance);
	prob = fWidthHeightWeightRatio_Width*(refWidth - fabs(refWidth - tgtWidth)) / refWidth +
		(1.f-fWidthHeightWeightRatio_Width)*(refHeight-fabs(refHeight-tgtHeight))/refHeight;
    //prob = prob*tgtWidthHeightRatio>= 1.0? 1.0: prob*tgtWidthHeightRatio; // size constraint min(1.0, prob*tgtWidthHeightRatio)
	objClassProbs.push_back(pair<string, float>("sedan", prob));
	// suv
	refHeight = polyvalue_suv_h.getPolyValue(fdistance);
	refWidth = polyvalue_suv_w.getPolyValue(fdistance);
	prob = fWidthHeightWeightRatio_Width*(refWidth - fabs(refWidth - tgtWidth)) / refWidth +
		(1.f - fWidthHeightWeightRatio_Width)*(refHeight - fabs(refHeight - tgtHeight)) / refHeight;
    //prob = prob*tgtWidthHeightRatio >= 1.0 ? 1.0 : prob*tgtWidthHeightRatio; // size constraint min(1.0, prob*tgtWidthHeightRatio)
	objClassProbs.push_back(pair<string, float>("suv", prob));
	// truck
	refHeight = polyvalue_truck_h.getPolyValue(fdistance);
	refWidth = polyvalue_truck_w.getPolyValue(fdistance);
	prob = fWidthHeightWeightRatio_Width*(refWidth - fabs(refWidth - tgtWidth)) / refWidth +
		(1.f - fWidthHeightWeightRatio_Width)*(refHeight - fabs(refHeight - tgtHeight)) / refHeight;
    //prob = prob*tgtWidthHeightRatio >= 1.0 ? 1.0 : prob*tgtWidthHeightRatio; // size constraint min(1.0, prob*tgtWidthHeightRatio)
	objClassProbs.push_back(pair<string, float>("truck", prob));
    
	// human
	refHeight = polyvalue_human_h.getPolyValue(fdistance);
	refWidth = polyvalue_human_w.getPolyValue(fdistance);
	prob = (1.f-fWidthHeightWeightRatio_Width)*(refWidth - fabs(refWidth - tgtWidth)) / refWidth +
		(fWidthHeightWeightRatio_Width)*(refHeight - fabs(refHeight - tgtHeight)) / refHeight; // height is more important for human 
    prob = prob*tgtWidthHeightRatio >= 1.0 ? 1.0 : prob*tgtWidthHeightRatio; // size constraint min(1.0, prob*tgtWidthHeightRatio)
	objClassProbs.push_back(pair<string, float>("human", prob));

	sort(objClassProbs.begin(), objClassProbs.end(), cmp); // sort the prob in decending order
	string strClass = objClassProbs.at(0).first; // class
	fprobability = objClassProbs.at(0).second;   // prob 
  if (fprobability >= fmininum_class_prob) {   // the if and its below can be replaced with (?) a:b; for speed
    if (strClass == "human")
      objClass = ObjectClass::OC_HUMAN;
    else
    {
      objClass = ObjectClass::OC_VEHICLE;
    }

  }
  else {
    objClass = ObjectClass::OC_OTHER; // not determined
  }
  // update currentBlob
  srcBlob.oc_prob = fprobability;
  srcBlob.oc = objClass;
}

void detectCascadeRoi(cv::Mat img, cv::Rect& rect)
{ /* please see more details in Object_Detector_Cascade Project */
	Mat roiImg = img(rect).clone();
	Mat hogImg;
	// debug details
	hogImg = roiImg.clone();
	int casWidth = 128; // ratio is 1: 1 for width to height
	int svmWidth = 64 * 1.5, svmHeight = 128 * 1.5;

	// adjust cascade window image
	float casRatio = (float)casWidth/roiImg.cols;
	//bool debugGeneralDetails = true;
	//bool debugShowImage = true;

	resize(roiImg, roiImg, Size(), casRatio, casRatio);

	Size img_size = roiImg.size();
	vector<Rect> object;
	vector<Rect> people;	
	cascade.detectMultiScale(roiImg, object, 1.1, 5/*1  cascadG */, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(0, 0), img_size); // detectio objects (car)
	//cascade.detectMultiScale(img, object, 1.04, 5, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(3, 8), img_size); // detectio objects (people)

	// adjust the size : hog is not working if the size of an image is not fitted to the definition	
	if (hogImg.cols < svmWidth) {
		float widthRatio = (float)svmWidth/hogImg.cols;
		resize(hogImg,hogImg, Size(), widthRatio,widthRatio); // same ratio is applied to both direction
	}
	if (hogImg.rows < svmHeight) {
		float heightRatio = (float)svmHeight/hogImg.rows;
		resize(hogImg, hogImg, Size(), heightRatio, heightRatio);
	}
	hog.detectMultiScale(hogImg, people, 0, Size(4, 4), Size(8, 8), 1.05, 2, false);							// detect people
	if(debugGeneralDetail){
			std::cout << "Total: " << object.size() << " cars detected." << std::endl;
			std::cout << "=>=> " << people.size() << " people detected." << std::endl;
	}

	for (int i = 0; i < (object.size() ? object.size()/*object->total*/ : 0); i++)
	{		
		Rect r = object.at(i);
		if(debugShowImagesDetail)
			cv::rectangle(roiImg,
				cv::Point(r.x, r.y),
				cv::Point(r.x + r.width, r.y + r.height),
				CV_RGB(255, 0, 0), 2, 8, 0);
	}	
	for (int i = 0; i < (people.size() ? people.size()/*object->total*/ : 0); i++)
	{
		//CvRect *r = (CvRect*)cvGetSeqElem(object, i);
		Rect r1 = people.at(i);
		if(debugShowImagesDetail)
				cv::rectangle(hogImg,
				cv::Point(r1.x, r1.y),
				cv::Point(r1.x + r1.width, r1.y + r1.height),
				CV_RGB(0, 255, 0), 2, 8, 0);
	}
	if (debugShowImagesDetail) {
		imshow("cascade image", roiImg);
		imshow("hog", hogImg);
		waitKey(1);		
	}
	
}
void detectCascadeRoiVehicle(/* put config file */cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _cars)
{ /* please see more details in Object_Detector_Cascade Project */
	Mat roiImg = img(rect).clone();	
		
	int casWidth = 128; // ratio is 1: 1 for width to height
	if (bgsubtype == BgSubType::BGS_CNT)
		casWidth = (int) ((float)casWidth *1.5);
	
	// adjust cascade window image
	float casRatio = (float)casWidth / roiImg.cols;
	
	resize(roiImg, roiImg, Size(), casRatio, casRatio);

	Size img_size = roiImg.size();
	vector<Rect> object;	
	cascade.detectMultiScale(roiImg, object, 1.1, 5/*1  cascadG */, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(0, 0), img_size); // detectio objects (car)
	//cascade.detectMultiScale(img, object, 1.04, 5, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(3, 8), img_size); // detectio objects (people)
	if (debugGeneralDetail) {
		std::cout << "Total: " << object.size() << " cars are detected in detectCascadeRoiVehicle function." << std::endl;		
	}

	for (int i = 0; i < (object.size() ? object.size()/*object->total*/ : 0); i++)
	{
		Rect r = object.at(i);
		// check the center point of the given ROI is in the rect of the output
		Rect tgtRect(roiImg.cols/2-1, roiImg.rows/2-1,3,3); // 3x3 at center point of the ROI 
		Rect inter = (tgtRect & r);
		if(inter.area())
			_cars.push_back(r);

		if (debugShowImagesDetail){
			if(roiImg.channels() < 3)
				cvtColor(roiImg, roiImg, CV_GRAY2BGR);
			cv::rectangle(roiImg,
				r,
				CV_RGB(255, 0, 0), 2, 8, 0);
			if(inter.area())
				cv::rectangle(roiImg, inter, CV_RGB(255,0,0), 2,8, 0);
		}
	}	
	if (debugShowImagesDetail) {
		imshow("vehicle in detection", roiImg);		
		waitKey(1);
	}
}
// find people in the given ROI
void detectCascadeRoiHuman(/* put config file */cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _people)
{ 
/* this function return the location of human according to the detection method
1. svm based algorithm which needs more computation time
2. cascade haar-like approach, which is fast but not much robust compared to SVM-based approach 
*/

	// debugging 
	//bool debugGeneralDetails = true;
	//bool debugShowImage = true;

	Mat hogImg = img(rect).clone();
	// debug details

#ifdef _CASCADE_HUMAN
	int casWidth = 128; // ratio is 1: 1 for width to height
	if (bgsubtype == BgSubType::BGS_CNT)
		casWidth = (int)((float)casWidth *1.5);
	float casRatio = (float)casWidth / hogImg.cols;

	resize(hogImg, hogImg, Size(), casRatio, casRatio);

	Size img_size = hogImg.size();
	vector<Rect> people;	
	cascade.detectMultiScale(hogImg, people, 1.1, 5/*1  cascadG */, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(0, 0), img_size); // detectio objects (car)
	//cascade.detectMultiScale(hogImg, people, 1.04, 5, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(3, 8), img_size); // detectio objects (people) 
	// need to change the xml file for human instead of using cars.xml
#else
	int svmWidth = 64 * 1.5, svmHeight = 128 * 1.5;
	if (bgsubtype == BgSubType::BGS_CNT) {
		svmWidth = (int)((float)svmWidth *2);
		svmHeight = (int)((float)svmHeight *2);
	}
	vector<Rect> people;
	if (hogImg.cols < svmWidth) {
		float widthRatio = (float)svmWidth / hogImg.cols;
		resize(hogImg, hogImg, Size(), widthRatio, widthRatio); // same ratio is applied to both direction
	}
	if (hogImg.rows < svmHeight) {
		float heightRatio = (float)svmHeight / hogImg.rows;
		resize(hogImg, hogImg, Size(), heightRatio, heightRatio);
	}
	hog.detectMultiScale(hogImg, people, 0, Size(4, 4), Size(8, 8), 1.05, 2, false);							// detect people

#endif
	if (debugGeneralDetail) {	
		std::cout << "=>=> " << people.size() << " people detected in detect cascadeRoiHuamn function." << std::endl;
	}
	
	for (int i = 0; i < (people.size() ? people.size()/*object->total*/ : 0); i++)
	{		
		Rect r1 = people.at(i);
		// check the center point of the given ROI is in the rect of the output
		Rect tgtRect(hogImg.cols / 2-1, hogImg.rows / 2-1, 3, 3); // 3x3 at center point of the ROI 
		Rect inter = (tgtRect & r1);
		if(inter.area())
			_people.push_back(r1);
		if (debugShowImagesDetail){
			if(hogImg.channels()<3)
				cvtColor(hogImg, hogImg, CV_GRAY2BGR);

			cv::rectangle(hogImg,
				r1,
				CV_RGB(0, 255, 0), 2, 8, 0);
			if(inter.area())
				cv::rectangle(hogImg, inter, CV_RGB(255,0,0), 2, 8, 0);
		}
	}
	if (debugShowImagesDetail) {		
		imshow("human in HOG SVM", hogImg);
		waitKey(1);
	}
}