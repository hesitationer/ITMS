// main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows
#include <time.h>

#include "../src/itms_Blob.h"
#include "../src/bgsubcnt.h"


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
void mergeBlobsInCurrentFrameBlobs(std::vector<Blob> &currentFrameBlobs);
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, int& id);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs,int &id);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
ObjectStatus getObjectStatusFromBlobCenters(const Blob &blob, const LaneDirection &lanedirection, int movingThresholdInPixels, int minTotalVisibleCount=3);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy, cv::Point Pt1, cv::Point Pt2, int &carCount, int &truckCount, int &bikeCount);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);

///////////////////////////////////////////////////////////////////////////////////////////////////
namespace FAV1
{
  int carArea = 0;
  int truckArea = 0;
  int bikeArea =  0;
  int humanArea = 0;

  char VideoPath[512];
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
  strcpy(FAV1::VideoPath, VP);
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
bool debugTrace = true;
bool debugTime = true;
int numberOfTracePoints = 15;	// # of tracking tracer in debug Image
int minVisibleCount = 3;		  // minimum survival consecutive frame for noise removal effect
int maxCenterPts = 300;			  // maximum number of center points (frames), about 10 sec.
int maxNumOfConsecutiveInFramesWithoutAMatch = 50; // it is used for track update
int maxNumOfConsecutiveInvisibleCounts = 100; // for removing disappeared objects from the screen
int movingThresholdInPixels = 0;              // motion threshold in pixels affected by scaleFactor, average point를 이용해야 함..
int img_dif_th = 10;                          // BGS_DIF biranry threshold (10~30) at day, 

bool isWriteToFile = false;

LaneDirection ldirection = LD_NORTH; // vertical lane
BgSubType bgsubtype = BGS_DIF;



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

  float scaleFactor = 0.5;


	std::vector<Blob> blobs;

	cv::Point crossingLine[2];

	int carCount = 0;
	int truckCount = 0;
	int bikeCount = 0;
	int humanCount = 0;
	int videoLength = 0;
  
	loadConfig();
	bool b = capVideo.open(FAV1::VideoPath);

	//capVideo.open("c:/sangkny/software/Projects/OpenCV_3_Car_Counting_Cpp-master/OpenCV_3_Car_Counting_Cpp-master/CarsDrivingUnderBridge.mp4");
   //capVideo.open("c:/sangkny/software/Projects/OpenCV_3_Car_Counting_Cpp-master/OpenCV_3_Car_Counting_Cpp-master/Relaxinghighwaytraffic.mp4");  // 768x576.avi
	//capVideo.open("c:/sangkny/software/Projects/OpenCV_3_Car_Counting_Cpp-master/OpenCV_3_Car_Counting_Cpp-master/768x576.avi");  // 
	//capVideo.open("C:/Users/MMC/Downloads/20180329_202747.mp4"); //20180329_202049
	//capVideo.open("C:/Users/MMC/Downloads/20180329_202049.mp4");
	//capVideo.open("D:/LectureSSD_rescue/project-related/도로-기상-유고-토페스/안개영상/400M이상.avi");

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
    if (debugShowImages && debugShowImagesDetail) {
      imshow("road mask", road_mask);
      waitKey(1);
    }

	
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
        if (isWriteToFile && frameCount == 150) {
          string filename = FAV1::VideoPath;
          filename.append("_"+to_string(scaleFactor)+"x.jpg");
          cv::imwrite(filename, bgImage);
          std::cout << " background image has been generated !!\n";
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
            /*cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::erode(imgThresh, imgThresh, structuringElement5x5);      */    
          cv::morphologyEx(imgThresh, imgThresh, CV_MOP_CLOSE, structuringElement7x7);
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

            /*if (possibleBlob.currentBoundingRect.area() > 400 &&
                possibleBlob.dblCurrentAspectRatio > 0.2 &&
                possibleBlob.dblCurrentAspectRatio < 4.0 &&
                possibleBlob.currentBoundingRect.width > 30 &&
                possibleBlob.currentBoundingRect.height > 30 &&
                possibleBlob.dblCurrentDiagonalSize > 60.0 &&
                (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
                currentFrameBlobs.push_back(possibleBlob);
            }*/
            if (possibleBlob.currentBoundingRect.area() > 10 &&
              possibleBlob.dblCurrentAspectRatio > 0.2 &&
              possibleBlob.dblCurrentAspectRatio < 6.0 &&
              possibleBlob.currentBoundingRect.width > 2 &&
              possibleBlob.currentBoundingRect.height > 2 &&
              possibleBlob.dblCurrentDiagonalSize > 3.0 &&
              (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
              currentFrameBlobs.push_back(possibleBlob);
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
    
    mergeBlobsInCurrentFrameBlobs(currentFrameBlobs);
    if (debugShowImages && debugShowImagesDetail) {
      drawAndShowContours(imgThresh.size(), currentFrameBlobs, "after merging currentFrameBlobs");
      waitKey(100);
    }



        if (blnFirstFrame == true) {
            for (auto &currentFrameBlob : currentFrameBlobs) {
                blobs.push_back(currentFrameBlob);
            }
        } else {
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs, trackId);
        }
		imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above
		if (debugShowImages && debugShowImagesDetail) {
			drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");
		
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

        //cv::waitKey(0);                 // uncomment this line to go frame by frame for debugging

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
        cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
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
  tgtBlob.centerPositions.push_back(srcBlob.centerPositions.back());

  
  tgtBlob.dblCurrentDiagonalSize = srcBlob.dblCurrentDiagonalSize;

  tgtBlob.dblCurrentAspectRatio = srcBlob.dblCurrentAspectRatio;

  tgtBlob.blnStillBeingTracked = srcBlob.blnStillBeingTracked;
  tgtBlob.blnCurrentMatchFoundOrNewBlob = tgtBlob.blnCurrentMatchFoundOrNewBlob;

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
        //currentBlob = blob;
        copyBlob2Blob(blob, *currentBlob);
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
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, int &id) {
	std::vector<Blob>::iterator existingBlob = existingBlobs.begin();
	while ( existingBlob != existingBlobs.end()) {
		// check if a block is too old after disappeared in the screen
		if (existingBlob->blnStillBeingTracked == false 
			&& existingBlob->intNumOfConsecutiveFramesWithoutAMatch>=maxNumOfConsecutiveInvisibleCounts) { 
			// remove from the list of existingBlobs			
			if (debugTrace) {
				cout << " (!)! Old blob id: " << existingBlob->id << " is eliminated (blobs Capacity: "<< existingBlobs.capacity()<<")" << endl;				
			}
			existingBlob= existingBlobs.erase(existingBlob);			
		}
		else {
			existingBlob->blnCurrentMatchFoundOrNewBlob = false;
			existingBlob->predictNextPosition();
      existingBlob->os = getObjectStatusFromBlobCenters(*existingBlob, ldirection, movingThresholdInPixels, minVisibleCount); // 벡터로 넣을지 생각해 볼 것
			++existingBlob;
		}
    }
	/*for (auto &existingBlob : existingBlobs) {	

		existingBlob.blnCurrentMatchFoundOrNewBlob = false;

		existingBlob.predictNextPosition();
	}*/

    for (auto &currentFrameBlob : currentFrameBlobs) {

        int intIndexOfLeastDistance = 0;
        double dblLeastDistance = 100000.0;

        for (unsigned int i = 0; i < existingBlobs.size(); i++) {

            if (existingBlobs[i].blnStillBeingTracked == true) { // find assigned tracks
                // it can be replaced with the tracking algorithm or assignment algorithm like KALMAN or Hungrian Assignment algorithm 
                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
            }
            else { // existingBlobs[i].bInStillBeingTracked == false;
              /* do something for unassinged tracks */
              int temp = 0; // no meaning 
            }
        }

        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
            addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
        }
        else { // this routine contains new and unassigned track(blob)s
            addNewBlob(currentFrameBlob, existingBlobs, id);
        }

    }
    // update tracks 
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

    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {
// update the status or correcte the status
    existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

    //if (existingBlobs[intIndex].totalVisibleCount >= 8 /* it should be a predefined threshold */)
      existingBlobs[intIndex].blnStillBeingTracked = true; /* it is easy to be exposed to noise, so it put constraints to this */
   // else
   //   int kkk = 0;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &id) {

  if (currentFrameBlob.totalVisibleCount > 1)
    int temp = 0;
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;
    currentFrameBlob.id = id;
	id = (id > 2048) ? 0 : ++id; // reset id according to the max number of type (int) or time (day or week)

    existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// it should inspect after predicting the next position
// it determines the status away at most 5 frame distance from past locations
// it shoud comes from the average but for efficiency, does come from the several past location. 
ObjectStatus getObjectStatusFromBlobCenters(const Blob &blob, const LaneDirection &lanedirection, int movingThresholdInPixels, int minTotalVisibleCount) {
  ObjectStatus objectstatus=ObjectStatus::OS_NOTDETERMINED; 
  if (blob.totalVisibleCount < minTotalVisibleCount) // !! parameter
	  return objectstatus;

  int deltaX = blob.predictedNextPosition.x - blob.centerPositions.back().x;
  int deltaY = blob.predictedNextPosition.y - blob.centerPositions.back().y; // have to use moving average after applying media filtering    
  int numPositions = (int)blob.centerPositions.size();
  int maxNumPosition = 5;

  if (numPositions <= 1) {
    deltaX = 0; deltaY = 0;
  }
  else {
    deltaX = blob.predictedNextPosition.x - blob.centerPositions[min(numPositions,maxNumPosition) - 1].x;
    deltaY = blob.predictedNextPosition.y - blob.centerPositions[min(numPositions, maxNumPosition) - 1].y;
  }
  // moving average가 필요하면 predictNextPosition 참조.
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
 
  return objectstatus;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

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

    std::vector<std::vector<cv::Point> > contours;

    for (auto &blob : blobs) {
        if (blob.blnStillBeingTracked == true /*&& blob.totalVisibleCount>= minVisibleCount*/) {
            contours.push_back(blob.currentContour);
        }
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
    cv::waitKey(10);
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

        if (blobs[i].blnStillBeingTracked == true && blobs[i].totalVisibleCount > 5) {
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

            infostr = std::to_string(blobs[i].id) + status;
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

