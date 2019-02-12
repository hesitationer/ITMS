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

		// raod configuration related
		float camera_height = 11.0 * 100;				// camera height 11 meter
		float lane_length = 200.0 * 100;				// lane length
		float lane2lane_width = 3.5 * 2 * 100;			// lane width

														// road points settings

														// object tracking related 
		int minVisibleCount = 3;						// minimum survival consecutive frame for noise removal effect
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
   
  //// system related  
  inline bool existFileTest(const std::string& name) {
	  struct stat buffer;
	  return (stat(name.c_str(), &buffer) == 0);
  }

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
  float getDistanceInMeterFromPixels(std::vector<cv::Point2f> &srcPx, cv::Mat &transmtx /* 3x3*/, float _laneLength = 20000, bool flagLaneDirectionTop2Bottom = false);
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
  ObjectStatus getObjectStatusFromBlobCenters(Blob &blob, const LaneDirection &lanedirection, int movingThresholdInPixels, int minTotalVisibleCount = 3);
  void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName, const cv::Scalar& _color=SCALAR_WHITE);
  void drawAndShowContours(itms::Config& _conf, cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
  bool checkIfBlobsCrossedTheLine(itms::Config& _conf, std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
  bool checkIfBlobsCrossedTheLine(itms::Config& _conf, std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy, cv::Point Pt1, cv::Point Pt2, int &carCount, int &truckCount, int &bikeCount);
  bool checkIfBlobsCrossedTheBoundary(itms::Config& _conf, std::vector<Blob> &blobs,/* cv::Mat &imgFrame2Copy,*/ itms::LaneDirection _laneDirection, std::vector<cv::Point> &_tboundaryPts);
  bool checkIfPointInBoundary(const itms::Config& _conf, const cv::Point& p1, const std::vector<cv::Point> &_tboundaryPts);
  void drawBlobInfoOnImage(itms::Config& _conf, std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
  void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);
  void drawRoadRoiOnImage(std::vector<std::vector<cv::Point>> &_roadROIPts, cv::Mat &_srcImg);
  void updateBlobProperties(itms::Blob &updateBlob, itms::ObjectStatus &curStatus); // update simple blob properties including os counters
  ObjectStatus computeObjectStatusProbability(const itms::Blob &srcBlob); // compute probability and returns object status 

																		  // classificy an object with distance and its size
  void classifyObjectWithDistanceRatio(itms::Config& _conf, Blob &srcBlob, float distFromZero/* distance from the starting point*/, ObjectClass & objClass, float& fprobability);

  // cascade detector related
  void detectCascadeRoi(itms::Config& _conf, cv::Mat img, cv::Rect& rect);
  void detectCascadeRoiVehicle(itms::Config& _conf, /* put config file */const cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _cars);
  void detectCascadeRoiHuman(itms::Config& _conf, /* put config file */const cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _people);
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
	  bool process(cv::Mat& curImg);
	  
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

