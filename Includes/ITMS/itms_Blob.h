// itms_Blob.h
// Intelligent Traffic Monitoring System (ITMS)
// Blob definition
// developed by sangkny
// 
#ifndef ITMS_BLOB_H
#define ITMS_BLOB_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<memory>
// #include "../src/fastdsst/fdssttracker.hpp" // fast tracker
#include"./psrdsst/dsst_tracker.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////
namespace itms {  
	enum BgSubType { // background substractor type
		BGS_DIF = 0, // difference
		BGS_CNT = 1, // COUNTER
		BGS_ACC = 2  // ACCUMULATER
	};

  enum ObjectClass {
    OC_VEHICLE  = 0,
    OC_HUMAN    = 1,
    OC_OTHER    = 2
  };
  enum ObjectStatus { // vehicle status
    OS_STOPPED        = 0,
    OS_MOVING_FORWARD = 1,
    OS_MOVING_BACKWARD= 2, // wrong way on a street
    OS_NOTDETERMINED  = 3   // and other cases
  };
  /*enum LaneDirection {
    LD_NONE       = 0,
    LD_HORIZONTAL = 1,
    LD_VERTICAL   = 2
  };*/
  enum LaneDirection { // ACTUALLY, IT IS CORRECT CAR-MOVING DIRECTION
    LD_NONE     = 0,
    LD_NORTH    = 1,
    LD_NORTHEAST= 2,
    LD_EAST     = 3,
    LD_SOUTHEAST= 4,
    LD_SOUTH    = 5,
    LD_SOUTHWEST= 6,
    LD_WEST     = 7,
    LD_NORTHWEST= 8
  };
  enum ObjectDirection {
    OD_AB = 0,              // correct direction
    OD_BA = 1,              // incorrect direction
    OD_ND = 2               // not determined
  };

  //// ----------------------------------------------------------------------------------
  /// \brief sqr
  /// \param val
  /// \return
  ///
  template<class T> inline
	  T sqr(T val)
  {
	  return val * val;
  }

  ////
  /// \brief get_lin_regress_params for getting x(t)_x = k_x*t + dx, x(t)_y = k_y*t + dy
  /// \param in_data
  /// \param start_pos
  /// \param in_data_size
  /// \param kx
  /// \param bx
  /// \param ky
  /// \param by
  ///
  template<typename T, typename CONT>
  void get_lin_regress_params(
	  const CONT& in_data,
	  size_t start_pos,
	  size_t in_data_size,
	  T& kx, T& bx, T& ky, T& by)
  {
	  T m1(0.), m2(0.);
	  T m3_x(0.), m4_x(0.);
	  T m3_y(0.), m4_y(0.);

	  const T el_count = static_cast<T>(in_data_size - start_pos);
	  for (size_t i = start_pos; i < in_data_size; ++i)
	  {
		  m1 += i;
		  m2 += sqr(i);

		  m3_x += in_data[i].x;
		  m4_x += i * in_data[i].x;

		  m3_y += in_data[i].y;
		  m4_y += i * in_data[i].y;
	  }
	  T det_1 = 1. / (el_count * m2 - sqr(m1));

	  m1 *= -1.;

	  kx = det_1 * (m1 * m3_x + el_count * m4_x);
	  bx = det_1 * (m2 * m3_x + m1 * m4_x);

	  ky = det_1 * (m1 * m3_y + el_count * m4_y);
	  by = det_1 * (m2 * m3_y + m1 * m4_y);
  }

  class Blob {      // it will be used as a track
	  
  public:
	  Blob(std::vector<cv::Point> _contour);

	  Blob(void) {};
	  ~Blob(void);

    // member variables ///////////////////////////////////////////////////////////////////////////
    std::vector<cv::Point> currentContour;
	// 2018. 12. 29 sangkny
	// mpoints for local search with m_collectPoints
	 std::vector<cv::Point2f> m_points;

    cv::Rect currentBoundingRect;

    std::vector<cv::Point> centerPositions;

    double dblCurrentDiagonalSize;
    double dblCurrentAspectRatio;

    bool blnCurrentMatchFoundOrNewBlob;
    bool blnStillBeingTracked;
    int intNumOfConsecutiveFramesWithoutAMatch;   // consecutiveInvisibleCount    
    int age;                                      // how many frames passed after birth
    int totalVisibleCount;                        // how many times Visible total whatever appear or disappeared
	int id;                                       // track id will be given
	//int showId;                                   // display id
	double speed;								  // km/hour
    
    // distance from starting point (0 meter(x100 centimeters))
    cv::Point startPoint; // save start center Point of the blob
	// blob 
    // blob object status information
    ObjectClass oc;	
    ObjectStatus os;
	ObjectStatus fos; // final object status
    ObjectDirection od; // lane direction will affect the result, and the lane direction will be given
	// counters
	int oc_vehicle_cnter; // objectClass vehicle counter
	int oc_human_cnter;
	int oc_other_cnter;
	double oc_prob;		  // oc probability

	int os_stopped_cnter;                   // object status counter
	int os_mvForward_cnter;	
	int os_mvBackward_cnter;	
	int os_notdetermined_cnter;
	int os_NumOfConsecutiveStopped_cnter;
	int os_NumOfConsecutivemvForward_cnter;	// number of consecutive moving forward counter, it can not be larger than os_mvForward_cnter
	int os_NumOfConsecutivemvBackward_cnter;
	double os_pro;			// os probability

	bool bNotifyMessage;	// message notification flag
	ObjectClass oc_notified; // final notified object class
	ObjectStatus os_notified; // final notified object status

    cv::Point predictedNextPosition; // corresponding to particles in Surveillance Camera

	/// visual tracking 
	//std::unique_ptr<FDSSTTracker> m_tracker;
	// cv::Ptr<FDSSTTracker> m_tracker; // sankgny 2019. 01. 25
	cv::Ptr<cf_tracking::DsstTracker> m_tracker_psr;
	bool m_tracker_initialized = false;

    // function prototypes ////////////////////////////////////////////////////////////////////////
   

    void predictNextPosition(void); // 
	bool resetBlobContourWithCenter(const cv::Point2f& _newCtrPt); // reset the contour of a blob using _newCtrPt, center itself is not moved
	bool resetBlobContourWithCenter(const cv::Point& _newCtrPt);
    cv::Point weightedPositionAverage(int bWeighted=0); // weighted centerposition average
    // numTap: # of coefficients, bWeighted: weighted or uniform (true/false)
    std::string getBlobStatus(void);
    std::string getBlobClass(void);

	void CreateExternalTracker(void);		// create Tracker for lost object using fast DSST

	void operator = (const Blob &rhBlob);	// operator overloading
  };
}

#endif    // ITMS_BLOB

