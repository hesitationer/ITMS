// itms_Blob.h
// Intelligent Traffic Monitoring System (ITMS)
// Blob definition
// developed by sangkny
// 
#ifndef ITMS_BLOB
#define ITMS_BLOB

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////////
namespace itms {  
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

  class Blob {      // it will be used as a track
  public:
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
    int id;                                       // will be given
    int showId;                                   // display id
    // distance from starting point (0 meter(x100 centimeters))
    cv::Point startPoint; // save start center Point of the blob
	// blob 
    // blob object status information
    ObjectClass oc;	
    ObjectStatus os;
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

    // function prototypes ////////////////////////////////////////////////////////////////////////
    Blob(std::vector<cv::Point> _contour);
	Blob(void){};
    void predictNextPosition(void); // 
    cv::Point weightedPositionAverage(int bWeighted=0); // weighted centerposition average
    // numTap: # of coefficients, bWeighted: weighted or uniform (true/false)
    std::string getBlobStatus(void);
    std::string getBlobClass(void);

	void operator = (const Blob &rhBlob);	// operator overloading
  };
}

#endif    // ITMS_BLOB

