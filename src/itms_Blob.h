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
  enum LaneDirection {
    LD_NONE       = 0,
    LD_HORIZONTAL = 1,
    LD_VERTICAL   = 2
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
    // blob object status information
    ObjectClass oc;
    ObjectStatus os;
    ObjectDirection od; // lane direction will affect the result, and the lane direction will be given

    cv::Point predictedNextPosition; // corresponding to particles in Surveillance Camera

    // function prototypes ////////////////////////////////////////////////////////////////////////
    Blob(std::vector<cv::Point> _contour);
    void predictNextPosition(void); // 

  };
}

#endif    // ITMS_BLOB

