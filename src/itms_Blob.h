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

    cv::Point predictedNextPosition; // corresponding to particles in Surveillance Camera

    // function prototypes ////////////////////////////////////////////////////////////////////////
    Blob(std::vector<cv::Point> _contour);
    void predictNextPosition(void);

  };
}

#endif    // MY_BLOB

