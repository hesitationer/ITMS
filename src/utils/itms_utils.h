/*	this header implements required misc functions for ITMS 
	implemented by sangkny
	sangkny@gmail.com
	last updated on 2018. 10. 07

*/
#ifndef _ITMS_UTILS_H
#define _ITMS_UTILS_H

#include <iostream>
#include "opencv/cv.hpp"


using namespace cv;
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

namespace itms {
  void imshowBeforeAndAfter(cv::Mat &before, cv::Mat &after, std::string windowtitle, int gabbetweenimages);
	
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
}
#endif // _ITMS_UTILS_H

