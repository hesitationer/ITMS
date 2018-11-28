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
using namespace std;
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
  Rect expandRect(Rect original, int expandXPixels, int expandYPixels, int maxX, int maxY); // 
  Rect maxSqRect(Rect& original, int maxX, int maxY); // make squre with max length
  Rect maxSqExpandRect(Rect& original, float floatScalefactor, int maxX, int maxY); // combine both above with scalefactor
	
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

  // Object size fitting class
  class ITMSPolyValues {
  public:
	  ITMSPolyValues() { mPolySize = 0; };
	  // put polynomial coefficient from the index 0 to the end
	  ITMSPolyValues(std::vector<float> polyCoeffs, int polyCoeffSize) {
		  assert(polyCoeffs.size() == (size_t)polyCoeffSize);
		  for (size_t i = 0; i < polyCoeffs.size();i++)
			  mPolyCoeffs.push_back(polyCoeffs.at(i));
		  mPolySize = mPolyCoeffs.size();
	  };
	  ~ITMSPolyValues() {};
  
	  // get a poly value 
	  double getPolyValue(float fValue) {
		  double pvalue=0;
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
}
#endif // _ITMS_UTILS_H

