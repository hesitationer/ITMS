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
	////// simple functions 

  void imshowBeforeAndAfter(cv::Mat &before, cv::Mat &after, std::string windowtitle, int gabbetweenimages);
  Rect expandRect(Rect original, int expandXPixels, int expandYPixels, int maxX, int maxY); // 
  Rect maxSqRect(Rect& original, int maxX, int maxY); // make squre with max length
  Rect maxSqExpandRect(Rect& original, float floatScalefactor, int maxX, int maxY); // combine both above with scalefactor
	
																					// returns the value according to the tp position according to starting point sP and ending point eP;
	// + : left/bottom(below), 0: on the line, -: right/top(above)
  inline bool isPointBelowLine(cv::Point sP, cv::Point eP, cv::Point tP) {
	  return ((eP.x - sP.x)*(tP.y - sP.y) - (eP.y - sP.y)*(tP.x - sP.x)) > 0;
  }
   
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
	  ~ITMSPolyValues() {
      mPolyCoeffs.clear();
    };
  
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

