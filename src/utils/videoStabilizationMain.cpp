#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

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

bool debugGeneral = true;
bool debugImshow = true;
bool flagVideostabilization = true;



// utility functions
// using namespace cv;
// and whatever header 'abs' requires...

//  offsetImageWithPadding : shift the image to offsetX, offsetY with backrouhdColour paddings
Mat offsetImageWithPadding(const Mat& originalImage, int offsetX, int offsetY, Scalar backgroundColour) {  
  cv::Mat padded = Mat(originalImage.rows + 2 * abs(offsetY), originalImage.cols + 2 * abs(offsetX), originalImage.type(), backgroundColour);
  originalImage.copyTo(padded(Rect(abs(offsetX), abs(offsetY), originalImage.cols, originalImage.rows)));
  return Mat(padded, Rect(abs(offsetX) + offsetX, abs(offsetY) + offsetY, originalImage.cols, originalImage.rows));
}
//example use with black borders along the right hand side and top:
//Mat offsetImage = offsetImageWithPadding(originalImage, -10, 6, Scalar(0, 0, 0));

int main(int, char*[])
{
#ifdef _sk_Memory_Leakag_Detector
#if _DEBUG
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif	
  std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

  std::string filename = "D:/LectureSSD_rescue/project-related/도로-기상-유고-토페스/code/ITMS/TrafficVideo/20180912_134130_cam_0.avi";
  //VideoCapture video(0);
  VideoCapture video(filename);
  if (!video.isOpened()) {
    std::cout << " the video file does not exist!!\n"; // iostream
    return 0;
  }

  Mat frame, curr, prev, curr64f, prev64f, hann;
  Mat correctedMat, diffMat;
  char key;
  int frameNumCounter = 0;
  float scalefactor = 0.25;
  do
  {
    video >> frame;

    if (frame.empty()) {
      cout << "the current frame has some problem!! and exit the program(!)\n";
      getchar();

#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
      _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	

      return 0;
    }

    resize(frame, frame, cv::Size(), scalefactor, scalefactor);
    cvtColor(frame, curr, COLOR_RGB2GRAY);

    if (prev.empty())
    {
      prev = curr.clone();
      createHanningWindow(hann, curr.size(), CV_64F);
    }

    prev.convertTo(prev64f, CV_64F);
    curr.convertTo(curr64f, CV_64F);

    Point2d shift = phaseCorrelate(prev64f, curr64f, hann);
    double radius = std::sqrt(shift.x*shift.x + shift.y*shift.y);    

    if (radius >= 1)
    {
      // draw a circle and line indicating the shift direction...
      Point center(curr.cols >> 1, curr.rows >> 1);
      circle(frame, center, (int)radius, Scalar(0, 255, 0), 3, LINE_AA);
      line(frame, center, Point(center.x + (int)shift.x, center.y + (int)shift.y), Scalar(0, 255, 0), 3, LINE_AA);
      // correct the current image    
      if (flagVideostabilization) {
        correctedMat = offsetImageWithPadding(curr, shift.x, shift.y, Scalar(0, 0, 0));
      }
      if (debugGeneral && debugImshow && flagVideostabilization) {
        cout << " shift (x,y) : (" << shift.x << ", " << shift.y << ")\n";

        cv::absdiff(prev, correctedMat, diffMat);
        cv::Mat thdiffMat= cv::Mat::zeros(diffMat.size(), diffMat.type());
        cv::threshold(diffMat, thdiffMat, 30, 255.0, CV_THRESH_BINARY);        
        imshow(" difference between pre and crtMat(curr)", thdiffMat);
        diffMat = cv::Mat::zeros(diffMat.size(), diffMat.type());
        cv::absdiff(prev, curr, diffMat);
        thdiffMat = cv::Mat::zeros(diffMat.size(), diffMat.type());
        cv::threshold(diffMat, thdiffMat, 30, 255.0, CV_THRESH_BINARY);
        imshow(" difference between pre and curr", thdiffMat);
      }
    }
    if (debugGeneral && debugImshow) {
      imshow("phase shift", frame);      

    }
    key = (char)waitKey(2);

    prev = curr.clone();
    if (debugGeneral) {
      cout << "frame #" << frameNumCounter << endl;      
    }
    frameNumCounter++;
  } while (key != 27); // Esc to exit...

#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	

  return 0;
}