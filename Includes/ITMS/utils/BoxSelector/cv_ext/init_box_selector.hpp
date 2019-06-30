
/*
// 
// modified by sangkny
// last updated on 2019. 01. 20
*/
#ifndef INIT_BOX_SELECTOR_HPP_
#define INIT_BOX_SELECTOR_HPP_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

class InitBoxSelector
{
public:
    static bool selectBox(cv::Mat& frame, cv::Rect& initBox);

private:
    static void onMouse(int event, int x, int y, int, void*);
    static bool startSelection;
    static bool selectObject;
    static cv::Rect initBox;
    static cv::Mat image;
    static const std::string windowTitle;
};
class InitPointSelector // 4 points
{
public:
	static bool selectPoints(cv::Mat& frame, std::vector<cv::Point>& initPoints);

private:
	static void onMouse(int event, int x, int y, int, void*);
	static bool startSelection;
	static bool selectObject;
	static std::vector<cv::Point> initBoxPts;
	static cv::Point initPt;
	static cv::Mat image;
	static const std::string windowTitle;
	static int numPoints; // number of selected points
	static int max_numPoints; //  max number of points : 4 
};

#endif /* INIT_BOX_SELECTOR_H_ */
