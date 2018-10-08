/*	this header implements required misc functions for ITMS 
	implemented by sangkny
	sangkny@gmail.com
	last updated on 2018. 10. 07

*/
#ifndef _ITMS_UTILS_H
#define _ITMS_UTILS_H

#include "opencv/cv.hpp"
using namespace cv;

namespace itms {
	void imshowBeforeAndAfter(cv::Mat &before, cv::Mat &after, std::string windowtitle,int gabbetweenimages)
	{
		if (before.size() != after.size() || before.type() != after.type()) {
			std::cout << "Please check the input file formats including size() and type() (!).\n";
			return;
		}
		int gab = max(5, gabbetweenimages);
		Mat canvas = Mat::zeros(after.rows, after.cols * 2 + gab, after.type());

		before.copyTo(canvas(Range::all(), Range(0, after.cols)));

		after.copyTo(canvas(Range::all(), Range(after.cols + gab, after.cols * 2 + gab)));

		if (canvas.cols > 1920)
		{
			resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));
		}
		imshow(windowtitle, canvas);
	}
}
#endif // _ITMS_UTILS_H

