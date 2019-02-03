#pragma once
#ifndef  ITMS_UTILS_DNN_H
#define ITMS_UTILS_DNN_H

#include <opencv2/dnn.hpp>
namespace itms {
	class itmsDNN {
		// dnn-based approach starts
		// Remove the bounding boxes with low confidence using non-maxima suppression
		void postprocess(Mat& frame, const vector<Mat>& out);

		// Draw the predicted bounding box
		void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

		// Get the names of the output layers
		vector<String> getOutputsNames(const Net& net);


		void getPredicInfo(const vector<Mat>& outs, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes);
		regions_t DetectInCrop(Net& net, cv::Mat& colorMat, cv::Size crop, vector<Mat>& outs);
		cv::Size adjustNetworkInputSize(Size inSize);
		// dnn-based approach ends
		itmsDNN() {};
		~itmsDNN() {};
	}

}

#endif // ! ITMS_UTILS_DNN_H
