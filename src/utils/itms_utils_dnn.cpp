#include "itms_utils_dnn.h"

namespace itms {
	// dnn-based approach
	// detect in crop
	regions_t itmsDNN::DetectInCrop(itms::Config& _conf, Net& net, cv::Mat& colorMat, cv::Size crop, vector<Mat>& outs) {
		cv::Mat blob;
		vector<int>  classIds;
		vector<float> confidences;
		vector<Rect> boxes;
		vector<int> indices;
		regions_t tmpregions;
		blobFromImage(colorMat, blob, 1 / 255.0, crop, Scalar(0, 0, 0), false, true);
		vector<cv::Mat > Blobs;
		imagesFromBlob(blob, Blobs);
		imshow("original blob", colorMat);
		for (int ii = 0; ii < Blobs.size(); ii++) {
			imshow(format("blob image: %d", ii), Blobs[ii]);
			waitKey(1);
		}
		//Sets the input to the network
		net.setInput(blob);
		// Runs the forward pass to get output of the output layers	
		try
		{
			net.forward(outs, getOutputsNames(net)); // forward pass for network layers	 
		}
		catch (...)
		{
			/*for (int ii = 0; ii < Blobs.size(); ii++) {
			imshow(format("blob image: %d", ii), Blobs[ii]);
			waitKey(0);
			}*/
			cout << "cout not pass forward to the network \n";
			return tmpregions;
		}


		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Scan through all the bounding boxes output from the network and keep only the
			// ones with high confidence scores. Assign the box's class label as the class
			// with the highest score for the box.
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				// Get the value and location of the maximum score
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > _conf.confThreshold)
				{
					int centerX = (int)(data[0] * colorMat.cols);
					int centerY = (int)(data[1] * colorMat.rows);
					int width = (int)(data[2] * colorMat.cols);
					int height = (int)(data[3] * colorMat.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences	
		NMSBoxes(boxes, confidences, _conf.confThreshold, _conf.nmsThreshold, indices);
		for (size_t i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];
			Rect box = boxes[idx];
			tmpregions.push_back(CRegion(box, classes[classIds[idx]], confidences[idx]));
		}
		return tmpregions;
	}

	void itmsDNN::getPredicInfo(const vector<Mat>& outs, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes) {

	}


	// DNN -realted functions
	// Remove the bounding boxes with low confidence using non-maxima suppression
	void itmsDNN::postprocess(itms::Config& _conf, Mat& frame, const vector<Mat>& outs)
	{
		vector<int> classIds;
		vector<float> confidences;
		vector<Rect> boxes;

		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Scan through all the bounding boxes output from the network and keep only the
			// ones with high confidence scores. Assign the box's class label as the class
			// with the highest score for the box.
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				// Get the value and location of the maximum score
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > _conf.confThreshold)
				{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		vector<int> indices;
		NMSBoxes(boxes, confidences, _conf.confThreshold, _conf.nmsThreshold, indices);
		for (size_t i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];
			Rect box = boxes[idx];
			/* drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);*/
			drawPred(classIds[idx], confidences[idx], box.x - box.width / 2, box.y - box.height / 2,
				box.x + box.width / 2, box.y + box.height / 2, frame);
		}
	}

	// Draw the predicted bounding box
	void itmsDNN::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
	{
		//Draw a rectangle displaying the bounding box
		rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

		//Get the label for the class name and its confidence
		string label = format("%.2f", conf);
		if (!classes.empty())
		{
			CV_Assert(classId < (int)classes.size());
			label = classes[classId] + ":" + label;
		}

		//Display the label at the top of the bounding box
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
		putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
	}

	// Get the names of the output layers
	vector<String> getOutputsNames(const Net& net)
	{
		static vector<String> names;
		if (names.empty())
		{
			//Get the indices of the output layers, i.e. the layers with unconnected outputs
			vector<int> outLayers = net.getUnconnectedOutLayers();

			//get the names of all the layers in the network
			vector<String> layersNames = net.getLayerNames();

			// Get the names of the output layers in names
			names.resize(outLayers.size());
			for (size_t i = 0; i < outLayers.size(); ++i)
				names[i] = layersNames[outLayers[i] - 1];
		}
		return names;
	}
	/*
	int inpWidthorg = 52;
	int inpHeightorg = 37;
	int inpWidth = (inpWidthorg) % 32 == 0 ? inpWidthorg : inpWidthorg + (32 - (inpWidthorg % 32));// 1280 / 4;  //min(416, int(1280. / 720.*64.*3.) + 1);// 160; //1920 / 2;// 416;  // Width of network's input image
	int inpHeight = (inpHeightorg) % 32 == 0 ? inpHeightorg : inpHeightorg + (32 - (inpHeightorg % 32));// 720 / 4;  //64;// 160;// 1080 / 2;// 416; // Height of network's input image
	*/
	cv::Size itmsDNN::adjustNetworkInputSize(Size inSize) {
		int inpWidth = (inSize.width) % 13 == 0 ? inSize.width : inSize.width + (13 - (inSize.width % 13));// we make the input mutiples of 32
		int inpHeight = (inSize.height) % 13 == 0 ? inSize.height : inSize.height + (13 - (inSize.height % 13));// it depends the network architecture
		if (inpHeight > inpWidth)
			inpWidth = inpHeight;

		return Size(inpWidth, inpHeight);
	}
	// ----------------------- DNN related functions  end ----------------------------------

}