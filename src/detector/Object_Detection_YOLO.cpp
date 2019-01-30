// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
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


const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
;
using namespace cv;
using namespace dnn;
using namespace std;


typedef std::vector<CRegion> regions_t;


// Initialize the parameters
float confThreshold = 0.1; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidthorg = 52;
int inpHeightorg = 37;
int inpWidth = (inpWidthorg) %32 == 0 ? inpWidthorg : inpWidthorg + (32 - (inpWidthorg %32));// 1280 / 4;  //min(416, int(1280. / 720.*64.*3.) + 1);// 160; //1920 / 2;// 416;  // Width of network's input image
int inpHeight = (inpHeightorg) % 32 == 0 ? inpHeightorg : inpHeightorg + (32 - (inpHeightorg % 32));// 720 / 4;  //64;// 160;// 1080 / 2;// 416; // Height of network's input image
vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);
void getPredicInfo(const vector<Mat>& outs, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes);
regions_t DetectInCrop(Net& net, cv::Mat& colorMat, cv::Size crop, vector<Mat>& outs);

int main(int argc, char** argv)
{
  CommandLineParser parser(argc, argv, keys);
  parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
  if (parser.has("help"))
  {
    parser.printMessage();
    return 0;
  }
  
  /*
  config_t config;
        //config["modelConfiguration"] = "../data/tiny-yolo.cfg";
        //config["modelBinary"] = "../data/tiny-yolo.weights";
        std::string runtime_data_dir = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/Multitarget-tracker-master/data/";
        config["modelConfiguration"] = runtime_data_dir + "yolov3-tiny.cfg";
        config["modelBinary"] = runtime_data_dir + "yolov3-tiny.weights";
        config["classNames"] = runtime_data_dir + "coco.names";
        config["dnnTarget"] = "DNN_TARGET_OPENCL_FP16";
        config["confidenceThreshold"] = "0.5";
        config["maxCropRatio"] = "3.0";
  */
  std::string runtime_data_dir = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/Multitarget-tracker-master/data/";  
  string classesFile = runtime_data_dir + "coco.names";
  ifstream ifs(classesFile.c_str());
  // Load names of classes
  string line;
  while (getline(ifs, line)) classes.push_back(line);

  // Give the configuration and weight files for the model
  String modelConfiguration = runtime_data_dir + "yolov3-tiny.cfg"; // was yolov3.cfg
  String modelWeights = runtime_data_dir + "yolov3-tiny.weights";   // was ylov3.weights
  //String modelConfiguration = runtime_data_dir + "tiny-yolo.cfg"; // was 
  //String modelWeights = runtime_data_dir + "tiny-yolo.weights";   // was 
  // Load the network
  Net net = readNetFromDarknet(modelConfiguration, modelWeights);
  net.setPreferableBackend(DNN_BACKEND_OPENCV);
  net.setPreferableTarget(DNN_TARGET_CPU);

  // Open a video file or an image file or a camera stream.
  string str, outputFile;
  VideoCapture cap;
  VideoWriter video;
  Mat frame, blob;

  try {

    outputFile = runtime_data_dir + "yolo_out_cpp.avi";
    if (parser.has("image"))
    {
      // Open the image file
      //str = parser.get<String>("image");
      str = runtime_data_dir + "car2_car_part2.jpg";// "car2.jpg";
      ifstream ifile(str);
      if (!ifile) throw("error");
      cap.open(str);
      str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.jpg");
      outputFile = str;
    }
    else if (parser.has("video"))
    {
      // Open the video file
      //str = parser.get<String>("video"); // by sangkny
      //str = runtime_data_dir + "smuglanka.mp4"; //20180911_113611_cam_0
      str = runtime_data_dir + "overpass.mp4";// "20180911_113611_cam_0.avi";// "20180912_192157_cam_0.avi";//20180912_201357_cam_0.avi";// 20180911_113611_cam_0.avi"; //
      ifstream ifile(str);
      if (!ifile) throw("error");
      cap.open(str);
      str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
      outputFile = str;
    }
    // Open the webcaom
    else cap.open(parser.get<int>("device"));

  }
  catch (...) {
    cout << "Could not open the input image/video stream" << endl;
    return 0;
  }

  // Get the video writer initialized to save the output video
  if (!parser.has("image")) {
    video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
  }

  // Create a window
  static const string kWinName = "Deep learning object detection in OpenCV";
  namedWindow(kWinName, WINDOW_AUTOSIZE/*WINDOW_NORMAL*/);

  // Process frames.
  float scaleFactor = .5;
  while (waitKey(1) < 0)
  {
    // get frame from the video
    cap >> frame;
	
    // Stop the program if reached end of video
    if (frame.empty()) {
      cout << "Done processing !!!" << endl;
      cout << "Output file is stored as " << outputFile << endl;
      waitKey(3000);
      break;
    }
	//resize the input image
	resize(frame, frame, Size(), scaleFactor, scaleFactor);
	// detect object using DNN
	vector<Mat> outs;	
	regions_t res_regions=DetectInCrop(net, frame, Size(inpWidth,inpHeight), outs);

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);

    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("time for a frame : %.2f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    cout << label << endl;

    // Write the frame with the detection boxes
    Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);
    if (parser.has("image")) imwrite(outputFile, detectedFrame);
    else video.write(detectedFrame);

    imshow(kWinName, frame);

  }

  cap.release();
  if (!parser.has("image")) video.release();

  return 0;
}
// detect in crop
regions_t DetectInCrop(Net& net, cv::Mat& colorMat, cv::Size crop, vector<Mat>& outs) {
	cv::Mat blob;
	vector<int>  classIds; 
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> indices;
	regions_t tmpregions;
	blobFromImage(colorMat, blob, 1 / 255.0, crop, Scalar(0, 0, 0), false, true);
	vector<cv::Mat > Blobs;
	imagesFromBlob(blob, Blobs);
	for (int ii = 0; ii < Blobs.size(); ii++) {
		imshow(format("blob image: %d", ii), Blobs[ii]);
		waitKey(1);
	}
	//Sets the input to the network
	net.setInput(blob);
	// Runs the forward pass to get output of the output layers	
	net.forward(outs, getOutputsNames(net)); // forward pass for network layers	

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
			if (confidence > confThreshold)
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
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);	
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		tmpregions.push_back(CRegion(box, classes[classIds[idx]], confidences[idx]));
	}
	return tmpregions;
}

void getPredicInfo(const vector<Mat>& outs, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes) {

}
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
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
      if (confidence > confThreshold)
      {
        int centerX = (int)(data[0] * frame.cols); 
        int centerY = (int)(data[1] * frame.rows);
        int width = (int)(data[2] * frame.cols);
        int height = (int)(data[3] * frame.rows);
        int left = centerX - width / 2;				// adjust to Rect structure
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
  NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
  for (size_t i = 0; i < indices.size(); ++i)
  {
    int idx = indices[i];
    Rect box = boxes[idx];
     drawPred(classIds[idx], confidences[idx], box.x, box.y,
      box.x + box.width, box.y + box.height, frame);    
  }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
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
