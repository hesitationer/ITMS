#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
const int KEY_SPACE = 32;
const int KEY_ESC = 27;

//CvHaarClassifierCascade *cascade;
cv::CascadeClassifier cascade;
cv::HOGDescriptor hog;
float scaleFactor = .5;

void detect(Mat img);
inline bool existFileTest(const std::string& name);

int main(int argc, char** argv)
{
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	//CvCapture *capture;

	//IplImage  *frame; // opencv 2.x
	Mat frame; // opencv 3.x
	int input_resize_percent = 100;
	std::string runtime_data_dir = "D:/LectureSSD_rescue/project-related/road-weather-topes/code/ITMS/";
	std::string xmlFile = runtime_data_dir + "config/cascade.xml"; // cars.xml with 1 neighbors good, cascade.xml with 5 neighbors, people cascadG.xml(too many PA) with 4 neighbors and size(30,80), size(80,200)
// http://funvision.blogspot.com/2016/12/lbp-cascade-for-car-detection-in-opencv.html
	std::string videoFile = runtime_data_dir + "TrafficVideo/20180911_113611_cam_0.avi"; //20180912_112338_cam_0  // 20180911_113611_cam_0 // 20180911_113611_cam_0


																						 /*if(argc < 3)
																						 {
																						 std::cout << "Usage " << argv[0] << " cascade.xml video.avi" << std::endl;
																						 return 0;
																						 }

																						 if(argc == 4)
																						 {
																						 input_resize_percent = atoi(argv[3]);
																						 std::cout << "Resizing to: " << input_resize_percent << "%" << std::endl;
																						 }*/

																						 //cascade = (CvHaarClassifierCascade*) cvLoad(xmlFile.c_str()/*argv[1]*/, 0, 0, 0);
	if (!cascade.load(xmlFile)) {
		std::cout << "Plase check the xml file in the given location !!(!)\n";
		std::cout << xmlFile << std::endl;
		return 0;
	}

	//storage = cvCreateMemStorage(0);
	bool bfile = false;
	bfile = existFileTest(videoFile);
	std::cout << " file name: " << videoFile.c_str() << std::endl;
	VideoCapture capture(videoFile);
	//capture = cvCaptureFromAVI(videoFile.c_str()/*argv[2]*/);
	// hog
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	if (!capture.isOpened()) {
		std::cout << " the video file does not exist!!\n"; // iostream
		return 0;
	}


	//assert(cascade /*&& storage*/ && capture);

	cvNamedWindow("video", 1);

	//IplImage* frame1 = cvQueryFrame(capture);
	//frame = cvCreateImage(cvSize((int)((frame1->width*input_resize_percent) / 100), (int)((frame1->height*input_resize_percent) / 100)), frame1->depth, frame1->nChannels);
	Mat frame1;
	//capture >> frame1;  


	int key = 0;
	do
	{
		//frame1 = cvQueryFrame(capture);
		capture >> frame1;
		if (frame1.empty()) {
			cout << "frame is not valid or end of video !!" << endl;
			break;
		}
		//cvResize(frame1, frame);
		resize(frame1, frame, Size(), scaleFactor, scaleFactor, 1);
		detect(frame);

		key = waitKey(100);

		if (key == KEY_SPACE)
			key = cvWaitKey(0);

		if (key == KEY_ESC)
			break;

	} while (1);

	/*cvDestroyAllWindows();
	cvReleaseImage(&frame);
	cvReleaseCapture(&capture);
	cvReleaseHaarClassifierCascade(&cascade);
	cvReleaseMemStorage(&storage);*/

	return 0;
}

void detect(cv::Mat img)
{
	CvSize img_size = img.size();
	vector<Rect> object;
	vector<Rect> people;
	// opencv 2.4.x
	//CvSeq *object = cvHaarDetectObjects(
	//  img,
	//  cascade,
	//  storage,
	//  1.1, //1.1,//1.5, //-------------------SCALE FACTOR
	//  1, //2        //------------------MIN NEIGHBOURS
	//  0, //CV_HAAR_DO_CANNY_PRUNING
	//  cvSize(0,0),//cvSize( 30,30), // ------MINSIZE
	//  img_size //cvSize(70,70)//cvSize(640,480)  //---------MAXSIZE
	//  );
	// opencv 3.x 
	cascade.detectMultiScale(img, object, 1.1, 5/*1  cascadG */, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(0, 0), img_size); // detectio objects (car)
																														//cascade.detectMultiScale(img, object, 1.04, 5, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(3, 8), img_size); // detectio objects (people)
	hog.detectMultiScale(img, people, 0, Size(4, 4), Size(8, 8), 1.05, 2, false);							// detect people

	std::cout << "Total: " << object.size() << " cars detected." << std::endl;

	for (int i = 0; i < (object.size() ? object.size()/*object->total*/ : 0); i++)
	{
		//CvRect *r = (CvRect*)cvGetSeqElem(object, i);
		Rect r = object.at(i);
		cv::rectangle(img,
			cv::Point(r.x, r.y),
			cv::Point(r.x + r.width, r.y + r.height),
			CV_RGB(255, 0, 0), 2, 8, 0);
	}
	std::cout << "=>=> " << people.size() << " people detected." << std::endl;
	for (int i = 0; i < (people.size() ? people.size()/*object->total*/ : 0); i++)
	{
		//CvRect *r = (CvRect*)cvGetSeqElem(object, i);
		Rect r1 = people.at(i);
		cv::rectangle(img,
			cv::Point(r1.x, r1.y),
			cv::Point(r1.x + r1.width, r1.y + r1.height),
			CV_RGB(0, 255, 0), 2, 8, 0);
	}

	imshow("video", img);
	waitKey(1);
}
inline bool existFileTest(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}