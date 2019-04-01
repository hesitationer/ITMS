// itms_APINativeClass_main.cpp
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
//#include <opencv2/dnn.hpp>

#include<iostream>			// cout etc
#include<fstream>			// file stream (i/ofstream) etc

#include <sstream>
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows
#include <time.h>

#include "./utils/itms_utils.h"


#define SHOW_STEPS            // un-comment or comment this line to show steps or not
//#define HAVE_OPENCV_CONTRIB
#ifdef HAVE_OPENCV_CONTRIB
#include <opencv2/video/background_segm.hpp>
using namespace cv::bgsegm;
#endif
using namespace cv;
using namespace std;
using namespace itms;


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


int main(void) {
#ifdef _sk_Memory_Leakag_Detector
#if _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif	
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	// -------------------------------------- HOW TO USE THE API --------------------------------------------
	itms::ITMSAPINativeClass *itmsNativeClass = new itms::ITMSAPINativeClass; 
	// constructor should not contain the Init() function
	itmsNativeClass->Init();
	// --------------------------------------------------------------------------------------------------------
	
	// video source loading
	cv::VideoCapture capVideo;

	cv::Mat imgFrame1; // previous frame
	cv::Mat imgFrame2; // current frame

  	int carCount = 0;
	int truckCount = 0;
	int bikeCount = 0;
	int humanCount = 0;
	int videoLength = 0;  
	
	//bool b = capVideo.open(conf.VideoPath);  
	bool b = capVideo.open(itmsNativeClass->conf.VideoPath);
      
	if (!capVideo.isOpened()) {                                                 // if unable to open video file
		std::cout << "reading video file error (!)" << std::endl << std::endl;      // show error message
		_getch();                   // it may be necessary to change or remove this line if not using Windows
		return(0);                                                              // and exit program
	}

  int max_frames = capVideo.get(CV_CAP_PROP_FRAME_COUNT);

	if ( max_frames< 2) {
		std::cout << "error: video file must have at least two frames" << std::endl;
		_getch();                   // it may be necessary to change or remove this line if not using Windows
		return(0);
	}
	/* Event Notice */
	std::cout << "Press 'ESC' to quit..." << std::endl;

	// video information
	int fps = 15;
	bool hasFile = true;  
	if (hasFile)
	{
		fps = int(capVideo.get(CAP_PROP_FPS));
		itmsNativeClass->conf.fps = fps;
		cout << "Video FPS: " << fps << endl;
	}
	
    capVideo.read(imgFrame1);    
    capVideo.read(imgFrame2);
    if (imgFrame1.empty() || imgFrame2.empty())
      return 0; 


    char chCheckForEscKey = 0;
	bool bsaveEventFile = true;
	float fscaleFractor = 0.5;
    bool blnFirstFrame = true;
	int m_startFrame = 0;	 // 240
    int frameCount = m_startFrame + 1;	    
	int PlayInterval = 1;                // make it increase if you want to speed up !!
	PlayInterval = std::max(1, PlayInterval);

	capVideo.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);
        
	
    while (capVideo.isOpened() && chCheckForEscKey != 27) {

		double t1 = (double)cvGetTickCount();   
  //      
		// -------------------------------------- HOW TO USE THE API --------------------------------------------
		
		itmsNativeClass->ResetAndProcessFrame(imgFrame2);		
		// check the object events 		
		if(itmsNativeClass->getObjectClass().size()){
			cout<< "///////////// object events occurred /////////////\n";
			cv::Mat debugImg = imgFrame2.clone();
			for (size_t i = 0; i < itmsNativeClass->getObjectClass().size(); i++) {
				cout << "objID: "<< itmsNativeClass->getObjectClass().at(i).first << ", class: "<< itmsNativeClass->getObjectClass().at(i).second<<endl;
				cout << "Status: " << itmsNativeClass->getObjectStatus().at(i).second << endl;
				cout << "Speed: " << itmsNativeClass->getObjectSpeed().at(i) << endl;
				cout << "Rect: " << itmsNativeClass->getObjectRect().at(i) << endl;
				if (bsaveEventFile) {					
					int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
					double dblFontScale = 2; //max(1., blobs[i].dblCurrentDiagonalSize / 60.0);
					int intFontThickness = (int)std::round(dblFontScale * 1.0);
					cv::Rect d_rect = itmsNativeClass->getObjectRect().at(i);
					d_rect.x = float(d_rect.x)/fscaleFractor;
					d_rect.y = float(d_rect.y)/fscaleFractor;
					d_rect.width = float(d_rect.width)/fscaleFractor;
					d_rect.height = float(d_rect.height)/fscaleFractor;

					string infostr;
					infostr.clear();
					infostr = "id:"+ std::to_string(itmsNativeClass->getObjectClass().at(i).first) + ", class:" + std::to_string(itmsNativeClass->getObjectClass().at(i).second)
						+", status:" + std::to_string(itmsNativeClass->getObjectStatus().at(i).second)+", speed:"+std::to_string(itmsNativeClass->getObjectSpeed().at(i));
					
					cv::putText(debugImg, infostr/*std::to_string(blobs[i].id)*/, cv::Point(10, 60*(i+1))/*d_rect.tl()*/, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
					cv::rectangle(debugImg, d_rect, SCALAR_BLUE, 2);
					cv::imshow("detected event", debugImg);
					cv::waitKey(1);
				}
			}
			if(bsaveEventFile){
				string saveFileName = itmsNativeClass->conf.VideoPath;
				saveFileName.append("-" +std::to_string(frameCount) +"_.jpg");
				cv::imwrite(saveFileName, debugImg);
			}
		}
		// -------------------------------------------------------------------------------------------------------
       // now we prepare for the next iteration
        imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
            for(int pI=0; pI<PlayInterval;pI++)
				capVideo.read(imgFrame2);
			frameCount = frameCount + (PlayInterval-1);
			
            ////resize(imgFrame2, imgFrame2, Size(), conf.scaleFactor, conf.scaleFactor);
            if (imgFrame2.empty()) {
              std::cout << "The input image is empty!! Please check the video file!!" << std::endl;
              _getch();
              break;
            }
        } else {
            std::cout << "end of video (!) \n";
            break;
        }

        blnFirstFrame = false;
        frameCount++;		
        chCheckForEscKey = cv::waitKey(1);
		if (itmsNativeClass->conf.debugTime) {
			double t2 = (double)cvGetTickCount();
			double t3 = (t2 - t1) / (double)(getTickFrequency()*PlayInterval);
			cout << "Processing time>>  #:" << (frameCount - 1) <<"/("<< max_frames<<")"<< " --> " << t3*1000.0<<"msec, "<< 1./t3 << "fps \n";			
		}
    }

    if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
        cv::waitKey(3000);                         // hold the windows open to allow the "end of video" message to show
		cv::destroyAllWindows();
    }
    // note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows
#ifdef _sk_Memory_Leakag_Detector
#ifdef _DEBUG
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
#endif // _sk_Memory_Leakag_Detector	
	if (itmsNativeClass)
		delete itmsNativeClass;
    return(0);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
