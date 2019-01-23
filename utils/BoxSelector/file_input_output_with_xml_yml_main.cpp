#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;


int main() {
	String filename = "RoadMapPoints.xml";
	
	{
		FileStorage fs(filename, FileStorage::WRITE);
		std::vector<Point2f> _boundingBoxPts = { {0,0}, {0,1}, {0,2},{0,3} }; // a box bounding points

		std::vector<std::vector<cv::Point2f>> _boundingBoxes; // bunch of boxes for specfic region boundaries
		_boundingBoxes.push_back(_boundingBoxPts);
		_boundingBoxes.push_back(_boundingBoxPts);
		_boundingBoxes.push_back(_boundingBoxPts);
		std::vector<Mat> vecMat;
		//Mat A(2, 4, CV_32F, Scalar(0));		
		Mat A(_boundingBoxPts);
		vecMat.push_back(A);
		vecMat.push_back(A);
		vecMat.push_back(A);
		//ostringstream oss;
		for (int i = 0; i < vecMat.size(); i++) {
			stringstream ss;
			ss << i;
			string str = "Road" + ss.str();
			fs << str << vecMat[i];
		}
		fs.release();
		// cvFileStorage
		string cvfilename = "CvFile_"+filename;
		CvFileStorage* storage = cvOpenFileStorage(cvfilename.c_str(), 0, CV_STORAGE_WRITE);
		//cvWrite(storage, "Boxes", vecMat[0].datastart, cvAttrList(0,0));
		if(storage)
			cvReleaseFileStorage(&storage);
	}
    {// read the file information
		std::vector<Point2f> _boundingBoxPts; // a box bounding points
		std::vector<std::vector<cv::Point2f>> _boundingBoxes; // bunch of boxes for specfic region boundaries
		vector<Mat> matVecRead;
		FileStorage fr(filename, FileStorage::READ);
		Mat aMat;
		int countlabel = 0;
		while (1) {
			_boundingBoxPts.clear();
			stringstream ss;
			ss << countlabel;
			string str = "Road" + ss.str();
			cout << str << endl;
			fr[str] >> aMat;
			if (fr[str].isNone() == 1) {
				break;
			}
			for(int i=0;i<aMat.rows;i++)
				_boundingBoxPts.push_back(Point(aMat.at<Vec2f>(i)));
			_boundingBoxes.push_back(_boundingBoxPts);
			matVecRead.push_back(aMat.clone());
			countlabel++;
		}
		fr.release();
		for (unsigned j = 0; j < matVecRead.size(); j++) {
			cout << "Mat:" << endl;
			cout << matVecRead[j] << endl;
			cout << "Vec:" << endl;
			cout << _boundingBoxes.at(j) << endl;
		}
	}
	{ // vehicle LUT generation for TOPES
		std::string lutFile = "vehicleRatio.xml";
		FileStorage fw(lutFile, FileStorage::WRITE);
		std::vector<Mat> vecMat;
		std::vector<float> sedan_h = { -0.00004444328872f*2.0, 0.01751602326f*2.0, -2.293443176f*2.0, 112.527668f*2.0 }; // scale factor 0.5 need to go header
		std::vector<float> sedan_w = { -0.00003734137716f*2.0, 0.01448943505f*2.0, -1.902199174f*2.0, 98.56691135f*2.0 };
		std::vector<float> suv_h = { -0.00005815785621f*2.0, 0.02216859672f*2.0, -2.797603666f*2.0, 139.0638999f*2.0 };
		std::vector<float> suv_w = { -0.00004854032314f*2.0, 0.01884736545f*2.0, -2.425686251f*2.0, 121.9226426f*2.0 };
		std::vector<float> truck_h = { -0.00006123592908f*2.0, 0.02373661426f*2.0, -3.064585294f*2.0, 149.6535855f*2.0 };
		std::vector<float> truck_w = { -0.00003778247771f*2.0, 0.015239317f*2.0, -2.091105041f*2.0, 110.7544702f*2.0 };
		std::vector<float> human_h = { -0.000002473245036f*2.0, 0.001813179193f*2.0, -0.5058008988f*2.0, 49.27950311f*2.0 };
		std::vector<float> human_w = { -0.000003459461125f*2.0, 0.001590306464f*2.0, -0.3208648543f*2.0, 28.23621306f*2.0 };

		vecMat.push_back(cv::Mat(sedan_h));
		vecMat.push_back(cv::Mat(sedan_w));
		vecMat.push_back(cv::Mat(suv_h));
		vecMat.push_back(cv::Mat(suv_w));
		vecMat.push_back(cv::Mat(truck_h));
		vecMat.push_back(cv::Mat(truck_w));
		vecMat.push_back(cv::Mat(human_h));
		vecMat.push_back(cv::Mat(human_w));
		
		//ostringstream oss;
		for (int i = 0; i < vecMat.size(); i++) {
			stringstream ss;
			ss << i;
			string str = "Vehicle" + ss.str();
			fw << str << vecMat[i];
			cout << i<<": " << vecMat[i] << endl;
		}
		fw.release();
	}
}