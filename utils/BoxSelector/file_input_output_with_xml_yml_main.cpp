#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;


int main() {
	String filename = "RoadMapPoints_20190430.xml";
	
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
		std::string lutFile = "vehicleRatio_20190506.xml";
		FileStorage fw(lutFile, FileStorage::WRITE);
		std::vector<Mat> vecMat;
		//std::vector<float> sedan_h = { -0.00004382882987f*2.0, 0.0173377779098603f*2.0, -2.28169958272933f*2.0, 112.308488612836f*2.0 }; // scale factor 0.5 need to go header
		//std::vector<float> sedan_w = { -0.00003795583601f*2.0, 0.0146676803999458f*2.0, -1.91394276754689f*2.0, 98.7860907208733f*2.0 };  // 20190430
		std::vector<float> sedan_h = { -0.000039987980491413f*2.0, 0.0161278690798142f*2.0, -2.15112517789863f*2.0, 109.244494635799f*2.0}; // scale factor 0.5 need to go header
		std::vector<float> sedan_w = { -0.0000383449290771945f*2.0, 0.0148587047328466f*2.0, -1.91739512923723f*2.0, 99.1472802559759f*2.0}; // 20190506
		std::vector<float> suv_h = { -0.0000573893011f*2.0, 0.02198602567f*2.0, -2.786735669f*2.0, 138.9535103f*2.0 };
		std::vector<float> suv_w = { -0.00004930887826f*2.0, 0.0190299365f*2.0, -2.436554248f*2.0, 122.0330322f*2.0 };
		std::vector<float> truck_h = { -0.00006180993767f*2.0, 0.02390822247f*2.0, -3.076351259f*2.0, 149.7855261f*2.0 };
		std::vector<float> truck_w = { -0.00003778247771f*2.0, 0.015239317f*2.0, -2.091105041f*2.0, 110.7544702f*2.0 };
		std::vector<float> human_h = { -0.000003756096433f*2.0, 0.002062517955f*2.0, -0.4861445445f*2.0, 48.88594015f*2.0 };
		std::vector<float> human_w = { -0.000006119547882f*2.0, 0.002164848881f*2.0, -0.3171686628f*2.0, 27.98164879f*2.0 };

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