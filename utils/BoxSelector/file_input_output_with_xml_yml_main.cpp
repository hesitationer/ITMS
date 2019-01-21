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
	}
	{
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
}