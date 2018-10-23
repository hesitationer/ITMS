// Blob.cpp

#include "itms_Blob.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
namespace itms {

  Blob::Blob(std::vector<cv::Point> _contour) {

    currentContour = _contour;

    currentBoundingRect = cv::boundingRect(currentContour);

    cv::Point currentCenter;

    currentCenter.x = (currentBoundingRect.x + currentBoundingRect.x + currentBoundingRect.width) / 2;
    currentCenter.y = (currentBoundingRect.y + currentBoundingRect.y + currentBoundingRect.height) / 2;

    centerPositions.push_back(currentCenter);

    dblCurrentDiagonalSize = sqrt(pow(currentBoundingRect.width, 2) + pow(currentBoundingRect.height, 2));

    dblCurrentAspectRatio = (float)currentBoundingRect.width / (float)currentBoundingRect.height;

    blnStillBeingTracked = true;
    blnCurrentMatchFoundOrNewBlob = true;

    intNumOfConsecutiveFramesWithoutAMatch = 0;

    age = 1;
    totalVisibleCount = 1;
    id = 0; // track id
    showId = 0;
    // object status information
    oc = OC_OTHER;
    os = OS_NOTDETERMINED;
    od = OD_ND; // lane direction will affect the result, and the lane direction will be given
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  void Blob::predictNextPosition(void) {

    int numPositions = (int)centerPositions.size();

    if (numPositions == 1) {

      predictedNextPosition.x = centerPositions.back().x;
      predictedNextPosition.y = centerPositions.back().y;

    }
    else if (numPositions == 2) {

      int deltaX = centerPositions[1].x - centerPositions[0].x;
      int deltaY = centerPositions[1].y - centerPositions[0].y;

      predictedNextPosition.x = centerPositions.back().x + deltaX;
      predictedNextPosition.y = centerPositions.back().y + deltaY;

    }
    else if (numPositions == 3) {

      int sumOfXChanges = ((centerPositions[2].x - centerPositions[1].x) * 2) +
        ((centerPositions[1].x - centerPositions[0].x) * 1);

      int deltaX = (int)std::round((float)sumOfXChanges / 3.0);

      int sumOfYChanges = ((centerPositions[2].y - centerPositions[1].y) * 2) +
        ((centerPositions[1].y - centerPositions[0].y) * 1);

      int deltaY = (int)std::round((float)sumOfYChanges / 3.0);

      predictedNextPosition.x = centerPositions.back().x + deltaX;
      predictedNextPosition.y = centerPositions.back().y + deltaY;

    }
    else if (numPositions == 4) {

      int sumOfXChanges = ((centerPositions[3].x - centerPositions[2].x) * 3) +
        ((centerPositions[2].x - centerPositions[1].x) * 2) +
        ((centerPositions[1].x - centerPositions[0].x) * 1);

      int deltaX = (int)std::round((float)sumOfXChanges / 6.0);

      int sumOfYChanges = ((centerPositions[3].y - centerPositions[2].y) * 3) +
        ((centerPositions[2].y - centerPositions[1].y) * 2) +
        ((centerPositions[1].y - centerPositions[0].y) * 1);

      int deltaY = (int)std::round((float)sumOfYChanges / 6.0);

      predictedNextPosition.x = centerPositions.back().x + deltaX;
      predictedNextPosition.y = centerPositions.back().y + deltaY;

    }
    else if (numPositions >= 5) {

      int sumOfXChanges = ((centerPositions[numPositions - 1].x - centerPositions[numPositions - 2].x) * 4) +
        ((centerPositions[numPositions - 2].x - centerPositions[numPositions - 3].x) * 3) +
        ((centerPositions[numPositions - 3].x - centerPositions[numPositions - 4].x) * 2) +
        ((centerPositions[numPositions - 4].x - centerPositions[numPositions - 5].x) * 1);

      int deltaX = (int)std::round((float)sumOfXChanges / 10.0);

      int sumOfYChanges = ((centerPositions[numPositions - 1].y - centerPositions[numPositions - 2].y) * 4) +
        ((centerPositions[numPositions - 2].y - centerPositions[numPositions - 3].y) * 3) +
        ((centerPositions[numPositions - 3].y - centerPositions[numPositions - 4].y) * 2) +
        ((centerPositions[numPositions - 4].y - centerPositions[numPositions - 5].y) * 1);

      int deltaY = (int)std::round((float)sumOfYChanges / 10.0);

      predictedNextPosition.x = centerPositions.back().x + deltaX;
      predictedNextPosition.y = centerPositions.back().y + deltaY;

    }
    else {
      // should never get here
    }

  }
  void Blob::operator=(const Blob &rhBlob) { 
	  currentContour.clear();
	  for (int i = 0; i < rhBlob.currentContour.size(); i++)
		  currentContour.push_back(rhBlob.currentContour.at(i));

	  currentBoundingRect = rhBlob.currentBoundingRect;

	  centerPositions.clear();
	  centerPositions.push_back(rhBlob.centerPositions.back());


	  dblCurrentDiagonalSize = rhBlob.dblCurrentDiagonalSize;

	  dblCurrentAspectRatio = rhBlob.dblCurrentAspectRatio;

	  blnStillBeingTracked = rhBlob.blnStillBeingTracked;
	  blnCurrentMatchFoundOrNewBlob = rhBlob.blnCurrentMatchFoundOrNewBlob;

	  intNumOfConsecutiveFramesWithoutAMatch = rhBlob.intNumOfConsecutiveFramesWithoutAMatch;

	  age = rhBlob.age;
	  totalVisibleCount = rhBlob.totalVisibleCount;
	  showId = rhBlob.showId;
	  // object status information
	  oc = rhBlob.oc;
	  os = rhBlob.os;
	 od = rhBlob.od; // lane direction will affect the result, and the lane direction will be given
  }

  cv::Point Blob::weightedPositionAverage(bool bWeighted) {
    cv::Point wpa(cv::Point(0, 0));
    //int maxTap = numMaxTap;
    int numPositions = (int)centerPositions.size();

    if (numPositions == 1) {

      wpa.x = centerPositions.back().x;
      wpa.y = centerPositions.back().y;

    }
    else if (numPositions == 2) {

      int deltaX = (bWeighted)? 2*centerPositions[1].x + centerPositions[0].x : centerPositions[1].x + centerPositions[0].x; // 
      deltaX = (bWeighted) ? std::round((float)deltaX / 3.0) : std::round((float)deltaX / 2.0);

      int deltaY = (bWeighted)? 2*centerPositions[1].y + centerPositions[0].y : centerPositions[1].y + centerPositions[0].y;
      deltaY = (bWeighted) ? std::round((float)deltaY / 3.0) : std::round((float)deltaY / 2.0);
      wpa = cv::Point(deltaX,deltaY);
    }
    else if (numPositions == 3) {

      int sumOfXChanges = (bWeighted)? (3*centerPositions[2].x +2* centerPositions[1].x + centerPositions[0].x): (centerPositions[2].x + centerPositions[1].x + centerPositions[0].x);

      int deltaX = (bWeighted)? (int)std::round((float)sumOfXChanges / 6.0) : (int)std::round((float)sumOfXChanges / 3.0);

      int sumOfYChanges = (bWeighted) ? (3 * centerPositions[2].y + 2 * centerPositions[1].y + centerPositions[0].y) : (centerPositions[2].y + centerPositions[1].y + centerPositions[0].y);
      int deltaY = (bWeighted) ? (int)std::round((float)sumOfYChanges / 6.0) : (int)std::round((float)sumOfYChanges / 3.0);

      wpa = cv::Point(deltaX, deltaY);

    }
    else if (numPositions == 4) {

      int sumOfXChanges = (bWeighted) ? (4*centerPositions[3].x + 3 * centerPositions[2].x + 2 * centerPositions[1].x + centerPositions[0].x) : (centerPositions[3].x + centerPositions[2].x + centerPositions[1].x + centerPositions[0].x);

      int deltaX = (bWeighted) ? (int)std::round((float)sumOfXChanges / 10.0) : (int)std::round((float)sumOfXChanges / 4.0);

      int sumOfYChanges = (bWeighted) ? (4*centerPositions[3].y + 3 * centerPositions[2].y + 2 * centerPositions[1].y + centerPositions[0].y) : (centerPositions[3].y + centerPositions[2].y + centerPositions[1].y + centerPositions[0].y);
      int deltaY = (bWeighted) ? (int)std::round((float)sumOfYChanges / 10.0) : (int)std::round((float)sumOfYChanges / 4.0);

      wpa = cv::Point(deltaX, deltaY);

    }
    else if (numPositions >= 5) {

      int sumOfXChanges = (bWeighted)? (5*centerPositions[numPositions - 1].x + 4*centerPositions[numPositions - 2].x + 3*centerPositions[numPositions - 3].x + 2*centerPositions[numPositions - 4].x+ centerPositions[numPositions - 5].x)
        : (centerPositions[numPositions - 1].x + centerPositions[numPositions - 2].x + centerPositions[numPositions - 3].x + centerPositions[numPositions - 4].x + centerPositions[numPositions - 5].x);

      int deltaX = (bWeighted) ? (int)std::round((float)sumOfXChanges / 15.0) : (int)std::round((float)sumOfXChanges / 5.0);

      int sumOfYChanges = (bWeighted) ? (5 * centerPositions[numPositions - 1].y + 4 * centerPositions[numPositions - 2].y + 3 * centerPositions[numPositions - 3].y + 2 * centerPositions[numPositions - 4].y + centerPositions[numPositions - 5].y)
        : (centerPositions[numPositions - 1].y + centerPositions[numPositions - 2].y + centerPositions[numPositions - 3].y + centerPositions[numPositions - 4].y + centerPositions[numPositions - 5].y);

      int deltaY = (bWeighted) ? (int)std::round((float)sumOfYChanges / 15.0) : (int)std::round((float)sumOfYChanges / 5.0);

      wpa = cv::Point(deltaX, deltaY);

    }
    else {
      // should never get here
    }

    return wpa;
  }
    
} // end of namespace itms
