
#include "./utils/itms_utils.h"

namespace itms {
	
  void imshowBeforeAndAfter(cv::Mat &before, cv::Mat &after, std::string windowtitle, int gabbetweenimages)
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

  ITMSVideoWriter::ITMSVideoWriter(bool writeToFile, const char* filename, int codec, double fps, Size frameSize, bool color) {
    this->writeToFile = writeToFile;
    if (writeToFile) {
      writer.open(filename, codec, fps, frameSize, color);
    }
  }

  void ITMSVideoWriter::write(Mat& frame) {
    if (writeToFile) {
      writer.write(frame);
    }
  }
  Rect expandRect(Rect original, int expandXPixels, int expandYPixels, int maxX, int maxY)
  {
    Rect expandedRegion = Rect(original);

    float halfX = round((float)expandXPixels / 2.0);
    float halfY = round((float)expandYPixels / 2.0);
    expandedRegion.x = expandedRegion.x - halfX;
    expandedRegion.width = expandedRegion.width + expandXPixels;
    expandedRegion.y = expandedRegion.y - halfY;
    expandedRegion.height = expandedRegion.height + expandYPixels;

    expandedRegion.x = std::min(std::max(expandedRegion.x, 0), maxX);

    expandedRegion.y = std::min(std::max(expandedRegion.y, 0), maxY);
    if (expandedRegion.x + expandedRegion.width > maxX)
      expandedRegion.width = maxX - expandedRegion.x;
    if (expandedRegion.y + expandedRegion.height > maxY)
      expandedRegion.height = maxY - expandedRegion.y;

    return expandedRegion;
  }
  Rect maxSqRect(Rect& original, int maxX, int maxY) {
    int intDifLength = original.width -original.height;    
    Rect expandedRegion = (intDifLength > 0) ? expandRect(original, 0, intDifLength, maxX, maxY) : expandRect(original, -1 * intDifLength, 0, maxX, maxY);
    return expandedRegion;  
  }
  
  Rect maxSqExpandRect(Rect& original, float floatScalefactor, int maxX, int maxY) {
    Rect maxSq = maxSqRect(original, maxX, maxY);
    maxSq = expandRect(maxSq, floatScalefactor*maxSq.width, floatScalefactor*maxSq.height, maxX, maxY);
    return maxSq;
  }
 
  ///////////////////////////////////////////////////////////////////////////////
  // main functions body
  bool isPointBelowLine(cv::Point sP, cv::Point eP, cv::Point tP) {
	  return ((eP.x - sP.x)*(tP.y - sP.y) - (eP.y - sP.y)*(tP.x - sP.x)) > 0;
  }

  void copyBlob2Blob(Blob &srcBlob, Blob &tgtBlob) {

	  tgtBlob.currentContour.clear();
	  for (int i = 0; i < srcBlob.currentContour.size(); i++)
		  tgtBlob.currentContour.push_back(srcBlob.currentContour.at(i));

	  tgtBlob.currentBoundingRect = srcBlob.currentBoundingRect;

	  tgtBlob.centerPositions.clear();
	  for (int i = 0; i<srcBlob.centerPositions.size(); i++)
		  tgtBlob.centerPositions.push_back(srcBlob.centerPositions.at(i));


	  tgtBlob.dblCurrentDiagonalSize = srcBlob.dblCurrentDiagonalSize;

	  tgtBlob.dblCurrentAspectRatio = srcBlob.dblCurrentAspectRatio;

	  tgtBlob.blnStillBeingTracked = srcBlob.blnStillBeingTracked;
	  tgtBlob.blnCurrentMatchFoundOrNewBlob = srcBlob.blnCurrentMatchFoundOrNewBlob;

	  tgtBlob.intNumOfConsecutiveFramesWithoutAMatch = srcBlob.intNumOfConsecutiveFramesWithoutAMatch;

	  tgtBlob.age = srcBlob.age;
	  tgtBlob.totalVisibleCount = srcBlob.totalVisibleCount;
//	  tgtBlob.showId = srcBlob.showId;	  
	  // object status information
	  tgtBlob.oc = srcBlob.oc;
	  tgtBlob.os = srcBlob.os;
	  tgtBlob.od = srcBlob.od; // lane direction will affect the result, and the lane direction will be given
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  void mergeBlobsInCurrentFrameBlobs(itms::Config& _conf, std::vector<Blob> &currentFrameBlobs) {
	  std::vector<Blob>::iterator currentBlob = currentFrameBlobs.begin();
	  while (currentBlob != currentFrameBlobs.end()) {
		  int intIndexOfLeastDistance = -1;
		  double dblLeastDistance = 100000.0;
		  for (unsigned int i = 0; i < currentFrameBlobs.size(); i++) {

			  double dblDistance = distanceBetweenPoints(currentBlob->centerPositions.back(), currentFrameBlobs[i].centerPositions.back());

			  if (dblDistance > 1/* same object */ && dblDistance < dblLeastDistance) { // center locations should be in the range
				  dblLeastDistance = dblDistance;
				  intIndexOfLeastDistance = i;
			  }
		  }
		  // at this point we have nearest countours
		  if (intIndexOfLeastDistance < 0) {
			  ++currentBlob;
			  continue;
		  }

		  // check the conditions
		  if (dblLeastDistance < currentBlob->dblCurrentDiagonalSize*1.25/*should be car size */) {
			  cv::Rect cFBrect = currentFrameBlobs[intIndexOfLeastDistance].currentBoundingRect;
			  Point cB = currentBlob->centerPositions.back();
			  bool flagMerge = false;
			  if (_conf.ldirection == LD_EAST || _conf.ldirection == LD_WEST /* horizontal*/) {
				  if ((cB.y >= cFBrect.y - (round)((float)(cFBrect.height) / 2.)) && (cB.y <= cFBrect.y + (round)((float)(cFBrect.height) / 2.)))
					  flagMerge = true;
			  }
			  else { // other lane direction only considers width and its center point
				  if ((cB.x >= cFBrect.x - (round)((float)(cFBrect.width) / 2.)) && (cB.x <= cFBrect.x + (round)((float)(cFBrect.width) / 2.)))
					  flagMerge = true;
			  }
			  // merge and erase index blob
			  if (flagMerge) {
				  if (_conf.debugGeneralDetail)
					  cout << "mergeing with " << to_string(intIndexOfLeastDistance) << " in blob" << currentBlob->centerPositions.back() << endl;

				  // countour merging
				  std::vector<cv::Point> points, contour;
				  points.insert(points.end(), currentBlob->currentContour.begin(), currentBlob->currentContour.end());
				  points.insert(points.end(), currentFrameBlobs[intIndexOfLeastDistance].currentContour.begin(), currentFrameBlobs[intIndexOfLeastDistance].currentContour.end());
				  convexHull(cv::Mat(points), contour);
				  itms::Blob blob(contour);
				  *currentBlob = blob;      // class Blob = operator overloading
											//copyBlob2Blob(blob, *currentBlob);
				  std::vector<Blob>::iterator tempBlob = currentFrameBlobs.begin();
				  currentBlob = currentFrameBlobs.erase(tempBlob + intIndexOfLeastDistance);
				  // do something after real merge
				  continue;
			  }

		  }
		  ++currentBlob;
	  }


  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  void mergeBlobsInCurrentFrameBlobsWithPredictedBlobs(std::vector<Blob> &currentFrameBlobs, std::vector<Blob> &predBlobs) {
	  // 0. assuming blobs in currentFrameBlobs are separated each other in contours
	  // 1. find the closest blob
	  // 2. check the intersection which has more than overlapRatio in area, 오버랩이 70%이상이면 넣지마라. 새로운 것에 있으니 하지만 그 이하면 넣어라=> 버그였음.
	  // 3. otherwise add new one if there is no matched blob
	  float overlapRatio = 0.5;
	  bool addNewBlob = false;

	  std::vector<int> predBlobInx;
	  for (int prblob = 0; prblob < predBlobs.size(); prblob++) {
		  addNewBlob = false;
		  cv::Rect r1 = predBlobs.at(prblob).currentBoundingRect;
		  int maxIdx = -1, maxArea = -1;
		  for (int i = 0; i < currentFrameBlobs.size(); i++) {
			  cv::Rect r2 = currentFrameBlobs.at(i).currentBoundingRect;
			  int Area = (r1& r2).area(); // intersection test and its area
			  if (Area > 0 && Area > maxArea) {
				  maxArea = Area;
				  maxIdx = i;
			  }
		  }
		  if (maxIdx != -1 && (float(maxArea) / (float)min(r1.area(), currentFrameBlobs.at(maxIdx).currentBoundingRect.area()) >= overlapRatio)) { // merge
			/*std::vector<cv::Point> _contour = currentFrameBlobs.at(maxIdx).currentContour, hull;
			for (int i = 0; i < prblob.currentContour.size(); i++)
			_contour.push_back(prblob.currentContour.at(i));	/// insert a point
			convexHull(_contour, hull);
			itms::Blob _blob(hull);
			currentFrameBlobs.at(maxIdx).currentContour.clear();
			currentFrameBlobs.at(maxIdx).currentContour = _blob.currentContour;
			currentFrameBlobs.at(maxIdx).currentBoundingRect = _blob.currentBoundingRect;
			currentFrameBlobs.at(maxIdx).centerPositions.at(currentFrameBlobs.at(maxIdx).centerPositions.size() - 1) = _blob.centerPositions.back();*/
			// car categorization is not yet included 
			  continue;
		  }
		  else { // add new
			  predBlobInx.push_back(prblob); // sangkny needs to check if the existing blob (prblob) is still tracked or not
											 //currentFrameBlobs.push_back(prblob);
		  }

	  }

	  for (int i = 0; i<predBlobInx.size(); i++) // need to check if the existing blob has been tracked or not and its categori
		  currentFrameBlobs.push_back(predBlobs.at(predBlobInx.at(i)));

  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  // 2019. 01. 04 -> Apply fastDSST when the existing blob was not matched by current blob
  void matchCurrentFrameBlobsToExistingBlobs(itms::Config& _conf, cv::Mat& preImg, const cv::Mat& srcImg, std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, int &id) {
	  // lost object tracker
	  //size_t N = existingBlobs.size();
	  //std::vector<int> assignment(N, -1); // if the blob is matched then it will has 1 value. However, we can make the value the matched index in current Frame
	  // sangkny 2019. 02. 11. NMS implementation for overlapped Blob
	  // 0. collect information from blobs
	  // 1. non maximum supression for Blobs
	  if (existingBlobs.size() > 1) {		// object # > 1
		  const float score_threshold = 0.5;	// min threshold 
		  const float nms_threshold = 0.02;		// 2 % overlap 하나의 object가 너무 작으면 문제가 생긴다.
		  
		  std::vector<float> confidences;
		  std::vector<cv::Rect> boxes;
		  std::vector<int>indices; 
		  // getting the information from blobs
		  for (size_t i = 0; i < existingBlobs.size(); i++) {
			  float value = 0;
			  // rect boxes
			  boxes.push_back(existingBlobs.at(i).currentBoundingRect);
			  if (existingBlobs.at(i).blnStillBeingTracked)
				  value += 0.5;
			  else
				  value += 0.3;

			  // scores
			  // age
			  value += (fmin(0.2, ((float)(existingBlobs.at(i).centerPositions.size()*10) / (float)_conf.max_Center_Pts))); // max 0.2
			  //value += (fmin(0.2, ((float)existingBlobs.at(i).totalVisibleCount / (float)existingBlobs.at(i).age))); // max 0.2
			  // area
			  if (existingBlobs.at(i).currentBoundingRect.area() > 1200*_conf.scaleFactor)  // max 0.3
				  value += 0.3;
			  else
				  value += 0.1;

			  confidences.push_back(value);

		  }
		  cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold, indices);
		  
		  // remove a blob if necessary
		  if (boxes.size() != indices.size()) {
			  if (_conf.debugGeneralDetail)
				  std::cout << " NMS, total: " << (boxes.size() - indices.size()) << " blob will be erased (!)(!)" << endl;

			  std::vector<Blob>::iterator exBlob = existingBlobs.begin();
			  size_t i = 0;
			  while (exBlob != existingBlobs.end()) {
				  bool foundIdx = false;
				  for (size_t idx = 0; idx < indices.size(); idx++) {
					  if (i == indices[idx]) {
						  foundIdx = true;
						  break;
					  }
				  }
				  if (foundIdx) {
					  ++exBlob;
				  }
				  else { // erase 
					  exBlob = existingBlobs.erase(exBlob);
					  if (_conf.debugGeneralDetail) {
						  cout << "Blob #: " << i << "was erased form the existingBlobs !!" << endl;
					  }
				  }
				  i++;  // keep increase the original order index
			  }
		  }
	  } // end of if (existingBlobs.size() > 1) {		// object # > 1

	  // blob iterator
	  std::vector<Blob>::iterator existBlob = existingBlobs.begin();
	  while (existBlob != existingBlobs.end()) {
		  // check if a block is too old after disappeared in the screen
		  //cv::dnn::NMSBoxes()
		  if (existBlob->blnStillBeingTracked == false
			  && existBlob->intNumOfConsecutiveFramesWithoutAMatch >= _conf.maxNumOfConsecutiveInvisibleCounts) {
			  // removing a blob from the list of existingBlobs	when it has been deserted long time or overlapped with the current active object					
			  // overlapping test if overlapped or background, we will erase.
			  std::vector<pair<int, int>> overlappedBobPair; // first:blob index, second:overlapped type
			  int intIndexOfOverlapped = -1;
			  double dblLeastDistance = 100000.0;
			  for (unsigned int i = 0; i < existingBlobs.size(); i++) {
				  int intSect = InterSectionRect(existBlob->currentBoundingRect, existingBlobs[i].currentBoundingRect);

				  if (intSect >= 1 && existingBlobs[i].blnStillBeingTracked) { // deserted blob will be eliminated, 1, 2, 3 indicate the inclusion of one block in another
					  overlappedBobPair.push_back(std::pair<int, int>(i, intSect));
				  }

			  }
			  // erase the blob
			  if (overlappedBobPair.size() > 0) { // how about itself ? 
				  if (_conf.debugTrace && _conf.debugGeneralDetail) {
					  cout << " (!)==> Old and deserted blob id: " << existBlob->id << " is eliminated at " << existBlob->centerPositions.back() << Size(existBlob->currentBoundingRect.width, existBlob->currentBoundingRect.height) << "\n(blobs Capacity: " << existingBlobs.capacity() << ")" << endl;
					  cout << "cpt size: " << existBlob->centerPositions.size() << endl;
					  cout << "age: " << existBlob->age << endl;
					  cout << "totalVisible #: " << existBlob->totalVisibleCount << endl;
				  }
				  existBlob = existingBlobs.erase(existBlob);
			  }
			  else { // partial(0) or no overlapped (-1)
					 // conditional elimination
				  if (_conf.debugTrace  && _conf.debugGeneralDetail) {
					  cout << " (!)! Old blob id: " << existBlob->id << " is conditionally eliminated at " << existBlob->centerPositions.back() << Size(existBlob->currentBoundingRect.width, existBlob->currentBoundingRect.height) << "\n(blobs Capacity: " << existingBlobs.capacity() << ")" << endl;
					  cout << "cpt size: " << existBlob->centerPositions.size() << endl;
					  cout << "age: " << existBlob->age << endl;
					  cout << "totalVisible #: " << existBlob->totalVisibleCount << endl;
				  }
				  if (existBlob->centerPositions.size() > _conf.maxNumOfConsecutiveInvisibleCounts / 2) { // current Blob was moving but now stopped.          
																										  // this blob needs to declare stopped object and erase : 전에 움직임이 있다가 현재는 없는 object임.
																										  // kind of trick !! which will remove the objects nodetected for long time
					  if (existBlob->centerPositions.size() > _conf.maxNumOfConsecutiveInvisibleCounts) {
						  if (_conf.debugTrace && _conf.debugGeneralDetail)
							  cout << "\n\n\n\n -------It--was moving--and --> stopped : object eliminated \n\n\n\n";
						  existBlob = existingBlobs.erase(existBlob);
						  continue;
						  //waitKey(0);
					  }
					  // -------------------------------------------------------------
					  // check this out : sangkny on 2018/12/17
					  existBlob->blnCurrentMatchFoundOrNewBlob = false;
					  if (!_conf.m_externalTrackerForLost) { // sangkny 2019/01/27 // 어짜피 아래 m_externalTrackerForLost 옵션이 있으면 다시 함.. 
						  existBlob->centerPositions.push_back(existBlob->centerPositions.back()); // this line required for isolated but Stopped
						  if (existBlob->centerPositions.size() > _conf.max_Center_Pts)						// sangkny 2019. 01. 18 for stable speed processing
							  pop_front(existBlob->centerPositions, existBlob->centerPositions.size() - _conf.max_Center_Pts);
						  existBlob->predictNextPosition(); // can be removed if the above line is commented
															////existingBlob->os = getObjectStatusFromBlobCenters(*existingBlob, ldirection, movingThresholdInPixels, minVisibleCount); // 벡터로 넣을지 생각해 볼 것, update로 이전 2018. 10.25
					  }
					  ++existBlob;
					  continue; // can be removed this
								// ---------------------------------------------------------------

				  }
				  else {
					  existBlob = existingBlobs.erase(existBlob);
				  }
			  }
		  }
		  else {
			  existBlob->blnCurrentMatchFoundOrNewBlob = false;
			  existBlob->predictNextPosition();
			  //existingBlob->os = getObjectStatusFromBlobCenters(*existingBlob, ldirection, movingThresholdInPixels, minVisibleCount); 
			  // 벡터로 넣을지 생각해 볼 것, update로 이전 2018. 10.25
			  ++existBlob;
		  }
	  } // end while ( existingBlob != existingBlobs.end())
		/*for (auto &existingBlob : existingBlobs) {
		existingBlob.blnCurrentMatchFoundOrNewBlob = false;
		existingBlob.predictNextPosition();
		}*/

		// candidate search only with distances between centers of currentFrameBlobs and existing blobs.
		// add more property including area and h/w ratio
		// serch around the nearest neighbor blob for tracking 
		// for searching larger area with more accuracy, we need to increase the search range (CurrentDiagonalSize) or to particle filter
		// with data, kalman or other tracking will be more accurate
	  for (auto &currentFrameBlob : currentFrameBlobs) {
		  int intIndexOfLeastDistance = 0;
		  //int intIndexOfHighestScore = 0;
		  double dblLeastDistance = 100000.0;
		  /*
		  double totalScore = 100.0, cutTotalScore = 20.0, maxTotalScore = 0.0;

		  float allowedPercentage = 0.25; // 20%
		  float minArea = currentFrameBlob.currentBoundingRect.width *(1.0f - allowedPercentage); // 편차가 너무 크므로 
		  float MaxArea = currentFrameBlob.currentBoundingRect.width *(1.0f + allowedPercentage); // width 와 height로 구성
		  float minDiagonal = currentFrameBlob.currentBoundingRect.height * (1.0f - allowedPercentage); // huMoment를 이용하는 방법 모색
		  float maxDiagonal = currentFrameBlob.currentBoundingRect.height * (1.0f + allowedPercentage);
		  */

		  for (unsigned int i = 0; i < existingBlobs.size(); i++) {
			  if (existingBlobs[i].blnStillBeingTracked == true) { // find assigned tracks
			  // it can be replaced with the tracking algorithm or assignment algorithm like KALMAN or Hungrian Assignment algorithm 
				  double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);
				  /* // sangkny 2019. 02. 15
				  totalScore -= dblDistance;
				  totalScore -= (existingBlobs[i].currentBoundingRect.height < minDiagonal || existingBlobs[i].currentBoundingRect.height>maxDiagonal) ? 10 : 0;
				  totalScore -= (existingBlobs[i].currentBoundingRect.width < minArea || existingBlobs[i].currentBoundingRect.width > MaxArea) ? 10 : 0;
				  totalScore -= (abs(existingBlobs[i].currentBoundingRect.area() - currentFrameBlob.currentBoundingRect.area()) / max(existingBlobs[i].currentBoundingRect.width, currentFrameBlob.currentBoundingRect.width));
				  */
				  if (existingBlobs[i].oc != currentFrameBlob.oc)
					  int kkk = 0;

				  if (dblDistance < dblLeastDistance && (existingBlobs[i].oc == currentFrameBlob.oc || existingBlobs[i].oc == OC_OTHER || currentFrameBlob.oc == OC_OTHER)) {
					  dblLeastDistance = dblDistance;
					  intIndexOfLeastDistance = i;
				  }
				  /*if (maxTotalScore < totalScore) {
				  maxTotalScore = totalScore;
				  intIndexOfHighestScore = i;
				  dblLeastDistance = dblDistance;
				  }*/
			  }
			  else { // existingBlobs[i].bInStillBeingTracked == false;
					 /* do something for unassinged tracks */
				  int temp = 0; // no meaning 
			  }
		  }
		  if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize) { // 충분히 클수록 좋다.
			  addBlobToExistingBlobs(_conf, currentFrameBlob, existingBlobs, intIndexOfLeastDistance);

			  //  if(maxTotalScore>=cutTotalScore && dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5){
			  //	addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfHighestScore);

			  // lost object detection 
			  //				assignment.at(intIndexOfLeastDistance) = 1;
		  }
		  else { // this routine contains new and unassigned track(blob)s
				 // add new blob
			  vector<Point2f> blobCenterPxs;
			  blobCenterPxs.push_back(currentFrameBlob.centerPositions.back());
			  float distance = getDistanceInMeterFromPixels(blobCenterPxs, _conf.transmtxH, _conf.lane_length, false);
			  if (0 && _conf.debugGeneralDetail)
				  cout << " distance: " << distance / 100 << " meters from the starting point.\n";
			  // do the inside
			  if (distance >= 100.00/* 1m */ && distance < 19900/*199m*/) {// between 1 meter and 199 meters
				  ObjectClass objclass;
				  float classProb = 0.f;
				  classifyObjectWithDistanceRatio(_conf, currentFrameBlob, distance / 100, objclass, classProb);
				  // update the blob info and add to the existing blobs according to the classifyObjectWithDistanceRatio function output
				  // verify the object with cascade object detection

				  if (classProb>0.5f) {
					  // check with a ML-based approach
					  float scaleRect = 1.5;										// put it to the config parameters
					  cv::Rect roi_rect(currentFrameBlob.currentBoundingRect);	// copy the current Rect and expand it
					  cv::Rect expRect = expandRect(roi_rect, scaleRect*roi_rect.width, scaleRect*roi_rect.height, srcImg.cols, srcImg.rows);
					  if (currentFrameBlob.oc == itms::ObjectClass::OC_VEHICLE) {
						  // verify it
						  std::vector<cv::Rect> cars;
						  detectCascadeRoiVehicle(_conf, srcImg, expRect, cars);
						  if (cars.size())
							  currentFrameBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
						  else if (_conf.img_dif_th >= _conf.nightBrightness_Th &&/* at night */ classProb >= _conf.nightObjectProb_Th)
							  ;	// just put this candidate
						  else //if(/* at night and */classProb < 0.8)
							  continue;
					  }
					  else if (currentFrameBlob.oc == itms::ObjectClass::OC_HUMAN) {
						  // verify it
						  std::vector<cv::Rect> people;
						  detectCascadeRoiHuman(_conf, srcImg, expRect, people);
						  if (people.size())
							  currentFrameBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
						  else if (_conf.img_dif_th >= _conf.nightBrightness_Th &&/* at night */ classProb >= _conf.nightObjectProb_Th) // at daytime, need to classify using cascade classifier, otherwise, at night, we put the candidate
							  ;		// put this contour as it is 
						  else
							  continue; // just skip the contour
					  }
					  else {// should not com in this loop (OC_OTHER)
						  ;
					  }
					  addNewBlob(currentFrameBlob, existingBlobs, id);
				  }
			  }

		  }

	  }

	  // update tracks 
	  // 2018. 10. 25 getObjectStatusFromBlobCenters 을 전반부에서 이동.. 그리고, 각종 object status object classification을 여기서 함..
	  // 2019. 01. 04 for unassigned tracks, perform the fast DSST to find out the lost object if any and check if therer exists missing objects.
	  size_t blobIdx = 0;
	  auto Clamp = [](int& v, int& size, int hi) -> bool
	  {
		  if (size < 2)
		  {
			  size = 2;
		  }
		  if (v < 0)
		  {
			  v = 0;
			  return true;
		  }
		  else if (v + size > hi - 1)
		  {
			  v = hi - 1 - size;
			  return true;
		  }
		  return false;
	  };

	  for (auto &existingBlob : existingBlobs) { // update track routine

		  if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) { // unassigned tracks            
			  existingBlob.age++;

			  if (_conf.m_externalTrackerForLost /* && (assignment[blobIdx] == -1)*/) { // 상관이 없네...
				  existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;// temporal line
																		// reinitialize the fastDSST with prevFrame and update the fastDSST with current Frame, finally check its robustness with template matching or other method
				  cv::Rect newRoi, m_predictionRect;
				  int expandY = 5;
				  float heightRatio = (float)(existingBlob.currentBoundingRect.height + expandY) / (existingBlob.currentBoundingRect.height);
				  int expandX = max(0, cvRound((float)(existingBlob.currentBoundingRect.width)*heightRatio - existingBlob.currentBoundingRect.width));
				  cv::Rect expRect = expandRect(existingBlob.currentBoundingRect, expandX, expandY, preImg.cols, preImg.rows);

				  if (0 && _conf.debugGeneral && _conf.debugGeneralDetail)
					  cout << "From boundingRect: " << existingBlob.currentBoundingRect << " => To expectedRect: " << expRect << endl;

				  if (!_conf.isSubImgTracking) { // Global/full image-based approach
					  if (!existingBlob.m_tracker_psr || existingBlob.m_tracker_psr.empty())
						  existingBlob.CreateExternalTracker();	// create ExternalTracker
					  bool success = false;
					  cv::Mat preImg3, srcImg3;
					  if (preImg.channels() < 3) {
						  cv::cvtColor(preImg, preImg3, CV_GRAY2BGR);
						  cv::cvtColor(srcImg, srcImg3, CV_GRAY2BGR);
					  }
					  newRoi = expRect;
					  if (!existingBlob.m_tracker_initialized) {		// do it only once
																		/*existingBlob.m_tracker->init(expRect, preImg);
																		existingBlob.m_tracker_initialized = true;*/
						  success = (preImg.channels() < 3) ? existingBlob.m_tracker_psr->reinit(preImg3, newRoi) : existingBlob.m_tracker_psr->reinit(preImg, newRoi);

						  existingBlob.m_tracker_initialized = true;
					  }
					  else
						  success = (preImg.channels() < 3) ? existingBlob.m_tracker_psr->updateAt(srcImg3, newRoi) : existingBlob.m_tracker_psr->updateAt(srcImg, newRoi);
					  //newRoi = existingBlob.m_tracker->update(srcImg); // do update for full image-based fast dsst

					  if (!success) {
						  newRoi = expRect;

						  success = (preImg.channels() < 3) ? existingBlob.m_tracker_psr->reinit(preImg3, newRoi) : existingBlob.m_tracker_psr->reinit(preImg, newRoi);
						  success = (preImg.channels() < 3) ? existingBlob.m_tracker_psr->update(srcImg3, newRoi) : existingBlob.m_tracker_psr->update(srcImg, newRoi);
					  }
					  // as of 2019. 01. 18, just put the new center points except for countour information	
					  if (success) {						  
						  // move a boundary and its boundingRect if necessary

						  if (existingBlob.resetBlobContourWithCenter(cv::Point(cvRound(newRoi.x + newRoi.width / 2.f), cvRound(newRoi.y + newRoi.height / 2.f)))) { // check if the center points will move or not
							  cv::Rect tmpRect = existingBlob.currentBoundingRect;
							  existingBlob.currentBoundingRect.x -= (tmpRect.x + tmpRect.width / 2.f - (newRoi.x + newRoi.width / 2.f));  // move to the newRoi center with keep the size of Boundary
							  Clamp(existingBlob.currentBoundingRect.x, existingBlob.currentBoundingRect.width, srcImg.cols);
							  existingBlob.currentBoundingRect.y -= (tmpRect.y + tmpRect.height / 2.f - (newRoi.y + newRoi.height / 2.f));
							  Clamp(existingBlob.currentBoundingRect.y, existingBlob.currentBoundingRect.height, srcImg.rows);
						  }
						  existingBlob.centerPositions.push_back(cv::Point(cvRound(newRoi.x + newRoi.width / 2.f), cvRound(newRoi.y + newRoi.height / 2.f))); // put a center anyway
					  }
					  else {
						  existingBlob.centerPositions.push_back(existingBlob.centerPositions.back()); // add one point
					  }
					  

					  if (existingBlob.centerPositions.size() > _conf.max_Center_Pts)
						  pop_front(existingBlob.centerPositions, (existingBlob.centerPositions.size() - _conf.max_Center_Pts));

					  existingBlob.predictNextPosition();

					  if (1 && _conf.debugShowImagesDetail) {
						  cv::Mat tmp2 = srcImg.clone();
						  if (tmp2.channels() < 3)
							  cvtColor(tmp2, tmp2, CV_GRAY2BGR);
						  cv::rectangle(tmp2, expRect, SCALAR_CYAN, 1);
						  cv::rectangle(tmp2, newRoi, SCALAR_MAGENTA, 2);
						  cv::imshow("full image tracking", tmp2);
						  cv::waitKey(1);
					  }
					  if (0 && _conf.debugShowImagesDetail) { // now only for full image coordinates
						  cv::Mat debugImage = srcImg.clone(), difImg;
						  absdiff(preImg, srcImg, difImg);
						  threshold(difImg, difImg, _conf.img_dif_th, 255, CV_THRESH_BINARY);
						  if (debugImage.channels() < 3)
							  cvtColor(debugImage, debugImage, CV_GRAY2BGR);
						  cv::rectangle(debugImage, existingBlob.currentBoundingRect, SCALAR_CYAN, 1);
						  cv::rectangle(debugImage, newRoi, SCALAR_MAGENTA, 2);
						  cv::imshow("Lost object tracking", debugImage);
						  cv::imshow("|pre - cur| frame", difImg);
						  cv::waitKey(1);
					  }
				  }
				  else { // sub image-based approach for lost object detection, in this case, init and update need to be carried out at ontime 
					  m_predictionRect = expRect;//existingBlob.currentBoundingRect;//expRect;			
												 // partial local tracking using FDSST 2019. 01. 17
					  cv::Size roiSize(max(2 * m_predictionRect.width, srcImg.cols / 8), std::max(2 * m_predictionRect.height, srcImg.rows / 8)); // small subImage selection, I need to check if we can reduce more
					  bool success = false;
					  if (roiSize.width > srcImg.cols)
					  {
						  roiSize.width = srcImg.cols;
					  }
					  if (roiSize.height > srcImg.rows)
					  {
						  roiSize.height = srcImg.rows;
					  }
					  cv::Point roiTL(m_predictionRect.x + m_predictionRect.width / 2 - roiSize.width / 2, m_predictionRect.y + m_predictionRect.height / 2 - roiSize.height / 2);
					  cv::Rect roiRect(roiTL, roiSize);			// absolute full image coordinates
					  Clamp(roiRect.x, roiRect.width, srcImg.cols);
					  Clamp(roiRect.y, roiRect.height, srcImg.rows);

					  cv::Rect2d lastRect(m_predictionRect.x - roiRect.x, m_predictionRect.y - roiRect.y, m_predictionRect.width, m_predictionRect.height);
					  // relative subImage coordinates
					  if (!existingBlob.m_tracker_psr || existingBlob.m_tracker_psr.empty()) {
						  existingBlob.CreateExternalTracker();
					  }
				  cv:Rect2d newsubRoi = lastRect;  // local window rect
					  if (lastRect.x >= 0 &&
						  lastRect.y >= 0 &&
						  lastRect.x + lastRect.width < roiRect.width &&
						  lastRect.y + lastRect.height < roiRect.height &&
						  lastRect.area() > 0) {
						  cv::Mat preImg3, srcImg3;
						  if (preImg.channels() < 3) {
							  cv::cvtColor(preImg, preImg3, CV_GRAY2BGR);
							  cv::cvtColor(srcImg, srcImg3, CV_GRAY2BGR);
						  }

						  if (!existingBlob.m_tracker_initialized) {
							  success = (preImg.channels() < 3) ?
								  existingBlob.m_tracker_psr->reinit(cv::Mat(preImg3, roiRect), newsubRoi) :
								  existingBlob.m_tracker_psr->reinit(cv::Mat(preImg, roiRect), newsubRoi);
							  existingBlob.m_tracker_initialized = true; // ??????????????						
						  }
						  else {
							  success = (preImg.channels() < 3) ?
								  existingBlob.m_tracker_psr->updateAt(cv::Mat(srcImg3, roiRect), newsubRoi) :
								  existingBlob.m_tracker_psr->updateAt(cv::Mat(srcImg, roiRect), newsubRoi);
						  }

						  // update position and update the current frame because we not do this sometime if necessary						
						  if (!success) { // lastRect is not new, and old one is used								
							  newsubRoi = lastRect;
							  success = (preImg.channels() < 3) ?
								  existingBlob.m_tracker_psr->reinit(cv::Mat(preImg3, roiRect), newsubRoi) :
								  existingBlob.m_tracker_psr->reinit(cv::Mat(preImg, roiRect), newsubRoi);

							  success = (preImg.channels() < 3) ?
								  existingBlob.m_tracker_psr->update(cv::Mat(srcImg3, roiRect), newsubRoi) :
								  existingBlob.m_tracker_psr->update(cv::Mat(srcImg, roiRect), newsubRoi);
						  } // else is not required because if updateAt is successful, the new location after update is same							

						  cv::Rect prect;
						  if (success)
							  prect = cv::Rect(cvRound(newsubRoi.x) + roiRect.x, cvRound(newsubRoi.y) + roiRect.y, cvRound(newsubRoi.width), cvRound(newsubRoi.height)); // new global location 
						  else
							  prect = cv::Rect(cvRound(lastRect.x) + roiRect.x, cvRound(lastRect.y) + roiRect.y, cvRound(lastRect.width), cvRound(lastRect.height)); // new global location using old one

						// sangkny update the center points of the existing blob which has been untracted						  
						  if (existingBlob.resetBlobContourWithCenter(cv::Point(cvRound(prect.x + prect.width / 2.f), cvRound(prect.y + prect.height / 2.f)))){ // boundary movement first 
							  cv::Rect tmpRect = existingBlob.currentBoundingRect;
							  existingBlob.currentBoundingRect.x -= (tmpRect.x + tmpRect.width / 2.f - (prect.x + prect.width / 2.f));  // move to the newRoi center with keep the size of Boundary
							  Clamp(existingBlob.currentBoundingRect.x, existingBlob.currentBoundingRect.width, srcImg.cols);
							  existingBlob.currentBoundingRect.y -= (tmpRect.y + tmpRect.height / 2.f - (prect.y + prect.height / 2.f));
							  Clamp(existingBlob.currentBoundingRect.y, existingBlob.currentBoundingRect.height, srcImg.rows);
						  }
						  existingBlob.centerPositions.push_back(cv::Point(cvRound(prect.x + prect.width / 2.f), cvRound(prect.y + prect.height / 2.f)));

						  if (existingBlob.centerPositions.size() >_conf.max_Center_Pts)
							  pop_front(existingBlob.centerPositions, (existingBlob.centerPositions.size() - _conf.max_Center_Pts));

						  existingBlob.predictNextPosition();

						  if (1 && _conf.debugShowImagesDetail) { // local image debug
							  cv::Mat tmp2 = cv::Mat(srcImg, roiRect).clone();
							  if (tmp2.channels() < 3)
								  cvtColor(tmp2, tmp2, CV_GRAY2BGR);
							  cv::rectangle(tmp2, lastRect, SCALAR_CYAN, 1);
							  cv::rectangle(tmp2, newsubRoi, SCALAR_MAGENTA, 2);
							  if (!success) {
								  cv::Point_<double> tl = newsubRoi.tl();
								  cv::Point_<double> br = newsubRoi.br();

								  line(tmp2, tl, br, Scalar(0, 0, 255));
								  line(tmp2, cv::Point_<double>(tl.x, br.y),
									  cv::Point_<double>(br.x, tl.y), Scalar(0, 0, 255));
							  }
							  cv::imshow("track", tmp2);
							  cv::waitKey(1);
						  }
						  if (1 && _conf.debugShowImagesDetail) { // full image debug
							  cv::Mat tmp2 = srcImg.clone();
							  if (tmp2.channels() < 3)
								  cvtColor(tmp2, tmp2, CV_GRAY2BGR);
							  cv::rectangle(tmp2, expRect, SCALAR_CYAN, 1);
							  cv::rectangle(tmp2, prect, SCALAR_MAGENTA, 2);
							  cv::imshow("full image location", tmp2);
							  cv::waitKey(1);
						  }
					  }
					  else {
						  if (existingBlob.m_tracker_psr || !existingBlob.m_tracker_psr.empty()) {
							  existingBlob.m_tracker_psr.release();
							  existingBlob.m_tracker_initialized = false;
						  }
					  }

				  }

			  }
			  else {
				  existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
				  // sangkny 2019. 01. 18 put the last center points not the prediction(to reflect the object status)
				  // because the prediction will affect the next frame processing.
				  // however, the countour and its siblings stay as it is for next frame.
				  // 
				  existingBlob.centerPositions.push_back(existingBlob.centerPositions.back());
				  if (existingBlob.centerPositions.size() > _conf.max_Center_Pts)						// sangkny 2019. 01. 18 for stable speed processing
					  pop_front(existingBlob.centerPositions, existingBlob.centerPositions.size() - _conf.max_Center_Pts);

				  existingBlob.predictNextPosition();
			  }

		  }
		  else { // update the assigned (matched) tracks
			  existingBlob.intNumOfConsecutiveFramesWithoutAMatch = 0; // reset because of appearance
			  existingBlob.age++;
			  existingBlob.totalVisibleCount++;
		  }

		  if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= _conf.maxNumOfConsecutiveInFramesWithoutAMatch/* 1sec. it should be a predefined threshold */) {
			  existingBlob.blnStillBeingTracked = false; /* still in the list of blobs */
		  }
		  // object status, class update routine starts        
		  //existingBlob.os = getObjectStatusFromBlobCenters(existingBlob, _conf.ldirection, _conf.movingThresholdInPixels, _conf.minVisibleCount);
		  existingBlob.os = getObjStatusUsingLinearRegression(_conf, existingBlob, _conf.ldirection, _conf.movingThresholdInPixels, _conf.minVisibleCount);
		  // 벡터로 넣을지 생각해 볼 것, 그리고, regression from kalman 으로 부터 정지 등을 판단하는 것도 고려 중....		  

		  // object status, class update routine ends
		  blobIdx++;
	  }

  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  void addBlobToExistingBlobs(itms::Config& _conf, Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {
	  // update the status or correcte the status
	  // here, we need to control the gradual change of object except the center point	
	  
	  float allowedPercentage = 1.0; // 20%
	  float minArea = existingBlobs[intIndex].currentBoundingRect.area() *(1.0f - allowedPercentage);
	  float MaxArea = existingBlobs[intIndex].currentBoundingRect.area() *(1.0f + allowedPercentage);
	  float minDiagonal = existingBlobs[intIndex].dblCurrentDiagonalSize * (1.0f - allowedPercentage);
	  float maxDiagonal = existingBlobs[intIndex].dblCurrentDiagonalSize * (1.0f + allowedPercentage);
	  
	  if (0 && (currentFrameBlob.currentBoundingRect.area() < minArea ||
		  currentFrameBlob.currentBoundingRect.area() > MaxArea ||
		  currentFrameBlob.dblCurrentDiagonalSize < minDiagonal ||
		  currentFrameBlob.dblCurrentDiagonalSize > maxDiagonal)) {

		  // if the given current frame blob's size is not propriate, we just move the center point of the existing blob
		  // 2018. 10. 27
		  // change the center point only to currentFrameBlob
		  std::vector<cv::Point> newContourPts = existingBlobs[intIndex].currentContour;
		  cv::Point cFBlobctrPt = currentFrameBlob.centerPositions.back();
		  cv::Point extBlobctrPt = existingBlobs[intIndex].centerPositions.back();
		  for (int i = 0; i<newContourPts.size(); i++)
			  newContourPts[i] += (cFBlobctrPt - extBlobctrPt); // center point movement to existing Blob

		  existingBlobs[intIndex].currentContour = newContourPts;
		  //existingBlobs[intIndex].currentBoundingRect = cv::boundingRect(newContourPts); // actually it is same as existingBlobs[intIndex]

		  existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());
		  if (existingBlobs[intIndex].centerPositions.size() > _conf.max_Center_Pts)						// sangkny 2019. 01. 18 for stable speed processing
			  pop_front(existingBlobs[intIndex].centerPositions, existingBlobs[intIndex].centerPositions.size() - _conf.max_Center_Pts);

		  //existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize; 
		  //existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;
	  }
	  else {
		  // if the given current frame blob's size is not propriate, we just move the center point of the existing blob
		  existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
		  existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

		  existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());
		  if (existingBlobs[intIndex].centerPositions.size() > _conf.max_Center_Pts)						// sangkny 2019. 01. 18 for stable speed processing
			  pop_front(existingBlobs[intIndex].centerPositions, existingBlobs[intIndex].centerPositions.size() - _conf.max_Center_Pts);


		  existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
		  existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;
		  // sangkny 2018. 12. 19, updated 2019. 01. 02
		  // check this out one more time 
		  if (existingBlobs[intIndex].oc != currentFrameBlob.oc) { // object classification determination
			  if (currentFrameBlob.oc == ObjectClass::OC_OTHER) {
				  if (existingBlobs[intIndex].oc_prob < 1.f)
					  existingBlobs[intIndex].oc = currentFrameBlob.oc; // need to do classification
			  }
			  else if (existingBlobs[intIndex].oc == ObjectClass::OC_OTHER) {
				  existingBlobs[intIndex].oc = currentFrameBlob.oc;	// need to do classification according to the class of currentFrameBlob
				  existingBlobs[intIndex].oc_prob = currentFrameBlob.oc_prob;
			  }
			  else { // they have different its own classes
				  std::cout << " algorithm can not be here !!!\n";
				  // they can not be here because this case should have been refined in the above step MatchCurrentBlobsToExistingBlbos
			  }
		  }
		  else { // they have the same class
			  existingBlobs[intIndex].oc = currentFrameBlob.oc; // do nothing because they are same
			  existingBlobs[intIndex].oc_prob = (existingBlobs[intIndex].oc_prob > currentFrameBlob.oc_prob) ? existingBlobs[intIndex].oc_prob : currentFrameBlob.oc_prob;
		  }
		  // sangkny 2018. 12. 31
		  existingBlobs[intIndex].m_points = currentFrameBlob.m_points;

	  }

	  //if (existingBlobs[intIndex].totalVisibleCount >= 8 /* it should be a predefined threshold */)
	  existingBlobs[intIndex].blnStillBeingTracked = true; /* it is easy to be exposed to noise, so it put constraints to this */
														   // else
														   //   int kkk = 0;
	  existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
	  // update the object class
	  if (existingBlobs[intIndex].oc_prob < 0.5 || existingBlobs[intIndex].oc == ObjectClass::OC_OTHER) {
		  // srcBlob-based classfication 2018. 12. 10
		  vector<Point2f> blobCenterPxs;
		  blobCenterPxs.push_back(currentFrameBlob.centerPositions.back());
		  float distance = getDistanceInMeterFromPixels(blobCenterPxs, _conf.transmtxH, _conf.lane_length, false);
		  if (_conf.debugGeneral && _conf.debugGeneralDetail)
			  cout << " distance: " << distance / 100 << " meters from the starting point.\n";

		  ObjectClass objclass;
		  float classProb = 0.f;
		  classifyObjectWithDistanceRatio(_conf, currentFrameBlob, distance / 100, objclass, classProb);
		  // update the blob info 
		  if (existingBlobs[intIndex].oc_prob < currentFrameBlob.oc_prob) {
			  existingBlobs[intIndex].oc_prob = currentFrameBlob.oc_prob;
			  existingBlobs[intIndex].oc = currentFrameBlob.oc;
		  }

	  }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &id) {

	  if (currentFrameBlob.totalVisibleCount > 1)
		  int temp = 0;
	  // check the size according to the distance from the starting point because NCC is already performed.

	  currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;	  
	  currentFrameBlob.id = id;
	  assert(currentFrameBlob.startPoint == currentFrameBlob.centerPositions.back()); // always true
	  existingBlobs.push_back(currentFrameBlob);
	  id = (id++ % 2048); // reset id according to the max number of type (int) or time (day or week)
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  // it should inspect after predicting the next position
  // it determines the status away at most 5 frame distance from past locations
  // it shoud comes from the average but for efficiency, does come from the several past location. 
  ObjectStatus getObjectStatusFromBlobCenters(const Config& config, Blob &blob, const LaneDirection &lanedirection, int movingThresholdInPixels, int minTotalVisibleCount) {
	  ObjectStatus objectstatus = ObjectStatus::OS_NOTDETERMINED;
	  if (blob.totalVisibleCount < minTotalVisibleCount) // !! parameter
		  return objectstatus;

	  int numPositions = (int)blob.centerPositions.size();
	  //int maxNumPosition = 5;
	  int bweightedAvg = -1; // -1: info from the past far away, 0: false (uniform average), 1: true (weighted average)
	  Blob tmpBlob = blob;
	  // it will affect the speed because of const Blob declaration in parameters !!!!
	  int deltaX;// = blob.predictedNextPosition.x - blob.centerPositions.back().x;
	  int deltaY;// = blob.predictedNextPosition.y - blob.centerPositions.back().y; // have to use moving average after applying media filtering    
	  cv::Point wpa = tmpBlob.weightedPositionAverage(bweightedAvg);
	  deltaX = blob.predictedNextPosition.x - wpa.x;
	  deltaY = blob.predictedNextPosition.y - wpa.y;

	  switch (lanedirection) {	  
	  case LD_NORTH:
	  case LD_SOUTH:
		  if (abs(deltaY) <= movingThresholdInPixels)
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_SOUTH) ? (deltaY > 0 ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : (deltaY > 0 ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  case LD_EAST:
	  case LD_WEST:
		  if (abs(deltaX) <= movingThresholdInPixels) // 
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_EAST) ? (deltaX > 0 ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : (deltaX > 0 ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  case LD_NORTHEAST:
	  case LD_SOUTHWEST:
		  if (abs(deltaX) + abs(deltaY) <= movingThresholdInPixels) // 
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_NORTHEAST) ? ((deltaX > 0 || deltaY < 0) ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : ((deltaX > 0 || deltaY <0) ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  case LD_SOUTHEAST:
	  case LD_NORTHWEST:
		  if (abs(deltaX) + abs(deltaY) <= movingThresholdInPixels) // 
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_SOUTHEAST) ? ((deltaX > 0 || deltaY > 0) ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : ((deltaX > 0 || deltaY >0) ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  default:
		  objectstatus = OS_NOTDETERMINED;
		  break;
	  }

	  // update object state and return the current estimated state (redundant because we already update the os status and update again later outside the function
	  // 2018. 10. 25
	  // 0. current os status will be previous status after update
	  // weight policy : current status consecutive counter, others, 1
	  // 2018. 10. 26 -> replaced with below functions

	  //itms::Blob orgBlob = blob; // backup 
	  updateBlobProperties(config, blob, objectstatus, 0.0); // update the blob properties including the os_prob
												//itms::ObjectStatus tmpOS = computeObjectStatusProbability(blob); // get the moving status according to the probability
												//if (tmpOS != objectstatus) {    
												//  objectstatus = tmpOS;
												//  // go back to original blob and update again correctly
												//  blob = orgBlob;
												//  updateBlobProperties(blob, objectstatus);
												//}


	  return objectstatus;
  }
  /////////////////////////////////////////-- Linear Regression-based Object Directiona and Speed Computation --//////////////////////////////////////////////////////////
  // using LastMinPoints
  ObjectStatus getObjStatusUsingLinearRegression(Config& config, Blob &blob, const LaneDirection &lanedirection, const int movingThresholdInPixels, const int minTotalVisibleCount) {
	  ObjectStatus objectstatus = ObjectStatus::OS_NOTDETERMINED;
	  
	  int lastMinPoints = config.lastMinimumPoints; // minimum Last frame numbers
	  
	  if (blob.totalVisibleCount < minTotalVisibleCount|| ((int)blob.centerPositions.size() < lastMinPoints)) // !! parameter
		  return objectstatus;

	  // get linear regression from  centers
	  track_t kx = 0;
	  track_t bx = 0;
	  track_t ky = 0;
	  track_t by = 0;
	  
	  int trajLen = ((int) blob.centerPositions.size() >= lastMinPoints)? lastMinPoints : (int) blob.centerPositions.size();
	  
	  get_lin_regress_params(blob.centerPositions, blob.centerPositions.size() - trajLen, blob.centerPositions.size(), kx, bx, ky, by);
	  track_t speed = sqrt(sqr(kx * trajLen) + sqr(ky * trajLen));
	  const track_t speedThresh = config.speedLimitForstopping;
	  
	  cv::Point2f sPt = cvtPx2RealPx(static_cast<cv::Point2f>(blob.centerPositions.at(blob.centerPositions.size()-trajLen)), config.transmtxH); // starting pt
	  cv::Point2f ePt = cvtPx2RealPx(static_cast<cv::Point2f>(blob.centerPositions.back()), config.transmtxH);                                  // end pt
	  
	  double dist = distanceBetweenPoints(sPt, ePt); // real distance (cm) in the ROI 
	  float fps = config.fps;
	  double realSpeedKmH = (dist * fps *3600)/(trajLen *100000); // (1 KM = 100000CM, 1 Hour = 3 Sec.)
	  if(0 || config.debugGeneralDetail)
		  cout<< "ID: "<< blob.id<<" , Speed (Km/h): "<< realSpeedKmH << endl;

	  switch (lanedirection) {
	  case LD_NORTH:
	  case LD_SOUTH:
		  if (realSpeedKmH < speedThresh /*abs(deltaY) <= movingThresholdInPixels*/)
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_SOUTH) ? (ky/*deltaY*/ > 0 ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : (ky/*deltaY*/ > 0 ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  case LD_EAST:
	  case LD_WEST:
		  if (realSpeedKmH < speedThresh /*abs(deltaX) <= movingThresholdInPixels*/) // 
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_EAST) ? (kx/*deltaX*/ > 0 ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : (kx/*deltaX*/ > 0 ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  case LD_NORTHEAST:
	  case LD_SOUTHWEST:
		  if (realSpeedKmH < speedThresh /*abs(deltaX) + abs(deltaY) <= movingThresholdInPixels*/) // 
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_NORTHEAST) ? ((kx/*deltaX*/ > 0 || ky/*deltaY*/ < 0) ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : ((kx/*deltaX*/ > 0 || ky/*deltaY*/ <0) ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  case LD_SOUTHEAST:
	  case LD_NORTHWEST:
		  if (realSpeedKmH < speedThresh /*abs(deltaX) + abs(deltaY) <= movingThresholdInPixels*/) // 
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_SOUTHEAST) ? ((kx/*deltaX*/ > 0 || ky/*deltaY*/ > 0) ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : ((kx/*deltaX*/ > 0 || ky/*deltaY*/ >0) ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  default:
		  objectstatus = OS_NOTDETERMINED;
		  break;
	  }

	  // update object state and return the current estimated state (redundant because we already update the os status and update again later outside the function
	  // 2018. 10. 25
	  // 0. current os status will be previous status after update
	  // weight policy : current status consecutive counter, others, 1
	  // 2018. 10. 26 -> replaced with below functions

	 // itms::Blob orgBlob = blob; // backup 
	  updateBlobProperties(config, blob, objectstatus, realSpeedKmH); // update the blob properties including the os_prob

	  return objectstatus;
  }  
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

	  int intX = (point1.x - point2.x);
	  int intY = (point1.y - point2.y);

	  return(sqrt(pow(intX, 2) + pow(intY, 2)));
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName, const cv::Scalar& _color) {
	  cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	  cv::drawContours(image, contours, -1, _color/*SCALAR_WHITE*/, -1);

	  cv::imshow(strImageName, image);
	  cv::waitKey(1);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  void drawAndShowContours(itms::Config& _conf, cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

	  cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	  std::vector<std::vector<cv::Point> > contours, contours_bg;

	  for (auto &blob : blobs) {
		  if (blob.blnStillBeingTracked == true /*&& blob.totalVisibleCount>= minVisibleCount*/) {
			  contours.push_back(blob.currentContour);
		  }
		  else {
			  contours_bg.push_back(blob.currentContour);
		  }
	  }

	  cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);
	  cv::drawContours(image, contours_bg, -1, SCALAR_RED, 2, 8);
	  if (_conf.m_useLocalTracking)
	  {
		  cv::Scalar cl = Scalar(0, 255, 0);// m_colors[track.m_trackID % m_colors.size()];
		  for (auto &blob : blobs) {
			  if (blob.blnStillBeingTracked == true /*&& blob.totalVisibleCount>= minVisibleCount*/)
				  for (auto pt : blob.m_points)
				  {
					  cv::circle(image, cv::Point(cvRound(pt.x), cvRound(pt.y)), 1, cl, -1, CV_AA);
				  }
		  }
	  }

	  cv::imshow(strImageName, image);
	  cv::waitKey(1);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  bool checkIfBlobsCrossedTheLine(itms::Config& _conf, std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount) {
	  bool blnAtLeastOneBlobCrossedTheLine = false;

	  for (auto blob : blobs) {

		  if (blob.blnStillBeingTracked == true && blob.totalVisibleCount >= _conf.minVisibleCount && blob.centerPositions.size() >= 2) {
			  int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			  int currFrameIndex = (int)blob.centerPositions.size() - 1;

			  if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition) {
				  carCount++;
				  blnAtLeastOneBlobCrossedTheLine = true;
			  }
		  }

	  }

	  return blnAtLeastOneBlobCrossedTheLine;
  }
  bool checkIfBlobsCrossedTheLine(itms::Config& _conf, std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy, cv::Point Pt1, cv::Point Pt2, int &carCount, int &truckCount, int &bikeCount) {
	  bool blnAtLeastOneBlobCrossedTheLine = false;

	  for (auto blob : blobs) {

		  if (blob.blnStillBeingTracked == true && blob.totalVisibleCount >= _conf.minVisibleCount && blob.centerPositions.size() >= 2) {
			  int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			  int currFrameIndex = (int)blob.centerPositions.size() - 1;

			  // Horizontal Line
			  if (/*blob.centerPositions[currFrameIndex].x > Pt1.x
				  && blob.centerPositions[currFrameIndex].x < Pt2.x
				  &&  blob.centerPositions[prevFrameIndex].y < std::max(Pt2.y,Pt1.y)
				  && blob.centerPositions[currFrameIndex].y >= std::min(Pt1.y,Pt2.y)*/ isPointBelowLine(Pt1, Pt2, blob.centerPositions[prevFrameIndex]) ^ isPointBelowLine(Pt1, Pt2, blob.centerPositions[currFrameIndex]))
			  {
				  carCount++;

#ifdef SHOW_STEPS
				  cv::Mat crop = Mat::zeros(Size(blob.currentBoundingRect.width, blob.currentBoundingRect.height), imgFrame2Copy.type());
				  crop = imgFrame2Copy(blob.currentBoundingRect).clone();
				  cv::imwrite("D:\\sangkny\\dataset\\test.png", crop);
				  cv::imshow("cropImage", crop);
				  cv::waitKey(1);
				  cout << "blob track id: " << blob.id << " is crossing the line." << endl;
				  cout << "blob infor: (Age, totalSurvivalFrames, ShowId)-(" << blob.age << ", " << blob.totalVisibleCount << ", " << blob.showId << ")" << endl;
#endif
				  blnAtLeastOneBlobCrossedTheLine = true;
			  }
		  }

	  }

	  return blnAtLeastOneBlobCrossedTheLine;
  }

  
  // this fucntion can detect the status of the blob and update the blob status, then the next module can erase the blob if necessary
  bool checkIfBlobsCrossedTheBoundary(itms::Config& _conf, std::vector<Blob> &blobs,/* cv::Mat &imgFrame2Copy,*/ itms::LaneDirection _laneDirection, std::vector<cv::Point>& _tboundaryPts) {
	  bool blnAtLeastOneBlobCrossedTheBoundary = false;
	  // boundary should be clockwise direction 
	  assert(_tboundaryPts.size() == 4); // the Tracking boudnary should be insize the real boundary
	  if (_tboundaryPts.size() < 4)
		  return blnAtLeastOneBlobCrossedTheBoundary;

	  // declare two top/left  and right/bottom lines according to Lane Direction
	  std::vector<cv::Point> tlLine(2); // top/ left line 
	  std::vector<cv::Point> brLine(2); // bottom/ right line	

	  switch (_laneDirection) { // if we put this at the beginning like configuration, the processing time will be reduced 
	  case itms::LD_NORTH:
	  case itms::LD_NORTHEAST:
	  case itms::LD_NORTHWEST:
		  tlLine.at(0) = _tboundaryPts[0]; // --> begin
		  tlLine.at(1) = _tboundaryPts[1]; // --> end
		  brLine.at(0) = _tboundaryPts[3]; // 
		  brLine.at(1) = _tboundaryPts[2];
		  break;

	  case itms::LD_SOUTH:
	  case itms::LD_SOUTHEAST:
	  case itms::LD_SOUTHWEST:
		  tlLine.at(0) = _tboundaryPts[2]; // <-- begin
		  tlLine.at(1) = _tboundaryPts[3]; // <-- end
		  brLine.at(0) = _tboundaryPts[1]; // 
		  brLine.at(1) = _tboundaryPts[0];
		  break;	  

	  case itms::LD_EAST:
		  tlLine.at(0) = _tboundaryPts[0]; // --> begin
		  tlLine.at(1) = _tboundaryPts[3]; // --> end
		  brLine.at(0) = _tboundaryPts[1]; // 
		  brLine.at(1) = _tboundaryPts[2];

		  break;
	  case itms::LD_WEST:
		  tlLine.at(0) = _tboundaryPts[3]; // <-- begin
		  tlLine.at(1) = _tboundaryPts[0]; // <-- end
		  brLine.at(0) = _tboundaryPts[2]; // 
		  brLine.at(1) = _tboundaryPts[1];
		  break;
	  default: // NORTH
		  tlLine.at(0) = _tboundaryPts[0]; // --> begin
		  tlLine.at(1) = _tboundaryPts[1]; // --> end
		  brLine.at(0) = _tboundaryPts[3]; // 
		  brLine.at(1) = _tboundaryPts[2];
		  break;
	  }

	  std::vector<Blob>::iterator blob = blobs.begin();
	  while (blob != blobs.end()) {

		  if (blob->blnStillBeingTracked == true && blob->totalVisibleCount >= _conf.minVisibleCount && blob->centerPositions.size() >= 2) {
			  int prevFrameIndex = (int)blob->centerPositions.size() - 2;
			  int currFrameIndex = (int)blob->centerPositions.size() - 1;

			  // Cross line checking
			  bool tlFlag_pre = false, tlFlag_cur = false, brFlag_pre = false, brFlag_cur = false; // flag for crossing the boundary
			  tlFlag_pre = isPointBelowLine(tlLine.at(0), tlLine.at(1), blob->centerPositions[prevFrameIndex]);
			  tlFlag_cur = isPointBelowLine(tlLine.at(0), tlLine.at(1), blob->centerPositions[currFrameIndex]); // top left line first
			  if (tlFlag_pre ^ tlFlag_cur) // XOR
			  { // update 
				// normal:
				// blob is out from inside, therefore, it should be erased
				// and the driving direction is forward moving
				// abnorma: Wrong Way Driving
#ifdef SHOW_STEPS
				  cv::Mat crop = Mat::zeros(Size(blob->currentBoundingRect.width, blob->currentBoundingRect.height), imgFrame2Copy.type());
				  crop = imgFrame2Copy(blob->currentBoundingRect).clone();
				  cv::imwrite("D:\\sangkny\\dataset\\test.png", crop);
				  cv::imshow("cropImage", crop);
				  cv::waitKey(1);
				  cout << "blob track id: " << blob->id << " is crossing the line." << endl;
				  cout << "blob infor: (Age, totalSurvivalFrames, ShowId)-(" << blob->age << ", " << blob->totalVisibleCount << ", " << blob->showId << ")" << endl;
				  cout << " --> --> tbLine: This object should be eliminated -------> \n";
#endif
				  blnAtLeastOneBlobCrossedTheBoundary = true;
				  if (tlFlag_pre) { // forward moving to go out of boundary, eliminate
					  if (_conf.debugGeneralDetail)
						  cout << " a blob: " << blob->id << " is eliminated at boundary!!! \n\n\n\n";
					  blob = blobs.erase(blob);
					  continue;
				  }
				  else {
					  blob->os = OS_MOVING_BACKWARD; // it should be notified 
					  blob->fos = OS_MOVING_BACKWARD;
					  blob->os_pro = 1.0;				// 100% confirmed
				  }

			  }
			  else { // to save the computation, I used if separately
				  brFlag_pre = isPointBelowLine(brLine.at(0), brLine.at(1), blob->centerPositions[prevFrameIndex]);
				  brFlag_cur = isPointBelowLine(brLine.at(0), brLine.at(1), blob->centerPositions[currFrameIndex]); // top left line first
				  if (brFlag_pre ^ brFlag_cur) {
#ifdef SHOW_STEPS
					  cv::Mat crop = Mat::zeros(Size(blob->currentBoundingRect.width, blob->currentBoundingRect.height), imgFrame2Copy.type());
					  crop = imgFrame2Copy(blob->currentBoundingRect).clone();
					  cv::imwrite("D:\\sangkny\\dataset\\test.png", crop);
					  cv::imshow("cropImage", crop);
					  cv::waitKey(1);
					  cout << "blob track id: " << blob->id << " is crossing the line." << endl;
					  cout << "blob infor: (Age, totalSurvivalFrames, ShowId)-(" << blob->age << ", " << blob->totalVisibleCount << ", " << blob->showId << ")" << endl;
					  cout << " --> --> brLine: This object should be eliminated -------> \n";
#endif
					  blnAtLeastOneBlobCrossedTheBoundary = true;
					  if (brFlag_cur) { //out of boundary and WWR (wrong way driving)
						  if (_conf.debugGeneralDetail)
							  cout << " A blob :" << blob->id << "at br line is erased !! \n\n\n\n";
						  blob = blobs.erase(blob);
						  continue;
					  }
					  else { // in 
						  blob->os = OS_MOVING_FORWARD; // check this out
						  blob->fos = OS_MOVING_FORWARD; // 
						  blob->os_pro = 1.0;			 // it is 100 % confirmed 	
					  }

				  }

			  }
		  }
		  ++blob;
	  }// while

	  return blnAtLeastOneBlobCrossedTheBoundary;
  }
  bool checkIfPointInBoundary(const itms::Config& _conf, const cv::Point& p1, const std::vector<cv::Point> &_tboundaryPts) {
	  //if(p1.inside())
	  return cv::pointPolygonTest(_tboundaryPts, static_cast<cv::Point2f>(p1), false)>0; // + inside , - outside
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  void drawBlobInfoOnImage(itms::Config& _conf, std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {

	  for (unsigned int i = 0; i < blobs.size(); i++) {

		  if (blobs[i].blnStillBeingTracked == true && blobs[i].totalVisibleCount >1) {
			  int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
			  double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
			  int intFontThickness = (int)std::round(dblFontScale * 1.0);
			  string infostr, status;
			  if (blobs[i].fos == OS_STOPPED) {
				  status = " STOP";
				  cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_BLUE, 2);
			  }
			  else if (blobs[i].fos == OS_MOVING_FORWARD) {
				  status = " MV";
				  cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_GREEN, 2);
			  }
			  else if (blobs[i].fos == OS_MOVING_BACKWARD) {
				  status = " WWR"; // wrong way on a road
				  cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);
			  }
			  else {
				  status = " ND"; // not determined
				  cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_YELLOW, 2);
			  }
			  
			  infostr = std::to_string(blobs[i].id) + status + blobs[i].getBlobClass();
			  if(_conf.debugShowImagesDetail)
				infostr =  std::to_string(blobs[i].id) + status + blobs[i].getBlobClass() +" " + std::to_string(blobs[i].speed) + " km/h";
			  cv::putText(imgFrame2Copy, infostr/*std::to_string(blobs[i].id)*/, blobs[i].currentBoundingRect.tl()/*centerPositions.back()*/, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
			  if (_conf.debugTrace) {
				  // draw the trace of object
				  std::vector<cv::Point> centroids2 = blobs[i].centerPositions;
				  for (std::vector<cv::Point>::iterator it3 = centroids2.end() - 1; it3 != centroids2.begin(); --it3)
				  {
					  cv::circle(imgFrame2Copy, cv::Point((*it3).x, (*it3).y), 1, SCALAR_YELLOW, -1); // draw the trace of the object with Yellow                                
					  if (centroids2.end() - it3 > _conf.numberOfTracePoints)
						  break;
				  }

			  }

		  }
	  }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {

	  int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	  double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
	  int intFontThickness = (int)std::round(dblFontScale * 1.5);

	  cv::Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);

	  cv::Point ptTextTopRightPosition;

	  ptTextTopRightPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25);
	  ptTextTopRightPosition.y = (int)((double)textSize.height * 1.25);

	  cv::putText(imgFrame2Copy, std::to_string(carCount), ptTextTopRightPosition, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);

  }
  void drawRoadRoiOnImage(std::vector<std::vector<cv::Point>> &_roadROIPts, cv::Mat &_srcImg) {
	  for (int i = 0; i<_roadROIPts.size(); i++)
		  for (int j = 0; j < _roadROIPts.at(i).size(); j++) {
			  cv::line(_srcImg, _roadROIPts.at(i).at(j), _roadROIPts.at(i).at((j + 1) % _roadROIPts.at(i).size()), SCALAR_YELLOW, 1);
		  }
  }

  // utils
  float getDistanceInMeterFromPixels(const std::vector<cv::Point2f> &srcPx, const cv::Mat &transmtx /* 3x3*/, const float _laneLength, const bool flagLaneDirectionTop2Bottom) {
	  assert(transmtx.size() == cv::Size(3, 3));
	  std::vector<cv::Point2f> H_Px;
	  bool flagLaneDirection = flagLaneDirectionTop2Bottom;
	  float laneLength = _laneLength, distance = 0;

	  cv::perspectiveTransform(srcPx, H_Px, transmtx);
	  distance = (flagLaneDirection) ? round(H_Px.back().y) : laneLength - round(H_Px.back().y);

	  return distance;
  }
  cv::Point2f cvtPx2RealPx(const cv::Point2f &srcPx, const cv::Mat &transmtx /* 3x3*/) {
	  assert(transmtx.size() == cv::Size(3, 3));
	  std::vector<cv::Point2f> _srcPx;
	  std::vector<cv::Point2f> H_Px;  
	  
	  _srcPx.push_back(srcPx);
	  cv::perspectiveTransform(_srcPx, H_Px, transmtx);

	  return H_Px.back();
  }
  float getNCC(itms::Config& _conf, cv::Mat &bgimg, cv::Mat &fgtempl, cv::Mat &fgmask, int match_method/* cv::TM_CCOEFF_NORMED*/, bool use_mask/*false*/) {
	  //// template matching algorithm implementation, demo	
	  if (0 && _conf.debugGeneralDetail) {
		  string ty = type2str(bgimg.type());
		  printf("Matrix: %s %dx%d \n", ty.c_str(), bgimg.cols, bgimg.rows);
		  ty = type2str(fgtempl.type());
		  printf("Matrix: %s %dx%d \n", ty.c_str(), fgtempl.cols, fgtempl.rows);
	  }
	  assert(bgimg.type() == fgtempl.type());

	  cv::Mat bgimg_gray, fgtempl_gray, res;
	  float ncc;

	  if (bgimg.channels() > 1) {
		  cvtColor(bgimg, bgimg_gray, CV_BGR2GRAY);
		  cvtColor(fgtempl, fgtempl_gray, CV_BGR2GRAY);
	  } {
		  bgimg_gray = bgimg.clone();
		  fgtempl_gray = fgtempl.clone();
	  }

	  bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
	  if (0 && _conf.debugShowImages && _conf.debugShowImagesDetail) {
		  imshow("img", bgimg_gray);
		  imshow("template image", fgtempl_gray);
		  waitKey(1);
	  }
	  if (use_mask && method_accepts_mask)
	  {
		  matchTemplate(bgimg_gray, fgtempl_gray, res, match_method, fgmask);
	  }
	  else
	  {
		  matchTemplate(bgimg_gray, fgtempl_gray, res, match_method);
	  }
	  ncc = res.at<float>(0, 0); // should be float type [-1 1]

	  if (_conf.debugGeneral && _conf.debugGeneralDetail) {
		  cout << " NCC value is : " << res << endl;
		  cout << " variable ncc: " << ncc << endl;
	  }

	  return ncc;
  }

  // call back functions 
  //// template matching algorithm implementation, demo
  //bool use_mask = false;
  //Mat img; Mat templ; Mat result; Mat mask;
  //char* image_window = "Source Image";
  //char* result_window = "Result window";
  //int match_method = cv::TM_CCOEFF_NORMED;
  //int max_Trackbar = 5;
  //void MatchingMethod(int, void*)
  //{
  //	Mat img_display;
  //	img.copyTo(img_display);
  //	int result_cols = img.cols - templ.cols + 1;
  //	int result_rows = img.rows - templ.rows + 1;
  //	result.create(result_rows, result_cols, CV_32FC1);
  //	bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
  //	if (use_mask && method_accepts_mask)
  //	{
  //		matchTemplate(img, templ, result, match_method, mask);
  //	}
  //	else
  //	{
  //		matchTemplate(img, templ, result, match_method);
  //	}
  //	
  //	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
  //	double minVal; double maxVal; Point minLoc; Point maxLoc;
  //	Point matchLoc;
  //	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
  //	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
  //	{
  //		matchLoc = minLoc;
  //	}
  //	else
  //	{
  //		matchLoc = maxLoc;
  //	}
  //	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
  //	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
  //	imshow(image_window, img_display);
  //	imshow(result_window, result);
  //	return;
  //}

  string type2str(int type) {
	  string r;

	  uchar depth = type & CV_MAT_DEPTH_MASK;
	  uchar chans = 1 + (type >> CV_CN_SHIFT);

	  switch (depth) {
	  case CV_8U:  r = "8U"; break;
	  case CV_8S:  r = "8S"; break;
	  case CV_16U: r = "16U"; break;
	  case CV_16S: r = "16S"; break;
	  case CV_32S: r = "32S"; break;
	  case CV_32F: r = "32F"; break;
	  case CV_64F: r = "64F"; break;
	  default:     r = "User"; break;
	  }

	  r += "C";
	  r += (chans + '0');

	  return r;
  }
  int InterSectionRect(cv::Rect &rect1, cv::Rect &rect2) {
	  // -------------------------------------
	  // returns intersection status when one is included to the other one
	  // no intersection -1,
	  // exist intersect 0 with a sub_region
	  // rect1 includes rect2 1
	  // rect2 includes rect1 2
	  // rect1 == rect 2  3
	  // -------------------------------------
	  int retvalue = -1;

	  cv::Rect intRect = (rect1 & rect2);
	  bool intersects = (intRect.area() > 0); // intersection 
	  if (intersects) {
		  retvalue = 0;
		  if (rect1.area() == rect2.area())
			  retvalue = 3;
		  else if (rect1.area() > rect2.area()) {
			  if (rect2.area() == intRect.area())
				  retvalue = 1;
		  }
		  else { // rect1 < rect2
			  if (rect1.area() == intRect.area())
				  retvalue = 2;
		  }
	  }

	  return retvalue;
  }

  // class Blob image processing blob_imp

  void updateBlobProperties(const Config& _conf, itms::Blob &updateBlob, itms::ObjectStatus &curStatus, const double _speed) {
	  itms::ObjectStatus prevOS = updateBlob.os;						// previous object status
	  int minConsecutiveFramesForOS = _conf.minConsecutiveFramesForOS;	// minConsecutiveFrames For OS
	  switch (curStatus) { // at this point, blob.os is the previous status !!!	  

	  case OS_MOVING_FORWARD:
		  updateBlob.os_mvForward_cnter++;
		  updateBlob.os_NumOfConsecutivemvForward_cnter = (prevOS == OS_MOVING_FORWARD) ? updateBlob.os_NumOfConsecutivemvForward_cnter + 1 : 1;
		  updateBlob.os_NumOfConsecutiveStopped_cnter = 0;  // reset the consecutive counter
		  //blob.os_NumOfConsecutivemvForward_cnter = 1;
		  updateBlob.os_NumOfConsecutivemvBackward_cnter = 0;
		  updateBlob.fos = (updateBlob.os_NumOfConsecutivemvForward_cnter >= minConsecutiveFramesForOS) ? OS_MOVING_FORWARD : OS_NOTDETERMINED;
		  break;

	  case OS_STOPPED:
		  updateBlob.os_stopped_cnter++;
		  updateBlob.os_NumOfConsecutiveStopped_cnter = (prevOS == OS_STOPPED) ? updateBlob.os_NumOfConsecutiveStopped_cnter + 1 : 1;
		  //blob.os_NumOfConsecutiveStopped_cnter = 1;		// reset the consecutive counter
		  updateBlob.os_NumOfConsecutivemvForward_cnter = 0;
		  updateBlob.os_NumOfConsecutivemvBackward_cnter = 0;
		  updateBlob.fos = (updateBlob.os_NumOfConsecutiveStopped_cnter >= minConsecutiveFramesForOS) ? OS_STOPPED : OS_NOTDETERMINED;
		  break;	  

	  case OS_MOVING_BACKWARD:
		  updateBlob.os_mvBackward_cnter++;
		  updateBlob.os_NumOfConsecutivemvBackward_cnter = (prevOS == OS_MOVING_BACKWARD) ? updateBlob.os_NumOfConsecutivemvBackward_cnter + 1 : 1;
		  updateBlob.os_NumOfConsecutiveStopped_cnter = 0;  // reset the consecutive counter
		  updateBlob.os_NumOfConsecutivemvForward_cnter = 0;
		  //blob.os_NumOfConsecutivemvBackward_cnter = 1;
		  updateBlob.fos = (updateBlob.os_NumOfConsecutivemvBackward_cnter >= minConsecutiveFramesForOS) ? OS_MOVING_BACKWARD : OS_NOTDETERMINED;
		  break;

	  case OS_NOTDETERMINED:
		  updateBlob.os_notdetermined_cnter++;
		  updateBlob.os_NumOfConsecutiveStopped_cnter = 0;  // reset the consecutive counter
		  updateBlob.os_NumOfConsecutivemvForward_cnter = 0;
		  updateBlob.os_NumOfConsecutivemvBackward_cnter = 0;

		  break;
	  defualt:
		  // no nothing...
		  cout << " object status is not correct!! inside updateBlobProperties \n";
		  break;
	  }
	  // update the os_prob, it needs to improve
	  updateBlob.os_pro = (float)updateBlob.totalVisibleCount / (float)max(1, updateBlob.age);
	  updateBlob.speed = _speed;

  }
  itms::ObjectStatus computeObjectStatusProbability(const itms::Blob &srcBlob) {
	  //determine the object status with probability computations
	  // weight policy : current status consecutive counter, others, 1
	  float total_NumOfConsecutiveCounter = (1 + 1 + srcBlob.os_NumOfConsecutivemvBackward_cnter + srcBlob.os_NumOfConsecutivemvForward_cnter + srcBlob.os_NumOfConsecutiveStopped_cnter);
	  float stop_prob = ((srcBlob.os_NumOfConsecutiveStopped_cnter + 1)*srcBlob.os_stopped_cnter);
	  float forward_prob = ((srcBlob.os_NumOfConsecutivemvForward_cnter + 1)*srcBlob.os_mvForward_cnter);
	  float backward_prob = ((srcBlob.os_NumOfConsecutivemvBackward_cnter + 1)*srcBlob.os_mvBackward_cnter);
	  float nondetermined_prob = 0; // dummy
									// find max vale
	  float os_max = -1;
	  int max_index = 0;
	  vector<float> os_vector; // put the same order with ObjectStatus
	  os_vector.push_back(stop_prob);
	  os_vector.push_back(forward_prob);
	  os_vector.push_back(backward_prob);
	  os_vector.push_back(nondetermined_prob); // dummy
	  assert((int)(ObjectStatus::OS_NOTDETERMINED + 1) == (int)os_vector.size());  // size check!
	  for (int i = 0; i < os_vector.size(); i++) {
		  if (os_vector.at(i) > os_max) {
			  os_max = os_vector.at(i);
			  max_index = i;
		  }
	  }
	  return  ObjectStatus(max_index);
  }
  auto cmp = [](std::pair<string, float > const & a, std::pair<string, float> const & b)
  {
	  return a.second > b.second; // descending order
  };

  // std::sort(items.begin(), items.end(), cmp);
  // classificy an object with distance and its size
  void classifyObjectWithDistanceRatio(itms::Config& _conf, Blob &srcBlob, float distFromZero/* distance from the starting point*/, ObjectClass & objClass, float& fprobability)
  {
	  // --- algorithm ---------------------------------------------------------------------
	  // get the pair infors for object, 
	  // 0. get the current object information of width and height
	  // 1. get the width and height of the reference class object from the given distance
	  // 2. compute the probability for each class, vector<pair<enum, float >>
	  // 3. sort with probabilty in descending order
	  // 4. determine the class for the given object
	  // ------------------------------------------------------------------------------------
	  float fmininum_class_prob = 0.5;  // minimum probability for declaring the class type
	  std::vector<std::pair<std::string, float>> objClassProbs; // object class with probabilities
	  float fdistance = distFromZero, prob = 0.f, perc_Thres = 0.5; // 25% error range	
	  float fWidthHeightWeightRatio_Width = 0.7; // width 0.7 height 0.3

	  if (_conf.bgsubtype == BgSubType::BGS_CNT)
		  perc_Thres = 1.0; // should be bigger

	  int tgtWidth, tgtHeight, refWidth, refHeight; // target, reference infors
	  float tgtWidthHeightRatio, tgtCredit = 1.1;   // credit 10 %
	  tgtWidth = srcBlob.currentBoundingRect.width;
	  tgtHeight = srcBlob.currentBoundingRect.height;
	  tgtWidthHeightRatio = (float)tgtHeight / (float)tgtWidth; // give the more credit according to the shape for vehicle or human 10 %


	  vector<float> objWidth;     // for panelty against distance
	  vector<float> objHeight;
	  //// configuration 
	  //vector<float> sedan_h = { -0.00004444328872f, 0.01751602326f, -2.293443176f, 112.527668f }; // scale factor 0.5
	  //vector<float> sedan_w = { -0.00003734137716f, 0.01448943505f, -1.902199174f, 98.56691135f };
	  //vector<float> suv_h = { -0.00005815785621f, 0.02216859672f, -2.797603666f, 139.0638999f };
	  //vector<float> suv_w = { -0.00004854032314f, 0.01884736545f, -2.425686251f, 121.9226426f };
	  //vector<float> truck_h = { -0.00006123592908f, 0.02373661426f, -3.064585294f, 149.6535855f };
	  //vector<float> truck_w = { -0.00003778247771f, 0.015239317f, -2.091105041f, 110.7544702f };
	  //vector<float> human_h = { -0.000002473245036f, 0.001813179193f, -0.5058008988f, 49.27950311f };
	  //vector<float> human_w = { -0.000003459461125f, 0.001590306464f, -0.3208648543f, 28.23621306f };
	  //ITMSPolyValues polyvalue_sedan_h(sedan_h, sedan_h.size());
	  //ITMSPolyValues polyvalue_sedan_w(sedan_w, sedan_w.size());
	  //ITMSPolyValues polyvalue_suv_h(suv_h, suv_h.size());
	  //ITMSPolyValues polyvalue_suv_w(suv_w, suv_w.size());
	  //ITMSPolyValues polyvalue_truck_h(truck_h, truck_h.size());
	  //ITMSPolyValues polyvalue_truck_w(truck_w, truck_w.size());
	  //ITMSPolyValues polyvalue_human_h(human_h, human_h.size());
	  //ITMSPolyValues polyvalue_human_w(human_w, human_w.size());
	  //// 
	  objWidth.push_back(_conf.polyvalue_sedan_w.getPolyValue(fdistance));
	  objWidth.push_back(_conf.polyvalue_suv_w.getPolyValue(fdistance));
	  objWidth.push_back(_conf.polyvalue_truck_w.getPolyValue(fdistance));
	  objWidth.push_back(_conf.polyvalue_human_w.getPolyValue(fdistance));
	  sort(objWidth.begin(), objWidth.end(), greater<float>()); // descending order
	  float minRefWidth = objWidth.at(objWidth.size() - 1), maxRefWidth = objWidth.at(0);

	  objHeight.push_back(_conf.polyvalue_sedan_h.getPolyValue(fdistance));
	  objHeight.push_back(_conf.polyvalue_suv_h.getPolyValue(fdistance));
	  objHeight.push_back(_conf.polyvalue_truck_h.getPolyValue(fdistance));
	  objHeight.push_back(_conf.polyvalue_human_h.getPolyValue(fdistance));
	  sort(objHeight.begin(), objHeight.end(), greater<float>());
	  float minRefHeight = objWidth.at(objWidth.size() - 1), maxRefHeight = objWidth.at(0);

	  // size constraints

	  if (tgtWidth > maxRefWidth*(1 + perc_Thres) || tgtWidth < minRefWidth*(1 - perc_Thres))
		  tgtWidth = 0;
	  if (tgtHeight > maxRefHeight*(1 + perc_Thres) || tgtHeight < minRefHeight*(1 - perc_Thres))
		  tgtHeight = 0;

	  // sedan
	  refHeight = _conf.polyvalue_sedan_h.getPolyValue(fdistance);
	  refWidth = _conf.polyvalue_sedan_w.getPolyValue(fdistance);
	  prob = fWidthHeightWeightRatio_Width*(refWidth - fabs(refWidth - tgtWidth)) / refWidth +
		  (1.f - fWidthHeightWeightRatio_Width)*(refHeight - fabs(refHeight - tgtHeight)) / refHeight;
	  //prob = prob*tgtWidthHeightRatio>= 1.0? 1.0: prob*tgtWidthHeightRatio; // size constraint min(1.0, prob*tgtWidthHeightRatio)
	  objClassProbs.push_back(pair<string, float>("sedan", prob));
	  // suv
	  refHeight = _conf.polyvalue_suv_h.getPolyValue(fdistance);
	  refWidth = _conf.polyvalue_suv_w.getPolyValue(fdistance);
	  prob = fWidthHeightWeightRatio_Width*(refWidth - fabs(refWidth - tgtWidth)) / refWidth +
		  (1.f - fWidthHeightWeightRatio_Width)*(refHeight - fabs(refHeight - tgtHeight)) / refHeight;
	  //prob = prob*tgtWidthHeightRatio >= 1.0 ? 1.0 : prob*tgtWidthHeightRatio; // size constraint min(1.0, prob*tgtWidthHeightRatio)
	  objClassProbs.push_back(pair<string, float>("suv", prob));
	  // truck
	  refHeight = _conf.polyvalue_truck_h.getPolyValue(fdistance);
	  refWidth = _conf.polyvalue_truck_w.getPolyValue(fdistance);
	  prob = fWidthHeightWeightRatio_Width*(refWidth - fabs(refWidth - tgtWidth)) / refWidth +
		  (1.f - fWidthHeightWeightRatio_Width)*(refHeight - fabs(refHeight - tgtHeight)) / refHeight;
	  //prob = prob*tgtWidthHeightRatio >= 1.0 ? 1.0 : prob*tgtWidthHeightRatio; // size constraint min(1.0, prob*tgtWidthHeightRatio)
	  objClassProbs.push_back(pair<string, float>("truck", prob));

	  // human
	  refHeight = _conf.polyvalue_human_h.getPolyValue(fdistance);
	  refWidth = _conf.polyvalue_human_w.getPolyValue(fdistance);
	  prob = (1.f - fWidthHeightWeightRatio_Width)*(refWidth - fabs(refWidth - tgtWidth)) / refWidth +
		  (fWidthHeightWeightRatio_Width)*(refHeight - fabs(refHeight - tgtHeight)) / refHeight; // height is more important for human 
	  prob = prob*tgtWidthHeightRatio >= 1.0 ? 1.0 : prob*tgtWidthHeightRatio; // size constraint min(1.0, prob*tgtWidthHeightRatio)
	  objClassProbs.push_back(pair<string, float>("human", prob));

	  sort(objClassProbs.begin(), objClassProbs.end(), cmp); // sort the prob in decending order
	  string strClass = objClassProbs.at(0).first; // class
	  fprobability = objClassProbs.at(0).second;   // prob 
	  if (fprobability >= fmininum_class_prob) {   // the if and its below can be replaced with (?) a:b; for speed
		  if (strClass == "human") {
			  const float minMax_threshold = 0.5; // 25 % margin
			  const float h_w_ratio = (float)srcBlob.currentBoundingRect.height / srcBlob.currentBoundingRect.width;
			  const float RefRatio = 2;			// reference ration 2 : 1
			  if (((RefRatio - minMax_threshold) > h_w_ratio)
				  || (h_w_ratio > (RefRatio + minMax_threshold))) { // need to change the status of object class???? 
				  objClass = ObjectClass::OC_OTHER;
				  if (_conf.debugGeneralDetail)
					  cout << "Human detection condition is not matched in detectCascadeRoiHuman.\n  with rect : " << srcBlob.currentBoundingRect << endl;				  
			  }
			  else {
				  objClass = ObjectClass::OC_HUMAN;
			  }
		  }
		  else
		  {
			  objClass = ObjectClass::OC_VEHICLE;
		  }

	  }
	  else {
		  objClass = ObjectClass::OC_OTHER; // not determined
	  }
	  // update currentBlob
	  srcBlob.oc_prob = fprobability;
	  srcBlob.oc = objClass;
  }

  bool checkObjectStatus(const itms::Config & _conf, const cv::Mat& _curImg, std::vector<Blob>& _Blobs, itms::ITMSResult & _itmsRes)
  {   
	  bool checkStatus = false;	  

	  std::vector<Blob>::iterator curBlob = _Blobs.begin();
	  while (curBlob != _Blobs.end()) {
		  if (curBlob->bNotifyMessage || (_conf.bStrictObjEvent && curBlob->fos == ObjectStatus::OS_NOTDETERMINED)) 
		  {   // notified, then skip, if bStrictObjEvent, strict determination is conducted according to ObjectStatus
			  ++curBlob;
			  continue;
		  }
		  if (curBlob->oc == ObjectClass::OC_HUMAN ) {
			  if(curBlob->oc_prob <= 0.99){
				  std::vector<cv::Rect> _people;
				  detectCascadeRoiHuman(_conf, _curImg, curBlob->currentBoundingRect,_people);
				  if(_people.size()==0){
					  ++curBlob;
					  continue;
				  }
			  }
			  _itmsRes.objClass.push_back(std::pair<int, int>(curBlob->id, ObjectClass::OC_HUMAN));
			  _itmsRes.objStatus.push_back(std::pair<int, int>(curBlob->id, curBlob->fos));
			  _itmsRes.objRect.push_back(curBlob->currentBoundingRect);
			  _itmsRes.objSpeed.push_back(curBlob->speed);
			  checkStatus = true;
			  curBlob->bNotifyMessage = (_conf.bNoitifyEventOnce )? true: false;
		  }
		  else {
			  // WWD
			  // STOP
			  if (curBlob->oc!= ObjectClass::OC_OTHER &&
			  (curBlob->fos == ObjectStatus::OS_MOVING_BACKWARD || 
			  curBlob->fos == ObjectStatus::OS_STOPPED)) {
				  _itmsRes.objClass.push_back(std::pair<int, int>(curBlob->id, curBlob->oc));
				  _itmsRes.objStatus.push_back(std::pair<int, int>(curBlob->id, curBlob->fos));
				  _itmsRes.objRect.push_back(curBlob->currentBoundingRect);
				  _itmsRes.objSpeed.push_back(curBlob->speed);
				  checkStatus = true;
				  curBlob->bNotifyMessage = (_conf.bNoitifyEventOnce) ? true : false;
			  }
		  }
		  ++curBlob;
	  }
	  return checkStatus;
  }

  void detectCascadeRoi(itms::Config& _conf, cv::Mat img, cv::Rect& rect)
  { /* please see more details in Object_Detector_Cascade Project */
	  Mat roiImg = img(rect).clone();
	  Mat hogImg;
	  // debug details
	  hogImg = roiImg.clone();
	  int casWidth = 128; // ratio is 1: 1 for width to height
	  int svmWidth = 64 * 1.5, svmHeight = 128 * 1.5;

	  // adjust cascade window image
	  float casRatio = (float)casWidth / roiImg.cols;
	  //bool debugGeneralDetails = true;
	  //bool debugShowImage = true;

	  resize(roiImg, roiImg, Size(), casRatio, casRatio);

	  Size img_size = roiImg.size();
	  vector<Rect> object;
	  vector<Rect> people;
	  _conf.cascade.detectMultiScale(roiImg, object, 1.1, 5/*1  cascadG */, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(0, 0), img_size); // detectio objects (car)
																															 //cascade.detectMultiScale(img, object, 1.04, 5, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(3, 8), img_size); // detectio objects (people)

																															 // adjust the size : hog is not working if the size of an image is not fitted to the definition	
	  if (hogImg.cols < svmWidth) {
		  float widthRatio = (float)svmWidth / hogImg.cols;
		  resize(hogImg, hogImg, Size(), widthRatio, widthRatio); // same ratio is applied to both direction
	  }
	  if (hogImg.rows < svmHeight) {
		  float heightRatio = (float)svmHeight / hogImg.rows;
		  resize(hogImg, hogImg, Size(), heightRatio, heightRatio);
	  }
	  _conf.hog.detectMultiScale(hogImg, people, 0, Size(4, 4), Size(8, 8), 1.05, 2, false);							// detect people
	  if (_conf.debugGeneralDetail) {
		  std::cout << "Total: " << object.size() << " cars detected." << std::endl;
		  std::cout << "=>=> " << people.size() << " people detected." << std::endl;
	  }

	  for (int i = 0; i < (object.size() ? object.size()/*object->total*/ : 0); i++)
	  {
		  Rect r = object.at(i);
		  if (_conf.debugShowImagesDetail)
			  cv::rectangle(roiImg,
				  cv::Point(r.x, r.y),
				  cv::Point(r.x + r.width, r.y + r.height),
				  CV_RGB(255, 0, 0), 2, 8, 0);
	  }
	  for (int i = 0; i < (people.size() ? people.size()/*object->total*/ : 0); i++)
	  {
		  //CvRect *r = (CvRect*)cvGetSeqElem(object, i);
		  Rect r1 = people.at(i);
		  if (_conf.debugShowImagesDetail)
			  cv::rectangle(hogImg,
				  cv::Point(r1.x, r1.y),
				  cv::Point(r1.x + r1.width, r1.y + r1.height),
				  CV_RGB(0, 255, 0), 2, 8, 0);
	  }
	  if (_conf.debugShowImagesDetail) {
		  imshow("cascade image", roiImg);
		  imshow("hog", hogImg);
		  waitKey(1);
	  }

  }
  void detectCascadeRoiVehicle(itms::Config& _conf,/* put config file */cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _cars)
  { /* please see more details in Object_Detector_Cascade Project */
	  Mat roiImg = img(rect).clone();

	  int casWidth = 128; // ratio is 1: 1 for width to height
	  if (_conf.bgsubtype == BgSubType::BGS_CNT)
		  casWidth = (int)((float)casWidth *1.5);

	  // adjust cascade window image
	  float casRatio = (float)casWidth / roiImg.cols;

	  resize(roiImg, roiImg, Size(), casRatio, casRatio);

	  Size img_size = roiImg.size();
	  vector<Rect> object;
	  _conf.cascade.detectMultiScale(roiImg, object, 1.1, 5/*1  cascadG */, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(0, 0), img_size); // detectio objects (car)
																															 //cascade.detectMultiScale(img, object, 1.04, 5, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(3, 8), img_size); // detectio objects (people)
	  if (_conf.debugGeneralDetail) {
		  std::cout << "Total: " << object.size() << " cars are detected in detectCascadeRoiVehicle function." << std::endl;
	  }

	  for (int i = 0; i < (object.size() ? object.size()/*object->total*/ : 0); i++)
	  {
		  Rect r = object.at(i);
		  // check the center point of the given ROI is in the rect of the output
		  Rect tgtRect(roiImg.cols / 2 - 1, roiImg.rows / 2 - 1, 3, 3); // 3x3 at center point of the ROI 
		  Rect inter = (tgtRect & r);
		  if (inter.area())
			  _cars.push_back(r);

		  if (_conf.debugShowImagesDetail) {
			  if (roiImg.channels() < 3)
				  cvtColor(roiImg, roiImg, CV_GRAY2BGR);
			  cv::rectangle(roiImg,
				  r,
				  CV_RGB(255, 0, 0), 2, 8, 0);
			  if (inter.area())
				  cv::rectangle(roiImg, inter, CV_RGB(255, 0, 0), 2, 8, 0);
		  }
	  }
	  if (_conf.debugShowImagesDetail) {
		  imshow("vehicle in detection", roiImg);
		  waitKey(1);
	  }
  }
  // find people in the given ROI
  void detectCascadeRoiHuman(const itms::Config& _conf, /* put config file */const cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _people)
  {
	  // sangkny 2019. 02. 12 : ration constraint actually, with/height 1/2 => 
	  assert(rect.area() > 0);
	  const float minMax_threshold = 0.5; // 25 % margin
	  const float h_w_ratio = (float)rect.height / (float)rect.width;
	  const float RefRatio = 2;			// reference ration 2 : 1
	  if (((RefRatio - minMax_threshold) > h_w_ratio) 
		  || (h_w_ratio > (RefRatio + minMax_threshold))) { // need to change the status of object class???? 
		  if (_conf.debugGeneralDetail)
			  cout << "Human detection condition is not matched in detectCascadeRoiHuman.\n  with rect : " << rect<<endl;

		  return;
	  }

	  /* this function return the location of human according to the detection method
	  1. svm based algorithm which needs more computation time
	  2. cascade haar-like approach, which is fast but not much robust compared to SVM-based approach
	  */

	  // debugging 
	  //bool debugGeneralDetails = true;
	  //bool debugShowImage = true;

	  cv::Mat hogImg = img(rect);// .clone();
	  // debug details

#ifdef _CASCADE_HUMAN
	  int casWidth = 128; // ratio is 1: 1 for width to height
	  if (bgsubtype == BgSubType::BGS_CNT)
		  casWidth = (int)((float)casWidth *1.5);
	  float casRatio = (float)casWidth / hogImg.cols;

	  resize(hogImg, hogImg, Size(), casRatio, casRatio);

	  Size img_size = hogImg.size();
	  vector<Rect> people;
	  cascade.detectMultiScale(hogImg, people, 1.1, 5/*1  cascadG */, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(0, 0), img_size); // detectio objects (car)
																															 //cascade.detectMultiScale(hogImg, people, 1.04, 5, 0 | CV_HAAR_DO_CANNY_PRUNING, cvSize(3, 8), img_size); // detectio objects (people) 
																															 // need to change the xml file for human instead of using cars.xml
#else
	  int svmWidth = 64 * 1.5, svmHeight = 128 * 1.5;
	  if (_conf.bgsubtype == BgSubType::BGS_CNT) {
		  svmWidth = (int)((float)svmWidth * 2/1.5);
		  svmHeight = (int)((float)svmHeight * 2/1.5);
	  }
	  vector<Rect> people;
	  if (hogImg.cols < svmWidth) {
		  float widthRatio = (float)svmWidth / hogImg.cols;
		  resize(hogImg, hogImg, Size(), widthRatio, widthRatio); // same ratio is applied to both direction
	  }
	  if (hogImg.rows < svmHeight) {
		  float heightRatio = (float)svmHeight / hogImg.rows;
		  resize(hogImg, hogImg, Size(), heightRatio, heightRatio);
	  }
	  _conf.hog.detectMultiScale(hogImg, people, 0, Size(4, 4), Size(8, 8), 1.05, 2, false);							// detect people

#endif
	  if (_conf.debugGeneralDetail) {
		  std::cout << "=>=> " << people.size() << " people detected in detect cascadeRoiHuamn function." << std::endl;
	  }

	  for (size_t i = 0; i < (people.size() ? people.size()/*object->total*/ : 0); i++)
	  {
		  Rect r1 = people.at(i);
		  // check the center point of the given ROI is in the rect of the output
		  Rect tgtRect(hogImg.cols / 2 - 1, hogImg.rows / 2 - 1, 3, 3); // 3x3 at center point of the ROI 
		  Rect inter = (tgtRect & r1);
		  if (inter.area())
			  _people.push_back(r1);
		  if (_conf.debugShowImagesDetail) {
			  if (hogImg.channels()<3)
				  cvtColor(hogImg, hogImg, CV_GRAY2BGR);

			  cv::rectangle(hogImg,
				  r1,
				  CV_RGB(0, 255, 0), 2, 8, 0);
			  if (inter.area())
				  cv::rectangle(hogImg, inter, CV_RGB(255, 0, 0), 2, 8, 0);
		  }
	  }
	  if (_conf.debugShowImagesDetail) {
		  imshow("human in HOG SVM", hogImg);
		  waitKey(1);
	  }
  }

  void collectPointsInBlob(Blob &_blob) {

	  //cv::Rect r = itms::expandRect(_blob.currentBoundingRect, 10, 10, _blob.currentBoundingRect.x+_blob.currentBoundingRect.width*2, _blob.currentBoundingRect.y+_blob.currentBoundingRect.height*2);
	  // it is not working (keep increasing the ROI)
	  cv::Rect r = _blob.currentBoundingRect;
	  cv::Point2f center(r.x + 0.5f * r.width, r.y + 0.5f * r.height);
	  const int yStep = 5;
	  const int xStep = 5;

	  for (int y = r.y; y < r.y + r.height; y += yStep)
	  {
		  cv::Point2f pt(0, static_cast<float>(y));
		  for (int x = r.x; x < r.x + r.width; x += xStep)
		  {
			  pt.x = static_cast<float>(x);
			  if (1/*cv::pointPolygonTest(_blob.currentContour, pt, false) > 0*/)
			  {
				  _blob.m_points.push_back(pt);
			  }
		  }
	  }

	  if (_blob.m_points.empty())
	  {
		  _blob.m_points.push_back(center);
	  }

  }
  void collectPointsInBlobs(std::vector<Blob> &_blobs, bool _collectPoints) {
	  if (_collectPoints)
		  for (auto& blob : _blobs) {
			  if (blob.m_points.size()<1)
				  collectPointsInBlob(blob);
		  }
  }
  void getCollectPoints(Blob& _blob, std::vector<Point2f> &_collectedPts)
  {
	  if (_blob.m_points.size()<1) { // no exist and then collect it	
		  const int yStep = 5;
		  const int xStep = 5;
		  cv::Rect region = _blob.currentBoundingRect;

		  for (int y = region.y, yStop = region.y + region.height; y < yStop; y += yStep)
		  {
			  for (int x = region.x, xStop = region.x + region.width; x < xStop; x += xStep)
			  {
				  if (region.contains(cv::Point(x, y)))
				  {
					  _blob.m_points.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
				  }
			  }
		  }

		  if (_collectedPts.empty())
		  {
			  _blob.m_points.push_back(cv::Point2f(region.x + 0.5f * region.width, region.y + 0.5f * region.height));
		  }
	  }

	  for (unsigned int i = 0; i < _blob.m_points.size(); i++)
		  _collectedPts.push_back(_blob.m_points.at(i));

  }
  void predictBlobs(itms::Config& _conf, std::vector<Blob>& tracks/* existing blobs */, cv::UMat prevFrame, cv::UMat curFrame, std::vector<Blob>& predBlobs) {
	  // copy first
	  bool bdebugshowImage = _conf.debugShowImagesDetail;
	  Mat debugImg = Mat::zeros(prevFrame.size(), CV_8UC3);
	  std::vector<cv::Point2f> points[2];

	  points[0].reserve(8 * tracks.size());	// reserve the memory
	  for (auto& track : tracks) // all existing blobs including untracked blob for a while
	  {
		  if (track.currentContour.size() > 0 && track.m_points.size() <= 1)
			  collectPointsInBlob(track);
		  for (const auto& pt : track.m_points)
		  {
			  points[0].push_back(pt);
		  }
	  }
	  if (points[0].empty())
	  {
		  return;
	  }

	  cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
	  cv::Size subPixWinSize(3, 3);
	  cv::Size winSize(25, 25);  // if this size is smaller than 25, the estimated points will be out of expectation
	  if (bdebugshowImage) {
		  for (int i = 0; i<points[0].size(); i++)
			  circle(debugImg, points[0].at(i), 2, Scalar(255, 255, 255));
	  }
	  cv::cornerSubPix(prevFrame, points[0], subPixWinSize, cv::Size(-1, -1), termcrit);
	  if (0 && bdebugshowImage) {
		  for (int i = 0; i<points[0].size(); i++)
			  circle(debugImg, points[0].at(i), 2, Scalar(0, 0, 255));
	  }
	  std::vector<uchar> status;
	  std::vector<float> err;

	  cv::calcOpticalFlowPyrLK(prevFrame, curFrame, points[0], points[1], status, err, winSize, 1, termcrit, 0, 0.001);
	  //cv::cornerSubPix(curFrame, points[1], subPixWinSize, cv::Size(-1, -1), termcrit);
	  if (bdebugshowImage) {
		  for (int i = 0; i<points[1].size(); i++)
			  circle(debugImg, points[1].at(i), 2, Scalar(0, 255, 0));
		  imshow("calcOpticalFlowPyr", debugImg);
		  waitKey(1);
	  }
	  size_t i = 0;
	  for (auto track : tracks)
	  {
		  itms::Blob blob;
		  blob = track;				// inherit information from track
		  cv::Point_<float> m_averagePoint = cv::Point_<float>(0, 0);
		  blob.currentBoundingRect = cv::Rect(0, 0, 0, 0);
		  if (bdebugshowImage) {
			  std::vector<vector<Point>> contours;
			  contours.push_back(blob.currentContour);
			  debugImg = 0;
			  drawContours(debugImg, contours, -1, Scalar(0, 255, 255));
		  }
		  for (auto it = blob.m_points.begin(); it != blob.m_points.end();)
		  {
			  if (status[i] && distanceBetweenPoints(track.centerPositions.at(track.centerPositions.size() - 1), static_cast<Point>(points[1][i] + Point2f(0.5, 0.5))) <= 0.7*track.dblCurrentDiagonalSize) // be whin the range of diagonal distance in the previous blob
			  {
				  *it = points[1][i];
				  m_averagePoint += *it;

				  ++it;
			  }
			  else
			  {
				  it = blob.m_points.erase(it);
			  }

			  ++i;
		  }

		  if (!blob.m_points.empty())
		  {
			  m_averagePoint /= static_cast<float>(blob.m_points.size());

			  cv::Rect br = cv::boundingRect(blob.m_points);
#if 0
			  br.x -= subPixWinSize.width;
			  br.width += 2 * subPixWinSize.width;
			  if (br.x < 0)
			  {
				  br.width += br.x;
				  br.x = 0;
			  }
			  if (br.x + br.width >= curFrame.cols)
			  {
				  br.x = curFrame.cols - br.width - 1;
			  }

			  br.y -= subPixWinSize.height;
			  br.height += 2 * subPixWinSize.height;
			  if (br.y < 0)
			  {
				  br.height += br.y;
				  br.y = 0;
			  }
			  if (br.y + br.height >= curFrame.rows)
			  {
				  br.y = curFrame.rows - br.height - 1;
			  }
#endif
			  blob.currentBoundingRect = br;
			  std::vector<cv::Point> contour, preContour;
			  for (size_t j = 0; j<blob.m_points.size(); j++)
				  preContour.push_back(static_cast<Point>(blob.m_points.at(j) + cv::Point2f(0.5, 0.5))); // rounding 
			  cv::convexHull(preContour, contour, false);

			  blob.currentContour = contour;

			  if (track.blnStillBeingTracked)
				  predBlobs.push_back(blob); // can put predBlobs.push_back(contour) if a basic constructor only with contour
											 // get a real contour or Blob tmp(contour) and push_back;

			  if (bdebugshowImage) {
				  std::vector<vector<Point>> contours;
				  contours.push_back(contour);
				  drawContours(debugImg, contours, -1, Scalar(0, 0, 255));
				  imshow("contours_", debugImg);
				  waitKey(1);
			  }
		  }
	  }
  }

  // itmsFunctions definition
  itmsFunctions::itmsFunctions(Config* config){
	  if (!config->isLoaded) {
		  isConfigFileLoaded  = false;		  
	  }
	  else {
		  _config = config;
		  isConfigFileLoaded = true;
		  Init();
	  }	  
  }
  bool itmsFunctions::Init() {
	  //pBgSub = cv::bgsubcnt::createBackgroundSubtractorCNT(fps, true, fps * 60);
	  pBgSub = createBackgroundSubtractorMOG2();
	  blobs.clear();
	  pastBrightnessLevels.clear();

	  brightnessRoi = _config->AutoBrightness_Rect;
	  m_collectPoints = _config->m_useLocalTracking;
	  blnFirstFrame = true;
	  if (existFileTest(_config->BGImagePath)) {
		  BGImage = cv::imread(_config->BGImagePath);
		  if (!BGImage.empty()) {
			  if (BGImage.channels() > 1)
				  cv::cvtColor(BGImage, BGImage, cv::COLOR_BGR2GRAY);
			  cv::resize(BGImage, BGImage,cv::Size(), _config->scaleFactor, _config->scaleFactor);
			  // sangkny 2019. 02. 09
			  accmImage = BGImage; // copy the background as an initialization
			  // road_mask define
			  road_mask = cv::Mat::zeros(BGImage.size(), BGImage.type());
			  for (int ir = 0; ir<_config->Road_ROI_Pts.size(); ir++)
				  fillConvexPoly(road_mask, _config->Road_ROI_Pts.at(ir).data(), _config->Road_ROI_Pts.at(ir).size(), Scalar(255, 255, 255), 8);

			  if (_config->debugShowImages && _config->debugShowImagesDetail) {
				  cv::Mat debugImg = road_mask.clone();
				  if (debugImg.channels() < 3)
					  cvtColor(debugImg, debugImg, CV_GRAY2BGR);
				  for (int i = 0; i < _config->Boundary_ROI_Pts.size(); i++)
					  line(debugImg, _config->Boundary_ROI_Pts.at(i% _config->Boundary_ROI_Pts.size()), _config->Boundary_ROI_Pts.at((i + 1) % _config->Boundary_ROI_Pts.size()), SCALAR_BLUE, 2);
				  imshow("road mask", debugImg);
				  waitKey(1);
			  }
		  }
	  }
	  else {
		  if (_config->debugGeneralDetail)
			  cout << "Please check the background file : " << _config->BGImagePath << endl;
		  // alternative frame will be the previous frame at processing
	  }
	  return isInitialized = true;
  }

  bool itmsFunctions::process(cv::Mat& curImg, ITMSResult& _itmsRes) {
	  //_itmsRes.objRect.push_back(cv::Rect(1,1,1,1));
	  if (!isInitialized) {
		  cout << "itmsFunctions is not initialized (!)(!)\n";
		  return false;
	  }
	  if (curImg.empty())
		  return false;
	  if (curImg.channels() > 1) {		  
		  cv::cvtColor(curImg, curImg, cv::COLOR_BGR2GRAY);
	  }	  
	  // out side processing will be better if the frame can be maintained out side as well
	  // new approach to save the time
	  resize(curImg, curImg, cv::Size(), _config->scaleFactor, _config->scaleFactor);
	  cv::GaussianBlur(curImg, curImg, cv::Size(5, 5), 0);

	  if (preImg.empty()) {
		  preImg = curImg.clone();
	  }
	  if (BGImage.empty()) {
		  preImg.copyTo(BGImage);
		  accmImage = BGImage; // accumulate the image for background generation
		  road_mask = cv::Mat::zeros(BGImage.size(), BGImage.type());
		  for (int ir = 0; ir<_config->Road_ROI_Pts.size(); ir++)
			  fillConvexPoly(road_mask, _config->Road_ROI_Pts.at(ir).data(), _config->Road_ROI_Pts.at(ir).size(), Scalar(255, 255, 255), 8);

		  if (_config->debugShowImages && _config->debugShowImagesDetail) {
			  cv::Mat debugImg = road_mask.clone();
			  if (debugImg.channels() < 3)
				  cvtColor(debugImg, debugImg, CV_GRAY2BGR);
			  for (int i = 0; i < _config->Boundary_ROI_Pts.size(); i++)
				  line(debugImg, _config->Boundary_ROI_Pts.at(i% _config->Boundary_ROI_Pts.size()), _config->Boundary_ROI_Pts.at((i + 1) % _config->Boundary_ROI_Pts.size()), SCALAR_BLUE, 2);
			  imshow("road mask", debugImg);
			  waitKey(1);
		  }
	  }
	  // we have two blurred images, now process the two image 
	  cv::Mat imgDifference;
	  cv::Mat imgThresh;
	  if (_config->bgsubtype == itms::BgSubType::BGS_CNT) {
		  pBgSub->apply(curImg, imgDifference);
		  if (_config->debugShowImages && _config->debugShowImagesDetail) {
			  Mat bgImage = Mat::zeros(curImg.size(), curImg.type());
			  pBgSub->getBackgroundImage(bgImage);
			  cv::imshow("backgroundImage", bgImage);
			  cv::waitKey(1);
			  /*if (isWriteToFile && frameCount == 200) {
				  string filename = conf.VideoPath;
				  filename.append("_" + to_string(conf.scaleFactor) + "x.jpg");
				  cv::imwrite(filename, bgImage);
				  std::cout << " background image has been generated (!!)\n";
			  }*/
		  }
	  }
	  else {
		  cv::absdiff(preImg, curImg, imgDifference);
	  }
	  if (!road_mask.empty()) {
		  cv::bitwise_and(road_mask, imgDifference, imgDifference);
	  }

	  // if needs to adjust img_dif_th
	  if (_config->isAutoBrightness) {
		  //compute the roi brightness and then adjust the img_dif_th withe the past max_past_frames 
		  float roiMean = mean(curImg(brightnessRoi)/*currentGray roi*/)[0];
		  if (pastBrightnessLevels.size() >= _config->max_past_frames_autoBrightness) // the size of vector is max_past_frames
			pop_front(pastBrightnessLevels, pastBrightnessLevels.size() - _config->max_past_frames_autoBrightness + 1); // keep the number of max_past_frames
			//  pop_front(pastBrightnessLevels); // remove an elemnt from the front of 			
			  
		  pastBrightnessLevels.push_back(cvRound(roiMean));
		  // adj function for adjusting image difference thresholding
		  int newTh = weightFnc(pastBrightnessLevels);
		  _config->img_dif_th = newTh;
		  if (_config->debugGeneral &&_config->debugGeneralDetail) {
			  std::cout << "Brightness mean for Roi: " << roiMean << "\n";
			  int m = mean(pastBrightnessLevels)[0];
			  std::cout << "Brightness mean for Past Frames: " << m << "\n";
			  std::cout << "New Dif Th : " << newTh << "\n";
		  }

	  }
	  cv::threshold(imgDifference, imgThresh, _config->img_dif_th, 255.0, CV_THRESH_BINARY);

	  if (_config->debugShowImages && _config->debugShowImagesDetail) {
		  cv::imshow("imgThresh", imgThresh);
		  cv::waitKey(1);
	  }

	  for (unsigned int i = 0; i < 1; i++) {
		  if (_config->bgsubtype == BgSubType::BGS_CNT)
			  cv::erode(imgThresh, imgThresh, structuringElement3x3);
		  cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		  cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		  if (_config->bgsubtype == BgSubType::BGS_DIF)
			  cv::erode(imgThresh, imgThresh, structuringElement5x5);
	  }

	  cv::Mat imgThreshCopy = imgThresh.clone();

	  std::vector<std::vector<cv::Point> > contours;

	  cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	  if (_config->debugShowImages && _config->debugShowImagesDetail) {
		  drawAndShowContours(imgThresh.size(), contours, "imgContours");
	  }

	  std::vector<std::vector<cv::Point> > convexHulls(contours.size());

	  for (unsigned int i = 0; i < contours.size(); i++) {
		  cv::convexHull(contours[i], convexHulls[i]);
	  }

	  if (_config->debugShowImages && _config->debugShowImagesDetail) {
		  drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");
	  }

	  std::vector<Blob> currentFrameBlobs;

	  for (auto &convexHull : convexHulls) {
		  Blob possibleBlob(convexHull);

		  if (possibleBlob.currentBoundingRect.area() > 10 &&
			  possibleBlob.dblCurrentAspectRatio > 0.2 &&
			  possibleBlob.dblCurrentAspectRatio < 6.0 &&
			  possibleBlob.currentBoundingRect.width > 3 &&
			  possibleBlob.currentBoundingRect.height > 3 &&
			  possibleBlob.dblCurrentDiagonalSize > 3.0 &&
			  (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
			  //  new approach according to 
			  // 1. distance, 2. correlation within certain range
			  std::vector<cv::Point2f> blob_ntPts;
			  blob_ntPts.push_back(Point2f(possibleBlob.centerPositions.back()));			  
			  cv::Rect roi_rect = possibleBlob.currentBoundingRect;
			  float blobncc = 0;			  
			  // bg image
			  // currnt image
			  // blob correlation
			  blobncc = getNCC(*_config, BGImage(roi_rect), curImg(roi_rect), Mat(), _config->match_method, _config->use_mask); 
			  // backgrdoun image need to be updated periodically 
			  // option double d3 = matchShapes(BGImage(roi_rect), imgFrame2Copy(roi_rect), CONTOURS_MATCH_I3, 0);
			  if (blobncc <= abs(_config->BlobNCC_Th)  
				  && checkIfPointInBoundary(*_config, blob_ntPts.back(), _config->Boundary_ROI_Pts)
				  /*realDistance >= 100 && realDistance <= 19900*//* distance constraint */)
			  {// check the correlation with bgground, object detection/classification
				  float realDistance = getDistanceInMeterFromPixels(blob_ntPts, _config->transmtxH, _config->lane_length, false);
				  if (_config->debugGeneral && _config->debugGeneralDetail) {
					  cout << "Candidate object:" << blob_ntPts.back() << "(W,H)" << cv::Size(roi_rect.width, roi_rect.height) << " is in(" << to_string(realDistance / 100.) << ") Meters ~(**)\n";
				  }
				  ObjectClass objclass;
				  float classProb = 0.f;
				  classifyObjectWithDistanceRatio(*_config, possibleBlob, realDistance / 100, objclass, classProb);
				  // update the blob info and add to the existing blobs according to the classifyObjectWithDistanceRatio function output
				  // verify the object with cascade object detection
				  if (classProb > 0.79 /* 1.0 */) {					  
					  currentFrameBlobs.push_back(possibleBlob);
				  }
				  else if (classProb>0.5f) {

					  // check with a ML-based approach
					  //float scaleRect = 1.5;
					  //Rect expRect = expandRect(roi_rect, scaleRect*roi_rect.width, scaleRect*roi_rect.height, imgFrame2Copy.cols, imgFrame2Copy.rows);

					  //if (possibleBlob.oc == itms::ObjectClass::OC_VEHICLE) {
					  //	// verify it
					  //	std::vector<cv::Rect> cars;
					  //	detectCascadeRoiVehicle(imgFrame2Copy, expRect, cars);
					  //	if (cars.size())
					  //		possibleBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
					  //	//else													// commented  :  put the all candidates commented out : it does not put the object in the candidates
					  //	//	continue;
					  //}
					  //else if (possibleBlob.oc == itms::ObjectClass::OC_HUMAN) {
					  //	// verify it
					  //	std::vector<cv::Rect> people;
					  //	detectCascadeRoiHuman(imgFrame2Copy, expRect, people);
					  //	if (people.size())
					  //		possibleBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
					  //	//else
					  //	//	continue;
					  //}
					  //else {// should not com in this loop (OC_OTHER)
					  //	int kkk = 0;
					  //}											  
					  currentFrameBlobs.push_back(possibleBlob);
				  }

			  }
		  }
	  }

	  if (_config->debugShowImages && _config->debugShowImagesDetail) {
		  // all of the currentFrameBlobs at this stage have 1 visible count yet. 
		  drawAndShowContours(*_config, imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");
	  }
	  // merge assuming
	  // blobs are in the ROI because of ROI map
	  // 남북 이동시는 가로가 세로보다 커야 한다.
	  //     
	  mergeBlobsInCurrentFrameBlobs(*_config, currentFrameBlobs);			// need to consider the distance
	  if (m_collectPoints) {
		  if (_config->debugShowImages && _config->debugShowImagesDetail) {
			  drawAndShowContours(*_config, imgThresh.size(), currentFrameBlobs, "before merging predictedBlobs into currentFrameBlobs");
			  waitKey(1);
		  }
		  //collectPointsInBlobs(currentFrameBlobs, m_collectPoints);	// collecting points in all blobs for local tracking , please check this out
		  //collectPointsInBlobs(blobs, m_collectPoints);	// collecting points in all blobs for local tracking 
		  // if local tracking, then check the local movements of the existing blobs and merge with the current 
		  // merging blobs with frame difference-based blobs.
		  // 0. get the blob prediction
		  // 1. draw for verification
		  // 2. merge them into currentFrameBlbos
		  std::vector<itms::Blob> predictedBlobs;
		  predictBlobs(*_config, blobs/* existing blbos */, preImg.getUMat(cv::ACCESS_READ)/* prevFrame */, curImg.getUMat(cv::ACCESS_READ)/* curFrame */, predictedBlobs);
		  if (predictedBlobs.size()) {
			  /*	imshow("imgFrame1CopyGray", imgFrame1Copy);
			  imshow("imgFrame2CopyGray", imgFrame2Copy);
			  Mat temp= Mat::zeros(imgFrame1Copy.size(), imgFrame1Copy.type());
			  absdiff(imgFrame1Copy, imgFrame2Copy, temp);
			  threshold(temp, temp, 10, 255,THRESH_BINARY);
			  imshow("diffFrame", temp);*/
			  /*for(auto prdBlob:predictedBlobs)
			  currentFrameBlobs.push_back(prdBlob);*/
			  mergeBlobsInCurrentFrameBlobsWithPredictedBlobs(currentFrameBlobs, predictedBlobs);
		  }
	  }
	  if (_config->debugShowImages && _config->debugShowImagesDetail) {
		  drawAndShowContours(*_config, imgThresh.size(), currentFrameBlobs, "after merging currentFrameBlobs");
		  waitKey(1);
	  }
	  if (blnFirstFrame == true) {
		  for (auto &currentFrameBlob : currentFrameBlobs) {
			  blobs.push_back(currentFrameBlob);
		  }
	  }
	  else {
		  _config->trackid = _config->trackid % _config->maxTrackIds;
		  matchCurrentFrameBlobsToExistingBlobs(*_config, preImg/* imgFrame1 */, curImg/* imgFrame2 */, blobs, currentFrameBlobs, _config->trackid);
	  }
	  //imgFrame2Copy = imgFrame2.clone();          // color get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

	  bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheBoundary(*_config, blobs,/* debugImg,*/ _config->ldirection, _config->Boundary_ROI_Pts);

	  if (_config->debugShowImages) {
		  cv::Mat debugImg = curImg.clone();
		  if (debugImg.channels() < 3)
			  cvtColor(debugImg, debugImg, cv::COLOR_GRAY2BGR);
		  if (_config->debugShowImagesDetail)
			  drawAndShowContours(*_config, imgThresh.size(), blobs, "All imgBlobs");

		  drawBlobInfoOnImage(*_config, blobs, debugImg);  // blob(tracked) information
		  drawRoadRoiOnImage(_config->Road_ROI_Pts, debugImg);		  
		  
		  std::vector<cv::Point> crossingLine;
		  crossingLine.push_back(cv::Point(0, _config->Boundary_ROI_Pts.at(3).y));
		  crossingLine.push_back(cv::Point(curImg.cols - 1, _config->Boundary_ROI_Pts.at(2).y));

		  if (blnAtLeastOneBlobCrossedTheLine == true) {
			  cv::line(debugImg, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
			  mCarCount = (mCarCount++)%maxCarCount;
		  }
		  else {
			  cv::line(debugImg, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
		  }		  
		  drawCarCountOnImage(mCarCount, debugImg);
		  cv::imshow("current Image", debugImg);
		  cv::waitKey(1);
	  }

	  // now we prepare for the next iteration
	  currentFrameBlobs.clear();

	  preImg = curImg.clone();           // move frame 1 up to where frame 2 is	  
	  // generate background image
	  if(_config->bGenerateBG && _config->intNumBGRefresh > 0){
		  
		  double learningRate = 1./_config->intNumBGRefresh;
		  //pBgSub->apply(curImg, tmp, learningRate); // slower than the below method
		  //pBgSub->getBackgroundImage(BGImage);
		  //cv::Mat tmp;//(curImg.size(), CV_32FC1);
		  accmImage.convertTo(accmImage, CV_32FC1);
		  accumulateWeighted(curImg, accmImage, learningRate, road_mask);
		  //addWeighted(curImg, learningRate, accmImage, 1.-learningRate, 0, accmImage);
		  convertScaleAbs(accmImage, accmImage);
		  setBGImage(accmImage);
		  if(_config->debugShowImagesDetail){
			  cv::imshow("generating BG", getBGImage());
			  waitKey(1);
		  }
	  }

	  // end generate backgroudimage 
	  blnFirstFrame = false;
	  if(checkObjectStatus(*_config, curImg, blobs, _itmsRes)){
		  if(_config->debugGeneralDetail)
			  cout<< "# of events: " << _itmsRes.objRect.size() <<" has been occurred (!)<!>" << endl;
	  }	  
  }// end process


} // itms namespace