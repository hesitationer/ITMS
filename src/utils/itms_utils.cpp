// please check the correct file location

#include "./utils/itms_utils.h"

#ifndef ITMS_DLL_EXPORT
// sangkny itms
#ifdef WIN32
#define ITMS_DLL_EXPORT __declspec(dllexport)
#else
#define ITMS_DLL_EXPORT 
#endif
#endif


namespace itms {
	
  void imshowBeforeAndAfter(cv::Mat &before_, cv::Mat &after_, std::string windowtitle, int gabbetweenimages)
  {
	  cv::Mat _before = before_.clone(), _after = after_.clone();
	  cv::Mat before, after;
	  if (_before.size() != _after.size() || (_before.type()!= _after.type())) {		  
			if(_before.type() != _after.type())
			{
				_after.convertTo(_after, _before.type());
			}
			float maxwidth = max(_before.cols, _after.cols);
			float maxheight = max(_before.rows, _after.rows);
			before = cv::Mat::zeros(cv::Size(maxwidth, maxheight), before_.type());
			after = before.clone();			
			_before.copyTo(before(cv::Rect(0,0,_before.cols, _before.rows)), _before);
			_after.copyTo(after(cv::Rect(0, 0, _after.cols, _after.rows)), _after);
	  }
	  else {
		  before = before_;
		  after = after_;		  
	  }
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
	cv::waitKey(1);
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
		  cv::Point2f sPt = cvtPx2RealPx(static_cast<cv::Point2f>(currentBlob->centerPositions.back()), _conf.transmtxH); // starting pt

		  for (unsigned int i = 0; i < currentFrameBlobs.size(); i++) {
			  cv::Point2f ePt = cvtPx2RealPx(static_cast<cv::Point2f>(currentFrameBlobs[i].centerPositions.back()), _conf.transmtxH);                                  // end pt
			  double realdist = distanceBetweenPoints(sPt, ePt); // real distance (cm) in the ROI 

			  double dblDistance = distanceBetweenPoints(currentBlob->centerPositions.back(), currentFrameBlobs[i].centerPositions.back());

			  if (realdist < 500/* 5 meters */ && dblDistance > 1/* same object */ && dblDistance < dblLeastDistance) { // center locations should be in the range
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
		  float realThreshold = 1.25; // according to real coordinate
		  if (dblLeastDistance < ((float)currentBlob->dblCurrentDiagonalSize)*realThreshold/*should be car size */) {
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

  // put the split logic as well on 2019
  void SplitBlobsInCurrentFrameBlobs(itms::Config& _conf, std::vector<Blob> &currentFrameBlobs, const cv::Mat & srcImg) {	  
	  // split a blob into two
	  std::vector<Blob>::iterator currentBlob = currentFrameBlobs.begin();
      size_t blobidx = 0;
	  while (currentBlob != currentFrameBlobs.end()) {		            
		  // condition, distance around top boundary and height/width ratio > 1 and cars.size()>1
		  if (currentBlob->oc == itms::ObjectClass::OC_VEHICLE && currentBlob->dblCurrentAspectRatio <=1.0) { // width/height
			  std::vector<cv::Point2f> blob_ntPts;
			  blob_ntPts.push_back(Point2f(currentBlob->centerPositions.back()));			  
			  float realDistance = getDistanceInMeterFromPixels(blob_ntPts, _conf.transmtxH, _conf.lane_length, false);
			  if (realDistance / 100 > 100) { // 100 meters 위에서만 한다 경계를 이미 통화해서 왔으니...
				  // vehicle detection
				  float scaleRect = 1.5;										// put it to the config parameters
				  cv::Rect roi_rect(currentBlob->currentBoundingRect);	// copy the current Rect and expand it
				  cv::Rect expRect = expandRect(roi_rect, scaleRect*roi_rect.width, scaleRect*roi_rect.height, srcImg.cols, srcImg.rows);
				  // verify it
					std::vector<cv::Rect> cars, allcars;
				    detectCascadeRoiVehicle(_conf, srcImg, expRect, cars, allcars);
					if (cars.size()>0 && allcars.size()>1) {// come this loop and see  // 일단 무조건 반으로 나눈다.
                        cv::Rect r = roi_rect;
                        itms::LineSegment ls(cv::Point(expRect.x, expRect.y + expRect.height / 2), cv::Point(expRect.x + expRect.width, expRect.y + expRect.height / 2)); // rect 중심에 라인을 긋고

                        std::vector<cv::Point> topContour, bottomContour; // 상단/하단 경계선으로 구분
                        std::vector<cv::Point> _contourPts= currentBlob->currentContour;                        
                                                                       
                        const int yStep = 1; // 간격 조절 step = 2로 하면 작아 질 수 있다. 
                        const int xStep = 1;  
                        // contour 내의 한점 Point pt(x,y)를 구하여 ls (직선) 상하로 구분하여 담는다. (convexhull 처리된 contour 로 인해) 채워진 다각형이 구해짐 
                        for (int y = r.y; y < r.y + r.height; y += yStep)
                        {
                            cv::Point2f pt(0, static_cast<float>(y));
                            for (int x = r.x; x < r.x + r.width; x += xStep)
                            {
                                pt.x = static_cast<float>(x);
                                if (cv::pointPolygonTest(_contourPts, pt, false) >= 0)
                                {
                                    if (ls.isPointBelowLine(pt)) {
                                        bottomContour.push_back(pt);
                                    }
                                    else {
                                        topContour.push_back(pt);
                                    }
                                    
                                }
                            }
                        }
                        if (0&&_conf.debugShowImagesDetail) {
                            std::vector<std::vector<cv::Point>> _contours;
                            _contours.push_back(topContour);
                            drawAndShowContours(srcImg.size(), _contours, "top contour in split blob", SCALAR_YELLOW);
                            _contours.clear();
                            _contours.push_back(bottomContour);
                            drawAndShowContours(srcImg.size(), _contours, "bottom contour in split blob", SCALAR_MAGENTA);                          
                        }
                        // insert two blobs, one for original , one for new one                        
                        itms::Blob _blob(bottomContour);
                                               
                        *currentBlob = _blob;     // 이곳 다음 에 blobidx++ 해도 된다. 그러면 되에서는 blobidx++로 마무리

                        itms::Blob _blob1(topContour);                        
                        currentBlob = currentFrameBlobs.insert(currentFrameBlobs.begin() + blobidx , _blob1); 
                        // 언제나 분리하기전 blob 앞으로 들어간다. +1은 뒤에 들어가고 횟수는 한번 줄인다. 하지만, 검출이 뒤에 것부터 먼저 되다가 바뀌게 되어 보기 좋지 않다.                        
                        blobidx += 2; // 1 for current, 1 for added one                        
                        if ( _conf.debugShowImagesDetail) {
                            std::vector<std::vector<cv::Point>> _contours;
                            _contours.push_back(_blob.currentContour);
                            drawAndShowContours(srcImg.size(), _contours, "bottom contour in split blob and assign", SCALAR_YELLOW);
                            _contours.clear();
                            _contours.push_back(_blob1.currentContour);
                            drawAndShowContours(srcImg.size(), _contours, "top contour in split blob and assign", SCALAR_MAGENTA);
                        }
                        continue;
                    } // end if (cars.size()
			  } // if(Distance/100 > 100) 			  
		  } // if(currentBlob->oc) loop
		  ++currentBlob;
          blobidx++;
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
  // 2019. 04. 05 -> Apply fastDSST at the searching stage for matching blobs, otherwise it can cross the different objects
  // 2019. 04. 08 -> fastDSST-based Approach => matchExistingBlobsToCurrentFrameBlobs 
  // 2019. 04. 18 -> back to the 04. 05 
  // 2019. 04. 28 , 29 ->  it is best for oc to be updated at every some time gaps otherwise, it can be corrupted under dynamic conditions 

  void matchCurrentFrameBlobsToExistingBlobs(itms::Config& _conf, const cv::Mat& orgImg, cv::Mat& preImg, const cv::Mat& srcImg, std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, int &id) {
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
						  cout << "Blob #: " << i << "was erased form the existingBlobs !! <<nmsboxes>>" << endl;
					  }
				  }
				  i++;  // keep increase the original order index
			  }
		  }
	  } // end of if (existingBlobs.size() > 1) {		// object # > 1

	  // --------------------------------- eliminate overlapped area --------------------------------------------
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

		// --------------------------------------------------------------------------------------------------------------------
		// ---------------------------------------- candidate search part -------------------------------------------------------
		// candidate search only with distances between centers of currentFrameBlobs and existing blobs.
		// add more property including area and h/w ratio
		// search around the nearest neighbor blob for tracking 
		// for searching larger area with more accuracy, we need to increase the search range (CurrentDiagonalSize) or to particle filter
		// with data, kalman or other tracking will be more accurate
	  // 2019. 04. 05, option: applying tracker-based matching option to match blobs
	  // 2019. 04. 29, 
	  // 2019. 05. 23 먼거리 자동차 분리 하기 
	  // --------------------------------------------------------------------------------------------------------------------------------
	  for (auto &currentFrameBlob : currentFrameBlobs) {
		  int intIndexOfLeastDistance = -1;
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
			  // 사실 this for loop 로 잘 못 되었다. 최종 out for 밖으로 빠져야 한다. 그렇게 하려면 for 들을 switch 하거나 assignment 로 할 당 후 이후에 처리해야 한다. 
			  // 2019. 04. 29
			  if (existingBlobs[i].blnStillBeingTracked == true) { // find assigned tracks
			  // it can be replaced with the tracking algorithm or assignment algorithm like KALMAN or Hungrian Assignment algorithm 
				  double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);
				  double dblDistance1 = distanceBetweenBlobs(currentFrameBlob, existingBlobs[i]); // sangkny 2019/04/28
				  /* // sangkny 2019. 02. 15
				  totalScore -= dblDistance;
				  totalScore -= (existingBlobs[i].currentBoundingRect.height < minDiagonal || existingBlobs[i].currentBoundingRect.height>maxDiagonal) ? 10 : 0;
				  totalScore -= (existingBlobs[i].currentBoundingRect.width < minArea || existingBlobs[i].currentBoundingRect.width > MaxArea) ? 10 : 0;
				  totalScore -= (abs(existingBlobs[i].currentBoundingRect.area() - currentFrameBlob.currentBoundingRect.area()) / max(existingBlobs[i].currentBoundingRect.width, currentFrameBlob.currentBoundingRect.width));
				  */				  
				  if (dblDistance1 < dblLeastDistance) { // 가장 좋은 방법은 object class를 일정 주기마다 조건에 따라 업데이트 하는 것이다. 2019. 04. 29  
					  dblLeastDistance = dblDistance1;
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

		  // ----------------- Add blobs ------------------------------------------------------------------
		  bool buseTrackerMatchingFlag = true;
		  if (buseTrackerMatchingFlag && dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize && intIndexOfLeastDistance != -1) { // 충분히 클수록 좋다. // 그리고, option 선택... 
			  addBlobToExistingBlobs(_conf, currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
			  
			  // lost object detection 
			  //				assignment.at(intIndexOfLeastDistance) = 1;
		  }
		  else { // add new blob: this routine contains new and unassigned track(blob)s				 
			  vector<Point2f> blobCenterPxs;
			  blobCenterPxs.push_back(currentFrameBlob.centerPositions.back());
			  float distance = getDistanceInMeterFromPixels(blobCenterPxs, _conf.transmtxH, _conf.lane_length, false);
			  if (0 && _conf.debugGeneralDetail)
				  cout << " distance: " << distance / 100 << " meters from the starting point.\n";
			  // do the inside
			  if (distance >= _conf.min_obj_distance /* 1m */ && distance < _conf.max_obj_distance/*(_conf.lane_length -100)*//*200m*/) {// between 1 meter and 200 meters, overlapped
				  ObjectClass objclass;
				  float classProb = 0.f;
				  classifyObjectWithDistanceRatio(_conf, currentFrameBlob, distance / 100, objclass, classProb); // 이미 했기에 다시 안하는 방법을 강구해야 함.
				  // update the blob info and add to the existing blobs according to the classifyObjectWithDistanceRatio function output
				  // verify the object with cascade object detection

				  if (classProb>0.5f) {
					  // check with a ML-based approach
					  float scaleRect = 1.5;										// put it to the config parameters
					  cv::Rect roi_rect(currentFrameBlob.currentBoundingRect);	// copy the current Rect and expand it
					  cv::Rect expRect = expandRect(roi_rect, scaleRect*roi_rect.width, scaleRect*roi_rect.height, srcImg.cols, srcImg.rows);
					  if (currentFrameBlob.oc == itms::ObjectClass::OC_VEHICLE) {
						  // verify it
						  std::vector<cv::Rect> cars, allcars;
						  detectCascadeRoiVehicle(_conf, srcImg, expRect, cars, allcars);
						  if (cars.size())
							  currentFrameBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
						  //else if (distance >= 10000 && _conf.scaleFactor < 1.0) {       // sangkny 2019. 04. 28
							 // //한번 더 원래 이미지로 차인지는 별로 의미가 없어 일단 comment ..							  
							 // cv::Rect _roi_rect = currentFrameBlob.currentBoundingRect; //상대좌표가 들어와야 한다.
							 // _roi_rect.x = (float)currentFrameBlob.currentBoundingRect.x / _conf.scaleFactor;
							 // _roi_rect.y = (float)currentFrameBlob.currentBoundingRect.y / _conf.scaleFactor;
							 // _roi_rect.width = (float)currentFrameBlob.currentBoundingRect.width / _conf.scaleFactor;
							 // _roi_rect.height = (float)currentFrameBlob.currentBoundingRect.height / _conf.scaleFactor;
							 // _roi_rect = expandRect(_roi_rect, 8, 8, orgImg.cols, orgImg.rows);

							 // detectCascadeRoiHuman(_conf, orgImg, _roi_rect, cars); // 
							 // if (cars.size()){
								//  currentFrameBlob.oc_prob = 1.0;
								//  currentFrameBlob.oc = itms::ObjectClass::OC_HUMAN; // 
							 // }
						  //}
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
						  else if (distance >= 100 && _conf.scaleFactor< 1.0) { // 1 미터이고 축소되었으면... 100 미터 이상부터 하려는 것 아니었나? 넘 멀다 50 미터 부터는 해야 함.
								  //한번 더 원래 이미지로
								  people.clear();
								  cv::Rect _roi_rect = currentFrameBlob.currentBoundingRect; //상대좌표가 들어와야 한다.
								  _roi_rect.x = (float)currentFrameBlob.currentBoundingRect.x / _conf.scaleFactor;
								  _roi_rect.y = (float)currentFrameBlob.currentBoundingRect.y / _conf.scaleFactor;
								  _roi_rect.width = (float)currentFrameBlob.currentBoundingRect.width / _conf.scaleFactor;
								  _roi_rect.height = (float)currentFrameBlob.currentBoundingRect.height / _conf.scaleFactor;
								  _roi_rect = expandRect(_roi_rect, 8, 8, orgImg.cols, orgImg.rows);

								  detectCascadeRoiHuman(_conf, orgImg, _roi_rect, people); // sangkny 20190331 check the human with original sized image 
								  if (people.size())
									currentFrameBlob.oc_prob = 1.0;								  							  
						  
						  }
						  else if (_conf.img_dif_th >= _conf.nightBrightness_Th &&/* at night */ classProb >= _conf.nightObjectProb_Th) 
							  // at daytime, need to classify using cascade classifier, otherwise, at night, we put the candidate
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
	  for (auto &existingBlob : existingBlobs) { // update track routine and prediction blob information

		  if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) { // unassigned tracks            
			  existingBlob.age++;

			  vector<Point2f> blobCenterPxs;
			  blobCenterPxs.push_back(existingBlob.centerPositions.back());
			  float startDist = _conf.sTrackingDist;

			  if (_conf.m_externalTrackerForLost && getDistanceInMeterFromPixels(blobCenterPxs, _conf.transmtxH, _conf.lane_length, false)> startDist /* && (assignment[blobIdx] == -1)*/) { // 상관이 없네...
				  existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;// temporal line
																		// reinitialize the fastDSST with prevFrame and update the fastDSST with current Frame, finally check its robustness with template matching or other method				  
				  int expandY = 2;
				  bool successTracking = false;
				  if (!_conf.isSubImgTracking) { // Global/full image-based approach				  					  
					  // as of 2019. 01. 18, just put the new center points except for countour information	
					  // 2019/04/07 function 
					  cv::Rect newRoi;
					  
					  successTracking = itms::trackNewLocationFromPrevBlob(_conf, preImg, srcImg, existingBlob, newRoi, expandY);
					  if (successTracking) {
						  // move a boundary and its boundingRect if necessary
						  if (existingBlob.resetBlobContourWithCenter(cv::Point(cvRound(newRoi.x + newRoi.width / 2.f), cvRound(newRoi.y + newRoi.height / 2.f)))) { // check if the center points will move or not
							  cv::Rect tmpRect = existingBlob.currentBoundingRect;
							  existingBlob.currentBoundingRect.x -= (tmpRect.x + tmpRect.width / 2.f - (newRoi.x + newRoi.width / 2.f));  // move to the newRoi center with keep the size of Boundary
							  //existingBlob.currentBoundingRect.width = newRoi.width;													  // sangkny 20190404
							  Clamp(existingBlob.currentBoundingRect.x, existingBlob.currentBoundingRect.width, srcImg.cols);
							  existingBlob.currentBoundingRect.y -= (tmpRect.y + tmpRect.height / 2.f - (newRoi.y + newRoi.height / 2.f));
							  //existingBlob.currentBoundingRect.height = newRoi.height;														// sangkny 20190404
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
					  
				  } // if(!_conf.isSubImgTracking) ends
				  else { // sub image-based approach for lost object detection, in this case, init and update need to be carried out at ontime 						
						// sangkny update the center points of the existing blob which has been untracted	
                        cv::Rect prect;//=existingBlob.currentBoundingRect;

						successTracking = itms::trackNewLocationFromPrevBlob(_conf, preImg, srcImg, existingBlob, prect, expandY);
						  
						  if (successTracking && existingBlob.resetBlobContourWithCenter(cv::Point(cvRound(prect.x + prect.width / 2.f), cvRound(prect.y + prect.height / 2.f)))){ // boundary movement first 							  
							  cv::Rect tmpRect = existingBlob.currentBoundingRect;
							  existingBlob.currentBoundingRect.x -= (tmpRect.x + tmpRect.width / 2.f - (prect.x + prect.width / 2.f));  // move to the newRoi center with keep the size of Boundary
							  //existingBlob.currentBoundingRect.width = prect.width;														// sangkny 20190404
							  Clamp(existingBlob.currentBoundingRect.x, existingBlob.currentBoundingRect.width, srcImg.cols);
							  existingBlob.currentBoundingRect.y -= (tmpRect.y + tmpRect.height / 2.f - (prect.y + prect.height / 2.f));
							  //existingBlob.currentBoundingRect.height = prect.height;													// sangkny 20190404
							  Clamp(existingBlob.currentBoundingRect.y, existingBlob.currentBoundingRect.height, srcImg.rows);
						  }
						  existingBlob.centerPositions.push_back(cv::Point(cvRound(prect.x + prect.width / 2.f), cvRound(prect.y + prect.height / 2.f)));
						
						  if (existingBlob.centerPositions.size() >_conf.max_Center_Pts)
							  pop_front(existingBlob.centerPositions, (existingBlob.centerPositions.size() - _conf.max_Center_Pts));
						  existingBlob.predictNextPosition();
				  }// else part ends : sub image-based approach

			  } // _cof.m_exernalTrackerLost
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
			  existingBlob.predictNextPosition();
		  }

		  if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= _conf.maxNumOfConsecutiveInFramesWithoutAMatch/* 1sec. it should be a predefined threshold */) {
			  existingBlob.blnStillBeingTracked = false; /* still in the list of blobs */
		  }
		  // object status, class update routine starts        
		  //existingBlob.os = getObjectStatusFromBlobCenters(existingBlob, _conf.ldirection, _conf.movingThresholdInPixels, _conf.minVisibleCount);
		  // -------------------------------------------------------------------------------------------------------  //
		  // --------- 2019. 04. 29 ---- object status 가 바뀌면 다시 한번 object class를 update 한다. 사람인 경우, 고려 필요..
		  itms::ObjectStatus preFos = existingBlob.fos;
		  existingBlob.os = getObjStatusUsingLinearRegression(_conf, existingBlob, _conf.ldirection, _conf.movingThresholdInPixels, _conf.minVisibleCount);
		  // 벡터로 넣을지 생각해 볼 것, 그리고, regression from kalman 으로 부터 정지 등을 판단하는 것도 고려 중....		  
		  itms::ObjectStatus finalFos = existingBlob.fos;		  // if blob fos has been changed, then object class needs to be updated as well
		  // ------------- sangkny 2019. 04. 29 -- OC Updated ----------------------------
		  if (preFos != finalFos) {
			  // object class detection
			  /*if (finalFos == ObjectStatus::OS_NOTDETERMINED) {

			  }*/
			  if (existingBlob.oc == itms::ObjectClass::OC_VEHICLE && finalFos == ObjectStatus::OS_NOTDETERMINED) {
					vector<Point2f> blobCenterPxs;
					blobCenterPxs.push_back(existingBlob.centerPositions.back());
					float distance = getDistanceInMeterFromPixels(blobCenterPxs, _conf.transmtxH, _conf.lane_length, false);				  
					ObjectClass objclass;
					float classProb = 0.f;
					 classifyObjectWithDistanceRatio(_conf, existingBlob, distance / 100, objclass, classProb);

			  }
		  }
		  // ----------------------------------------------------------------------------
		  // object status, class update routine ends

		  blobIdx++;
	  }

  }

  void matchExistingBlobsToCurrentFrameBlobs(itms::Config& _conf, cv::Mat& preImg, const cv::Mat& srcImg, std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, int &id) {
	  // ltracker-based approach // 2019. 04. 08
	  // traker first and register a new blob later
	  
	  // sangkny 2019. 02. 11. NMS implementation for overlapped Blob
	  // 0. collect information from blobs
	  // 1. non maximum supression for Blobs
	  // 2. need to check the following suppression is required when we pursue tracker-based approach
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
			  value += (fmin(0.2, ((float)(existingBlobs.at(i).centerPositions.size() * 10) / (float)_conf.max_Center_Pts))); // max 0.2
																															  //value += (fmin(0.2, ((float)existingBlobs.at(i).totalVisibleCount / (float)existingBlobs.at(i).age))); // max 0.2
																															  // area
			  if (existingBlobs.at(i).currentBoundingRect.area() > 1200 * _conf.scaleFactor)  // max 0.3
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
						  cout << "Blob #: " << i << "was erased form the existingBlobs !! <<nmsboxes>>" << endl;
					  }
				  }
				  i++;  // keep increase the original order index
			  }
		  }
	  } // end of if (existingBlobs.size() > 1) {		// object # > 1

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
		// --------------------------------- eliminate overlapped area --------------------------------------------
		// blob iterator
	  std::vector<Blob>::iterator existBlob = existingBlobs.begin();
	  while (existBlob != existingBlobs.end()) {
		  // check if a block is too old after disappeared in the screen		  
		  if (existBlob->blnStillBeingTracked == false
			  && existBlob->intNumOfConsecutiveFramesWithoutAMatch >= _conf.maxNumOfConsecutiveInvisibleCounts) {
			  // removing a blob from the list of existingBlobs	when it has been deserted long time or overlapped with the current active object					
			  // overlapping test if overlapped or background, we will erase.
			  std::vector<pair<int, int>> overlappedBobPair; // first:blob index, second:overlapped type
			  
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
					  continue; // this can be removed
								// ---------------------------------------------------------------
				  }
				  else {
					  existBlob = existingBlobs.erase(existBlob);
				  }
			  }
		  }
		  else { // if (existBlob->blnStillBeingTracked == true ) 
			  // tracking algorithm should be inserted here
			  // 
			  //existBlob->blnCurrentMatchFoundOrNewBlob = false;
			  //existBlob->predictNextPosition();
			  cv::Rect newRoi;
			  bool bSuccess = false;
			  bSuccess = trackNewLocation(_conf, preImg, srcImg, *existBlob, newRoi, 2);		
			  if (bSuccess ) {
				  existBlob->blnCurrentMatchFoundOrNewBlob = true;
				  existBlob->blnStillBeingTracked = true;
				  // update the center				  
				  //if (existBlob->resetBlobContourWithCenter(cv::Point(cvRound(newRoi.x + newRoi.width / 2.f), cvRound(newRoi.y + newRoi.height / 2.f)))) {
					  existBlob->currentBoundingRect = newRoi;					  
					  Clamp(existBlob->currentBoundingRect.x, existBlob->currentBoundingRect.width, srcImg.cols);					  
					  Clamp(existBlob->currentBoundingRect.y, existBlob->currentBoundingRect.height, srcImg.rows);
					  //existBlob->centerPositions.push_back(cv::Point(cvRound(newRoi.x + newRoi.width / 2.f), cvRound(newRoi.y + newRoi.height / 2.f)));
					  existBlob->centerPositions.push_back(cv::Point(cvRound(existBlob->currentBoundingRect.x + existBlob->currentBoundingRect.width / 2.f), cvRound(existBlob->currentBoundingRect.y + existBlob->currentBoundingRect.height / 2.f)));
				  //}				  
			  }
			  else {  // tracking fail or centerposition is same as before
				  existBlob->blnCurrentMatchFoundOrNewBlob = false;				  
				  //existBlob->centerPositions.push_back(existBlob->centerPositions.back()); // put one center point for lost object // 뒤에서 한다. 
				  
			  }		
			  if (existBlob->centerPositions.size() > _conf.max_Center_Pts)
				  pop_front(existBlob->centerPositions, existBlob->centerPositions.size() - _conf.max_Center_Pts);
			  existBlob->predictNextPosition(); // common both the above condition

			++existBlob;
		  }
	  } // end while ( existingBlob != existingBlobs.end())
		/*for (auto &existingBlob : existingBlobs) {
		existingBlob.blnCurrentMatchFoundOrNewBlob = false;
		existingBlob.predictNextPosition();
		}*/

	  // eliminate the current blob which has been tracked from the previous mother blobs	  
	  int ii = 0;
	  for(unsigned int i = 0; i<existingBlobs.size(); i++) {
		  cv::Rect exBlob_rect = existingBlobs[i].currentBoundingRect;
		  std::vector<Blob>::iterator _curBlob = currentFrameBlobs.begin();
		  while (_curBlob !=currentFrameBlobs.end()) {
			  cv::Rect curBlob_rect = _curBlob->currentBoundingRect;
			  cv::Rect intRect = (curBlob_rect & exBlob_rect);
			  if (_conf.debugGeneral && _conf.debugGeneralDetail) {				  
				  itms::imshowBeforeAndAfter(srcImg(curBlob_rect), srcImg(exBlob_rect), "curBlob / existing Blob rect", 2);
				  std::cout << " Area ratio : intRect/exBlob_rect area ratio --> " << (float)intRect.area() / exBlob_rect.area() << endl;
				  //cv::waitKey(1);
			  }
			  float allowedPct = 0.5;// _conf.useTrackerAllowedPercentage;
			  float areaRatio;
			  areaRatio = (float)intRect.area() / (float)exBlob_rect.area();

			  if ((areaRatio > (1 - allowedPct)) && (areaRatio <= (1 + allowedPct))) { // 1/2 
				  // eliminate the current blob because it has been already tracked and updated using tracking-based approach
				  _curBlob = currentFrameBlobs.erase(_curBlob);
				  if (_conf.debugGeneral && _conf.debugGeneralDetail) {
					  cout << "\n cur Blob #: " << ii << " was erased form current Frame Blobs !! <<tracking-based>> -------------------->" << endl;
				  }				  
			  }
			  else
				  ++_curBlob;

		  }		  
		  ii++;
	  }// while end for (curBlob)
	  
	  // --------------------------------------------------------------------------------------------------------------------
	  // 
	  // --------------------------------------------------------------------------------------------------------------------------------
	  for (auto &currentFrameBlob : currentFrameBlobs) { // then add new all the rest of current frame blobs
		  int intIndexOfLeastDistance = -1;
		  //int intIndexOfHighestScore = 0;
		  double dblLeastDistance = 100000.0;

		  //for (unsigned int i = 0; i < existingBlobs.size(); i++) {
			 // if (existingBlobs[i].blnStillBeingTracked == true) { // find assigned tracks
				//												   // it can be replaced with the tracking algorithm or assignment algorithm like KALMAN or Hungrian Assignment algorithm 
				//  double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);
				//  
				//  if (existingBlobs[i].oc != currentFrameBlob.oc)
				//	  int kkk = 0;

				//  if (dblDistance < dblLeastDistance && (existingBlobs[i].oc == currentFrameBlob.oc || existingBlobs[i].oc == OC_OTHER || currentFrameBlob.oc == OC_OTHER)) {
				//	  dblLeastDistance = dblDistance;
				//	  intIndexOfLeastDistance = i;
				//  }
				//  
			 // }
			 // else { // existingBlobs[i].bInStillBeingTracked == false;
				//	 /* do something for unassinged tracks */
				//  int temp = 0; // no meaning 
			 // }
		  //}

		  // ----------------- matching with tracker ------------------------------
		  // -------------------------------------------------
		  bool buseTrackerMatchingFlag = true;
		 
		  
		  if (0 && buseTrackerMatchingFlag && dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize) { // 충분히 클수록 좋다. // 그리고, option 선택... 
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
			  if (distance >= 100.00/* 1m */ && distance < (_conf.lane_length - 100)/*200m*/) {// between 1 meter and 200 meters
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
						  std::vector<cv::Rect> cars, allcars;
						  detectCascadeRoiVehicle(_conf, srcImg, expRect, cars, allcars);
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


	  for (auto &existingBlob : existingBlobs) { // update track routine

		  if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) { // unassigned tracks            
			  existingBlob.age++;

			  if (0 && _conf.m_externalTrackerForLost /* && (assignment[blobIdx] == -1)*/) { // 상관이 없네...
				  existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;// temporal line
																		// reinitialize the fastDSST with prevFrame and update the fastDSST with current Frame, finally check its robustness with template matching or other method
																		/*cv::Rect newRoi, m_predictionRect;
																		int expandY = 2;
																		float heightRatio = (float)(existingBlob.currentBoundingRect.height + expandY) / (existingBlob.currentBoundingRect.height);
																		int expandX = max(0, cvRound((float)(existingBlob.currentBoundingRect.width)*heightRatio - existingBlob.currentBoundingRect.width));
																		cv::Rect expRect = expandRect(existingBlob.currentBoundingRect, expandX, expandY, preImg.cols, preImg.rows);

																		if (0 && _conf.debugGeneral && _conf.debugGeneralDetail)
																		cout << "From boundingRect: " << existingBlob.currentBoundingRect << " => To expectedRect: " << expRect << endl;*/
				  int expandY = 2;
				  bool successTracking = false;
				  if (!_conf.isSubImgTracking) { // Global/full image-based approach				  
												 
					  cv::Rect newRoi;

					  successTracking = itms::trackNewLocation(_conf, preImg, srcImg, existingBlob, newRoi, expandY);
					  if (successTracking) {
						  // move a boundary and its boundingRect if necessary

						  if (existingBlob.resetBlobContourWithCenter(cv::Point(cvRound(newRoi.x + newRoi.width / 2.f), cvRound(newRoi.y + newRoi.height / 2.f)))) { // check if the center points will move or not
							  cv::Rect tmpRect = existingBlob.currentBoundingRect;
							  existingBlob.currentBoundingRect.x -= (tmpRect.x + tmpRect.width / 2.f - (newRoi.x + newRoi.width / 2.f));  // move to the newRoi center with keep the size of Boundary
																																		  //existingBlob.currentBoundingRect.width = newRoi.width;													  // sangkny 20190404
							  Clamp(existingBlob.currentBoundingRect.x, existingBlob.currentBoundingRect.width, srcImg.cols);
							  existingBlob.currentBoundingRect.y -= (tmpRect.y + tmpRect.height / 2.f - (newRoi.y + newRoi.height / 2.f));
							  //existingBlob.currentBoundingRect.height = newRoi.height;														// sangkny 20190404
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
					  
				  } // if(!_conf.isSubImgTracking) ends
				  else { // sub image-based approach for lost object detection, in this case, init and update need to be carried out at ontime 
						 
						 // sangkny update the center points of the existing blob which has been untracted	
					  cv::Rect prect;//=existingBlob.currentBoundingRect;

					  successTracking = itms::trackNewLocation(_conf, preImg, srcImg, existingBlob, prect, expandY);

					  if (successTracking && existingBlob.resetBlobContourWithCenter(cv::Point(cvRound(prect.x + prect.width / 2.f), cvRound(prect.y + prect.height / 2.f)))) { // boundary movement first 							  
						  cv::Rect tmpRect = existingBlob.currentBoundingRect;
						  existingBlob.currentBoundingRect.x -= (tmpRect.x + tmpRect.width / 2.f - (prect.x + prect.width / 2.f));  // move to the newRoi center with keep the size of Boundary
																																	//existingBlob.currentBoundingRect.width = prect.width;														// sangkny 20190404
						  Clamp(existingBlob.currentBoundingRect.x, existingBlob.currentBoundingRect.width, srcImg.cols);
						  existingBlob.currentBoundingRect.y -= (tmpRect.y + tmpRect.height / 2.f - (prect.y + prect.height / 2.f));
						  //existingBlob.currentBoundingRect.height = prect.height;													// sangkny 20190404
						  Clamp(existingBlob.currentBoundingRect.y, existingBlob.currentBoundingRect.height, srcImg.rows);
					  }
					  existingBlob.centerPositions.push_back(cv::Point(cvRound(prect.x + prect.width / 2.f), cvRound(prect.y + prect.height / 2.f)));

					  if (existingBlob.centerPositions.size() >_conf.max_Center_Pts)
						  pop_front(existingBlob.centerPositions, (existingBlob.centerPositions.size() - _conf.max_Center_Pts));

					  existingBlob.predictNextPosition();

					  

				  }// else part ends : sub image-based approach

			  } // _cof.m_exernalTrackerLost
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
			  existingBlob.predictNextPosition();
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
			  else { // they have different its own classes and it is not ObjectClass::OC_OTHER
				  if(existingBlobs[intIndex].bNotifyMessage && existingBlobs[intIndex].oc_notified != existingBlobs[intIndex].oc){
					  existingBlobs[intIndex].bNotifyMessage = false;
					  existingBlobs[intIndex].oc_notified = itms::ObjectClass::OC_OTHER;
					  existingBlobs[intIndex].os_notified = itms::ObjectStatus::OS_NOTDETERMINED;
					  // 2019. 05. 19
					  //if(currentFrameBlob.oc == ObjectClass::OC_HUMAN){
						  existingBlobs[intIndex].age = 1;
						  existingBlobs[intIndex].totalVisibleCount = 1;
						  existingBlobs[intIndex].startPoint = currentFrameBlob.centerPositions.back();
					  //}
				  }
			  if(_conf.debugGeneralDetail){
				  std::cout << " --------- ************* Class Human & Vehicle **************** ----------- \n";
				  std::cout << " algorithm can not be here !!!\n";
				  std::cout << " actually, it happens in the long distance -->!! \n";
				  std::cout << " 2019/04/30, bNotifyMessage will be false if necessary \n";
				  std::cout << " --------- ************* --------------------- **************** ----------- \n";
				  // they can not be here because this case should have been refined in the above step MatchCurrentBlobsToExistingBlbos
				  // 현재는 후자 (currentFrameBlob 을 무시함)
				  }
			  }
		  }
		  else { // they have the same class 2019. 05. 20
			  if(existingBlobs[intIndex].bNotifyMessage 
			  &&  existingBlobs[intIndex].oc_notified != currentFrameBlob.oc
			  && (currentFrameBlob.oc == itms::ObjectClass::OC_HUMAN || currentFrameBlob.oc == itms::ObjectClass::OC_VEHICLE)
			  ){ // reset the status of notification and get ready to notice other object class such as human and vehicle			   
			   existingBlobs[intIndex].bNotifyMessage = false;
			   existingBlobs[intIndex].oc_notified = itms::ObjectClass::OC_OTHER;
			   existingBlobs[intIndex].os_notified = itms::ObjectStatus::OS_NOTDETERMINED;			   
			   
			   existingBlobs[intIndex].age = 1;
			   existingBlobs[intIndex].totalVisibleCount = 1;
			   existingBlobs[intIndex].startPoint = currentFrameBlob.centerPositions.back();
			   //}
			  }
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
		  if (abs(deltaY) <= movingThresholdInPixels || (blob.speed <= config.speedLimitForstopping))
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_SOUTH) ? (deltaY > 0 ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : (deltaY > 0 ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  case LD_EAST:
	  case LD_WEST:
		  if (abs(deltaX) <= movingThresholdInPixels || (blob.speed <= config.speedLimitForstopping)) // 
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_EAST) ? (deltaX > 0 ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : (deltaX > 0 ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  case LD_NORTHEAST:
	  case LD_SOUTHWEST:
		  if (abs(deltaX) + abs(deltaY) <= movingThresholdInPixels || (blob.speed <= config.speedLimitForstopping)) // 
			  objectstatus = OS_STOPPED;
		  else { // moving anyway
			  objectstatus = (lanedirection == LD_NORTHEAST) ? ((deltaX > 0 || deltaY < 0) ? OS_MOVING_FORWARD : OS_MOVING_BACKWARD) : ((deltaX > 0 || deltaY <0) ? OS_MOVING_BACKWARD : OS_MOVING_FORWARD);
		  }
		  break;

	  case LD_SOUTHEAST:
	  case LD_NORTHWEST:
		  if (abs(deltaX) + abs(deltaY) <= movingThresholdInPixels || (blob.speed <= config.speedLimitForstopping)) // 
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
  /////////////////////////////////////////-- Linear Regression-based Object Direction and Speed Computation --//////////////////////////////////////////////////////////
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
	  //itms::ObjectStatus preOS = blob.os;
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
  
  float angleBetweenPoints(Point p1, Point p2)
  {
	  int deltaY = p2.y - p1.y;
	  int deltaX = p2.x - p1.x;

	  return atan2((float)deltaY, (float)deltaX) * (180 / CV_PI);
  }

  double distanceBetweenBlobs(const itms::Blob& _blob1, const itms::Blob& _blob2) {
	  std::array<track_t, 4> diff;
	  cv::Point _bPt1=_blob1.centerPositions.back();
	  cv::Point _bPt2=_blob2.centerPositions.back();
	  diff[0] = _bPt1.x - _bPt2.x;
	  diff[1] = _bPt1.y - _bPt2.y;
	  diff[2] = static_cast<track_t>(_blob1.currentBoundingRect.width - _blob2.currentBoundingRect.width);
	  diff[3] = static_cast<track_t>(_blob1.currentBoundingRect.height - _blob2.currentBoundingRect.height);

	  track_t dist = 0;
	  for (size_t i = 0; i < diff.size(); ++i)
	  {
		  dist += diff[i] * diff[i];
	  }
	  return sqrtf(dist);

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
				  //////cv::waitKey(1);
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
			  
			  // I think the below code is not neccessary please check it again later
			  // check the validation of the blob which should be inside the boundary
			  std::vector<cv::Point2f> blob_tlPt,blob_brPt;
			  blob_tlPt.push_back(Point2f(blob->currentBoundingRect.tl()));
			  blob_brPt.push_back(Point2f(blob->currentBoundingRect.br()));			  
			  float tlDist = getDistanceInMeterFromPixels(blob_tlPt, _conf.transmtxH, _conf.lane_length, false); // need to modify when the driving direction changes
			  float brDist = getDistanceInMeterFromPixels(blob_brPt, _conf.transmtxH, _conf.lane_length, false);
			  // end validation check
			  if (tlDist >= _conf.max_obj_distance || brDist <= _conf.min_obj_distance) {
				  if (_conf.debugGeneralDetail)
					  cout << " a blob: " << blob->id << " is eliminated at boundary due to distance condition !! in checkIfBlobsCrossedTheBoundary\n\n\n\n";
				  blob = blobs.erase(blob);
				  continue;
			  }
			  // Cross line checking
			  int prevFrameIndex = (int)blob->centerPositions.size() - 2;
			  int currFrameIndex = (int)blob->centerPositions.size() - 1;

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
				  //////cv::waitKey(1);
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
					  ////cv::waitKey(1);
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
  
  // this function checks the blob top-left and bottom-right points are within the boundary with corrent distance and returns the real distance of the center of the blob
  bool checkIfBlobInBoundaryAndDistance(const itms::Config& _conf, const itms::Blob& _blob, const std::vector<cv::Point> &_tboundaryPts, float& _realDistance) {
	  std::vector<cv::Point2f> blob_tlPt, blob_brPt, blob_ctPt;
	  blob_tlPt.push_back(Point2f(_blob.currentBoundingRect.tl()));
	  blob_brPt.push_back(Point2f(_blob.currentBoundingRect.br()));
	  blob_ctPt.push_back(Point2f(_blob.centerPositions.back()));
	  float tlDist = getDistanceInMeterFromPixels(blob_tlPt, _conf.transmtxH, _conf.lane_length, false); // need to modify when the driving direction changes
	  float brDist = getDistanceInMeterFromPixels(blob_brPt, _conf.transmtxH, _conf.lane_length, false);
	  float ctDist = getDistanceInMeterFromPixels(blob_ctPt, _conf.transmtxH, _conf.lane_length, false);
	  float ctrDist = (tlDist+brDist)/2;
	  _realDistance = ctDist;
	  // end validation check
	  bool bInBound = (!(tlDist >= _conf.max_obj_distance || brDist <= _conf.min_obj_distance))&&(checkIfPointInBoundary(_conf, blob_ctPt.back(), _tboundaryPts));
	  if (!bInBound) {
		  if (_conf.debugGeneralDetail){
			  cout << " a blob: " << _blob.id << " is out of boundary !!\n\n\n\n";
			  }
		  
	  }
	  return bInBound;
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  void drawBlobInfoOnImage(itms::Config& _conf, std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {

	  for (unsigned int i = 0; i < blobs.size(); i++) {

		  if (blobs[i].blnStillBeingTracked == true && blobs[i].totalVisibleCount >1) {
			  int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
			  double dblFontScale = max(1., blobs[i].dblCurrentDiagonalSize / 60.0);
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

			  // DAY/NIGHT MODE 
			  infostr.clear(); 
			  infostr = "mode: ";
			  if(_conf.img_dif_th >= _conf.nightBrightness_Th)
				  infostr += "Night --> ";
			  else
				  infostr += "Day --> ";

			  infostr += std::to_string(_conf.img_dif_th);
			  cv::putText(imgFrame2Copy, infostr, cv::Point(10,50), intFontFace, 1 /* intFontScale */, SCALAR_BLUE, 2/* intFontThickness */);

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
	  if (1&& _conf.debugShowImages && _conf.debugShowImagesDetail) {
		  imshow("img", bgimg_gray);
		  imshow("template image", fgtempl_gray);
		  //cv::waitKey(1);
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

  // --------------- track blob function -----------------------------------------------------------------------------
  // This function tries to find new location (new_rect) based on the previous blob(existing, previous) (refBlob)    //
  // ------------------------------------------------------------------------------------------------------------------
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

  bool trackNewLocationFromPrevBlob(const itms::Config& _conf, const cv::Mat& _preImg, const cv::Mat& _srcImg, itms::Blob& _ref, cv::Rect& _new_rect, const int _expandY) {

	  bool bexternalTrackerForLost = _conf.m_externalTrackerForLost;  // does use an external tracker?
	  bool bisSubImgTracking = _conf.isSubImgTracking;                // is it subImgTracking (local tracker) ? 
	  bool success = false;

	  if(bexternalTrackerForLost/*_conf.m_externalTrackerForLost*/){
		  // main routine -- common part for global and local region-based tracker 
		  cv::Rect newRoi, m_predictionRect;
		  int expandY = _expandY;
		  float heightRatio = (float)(_ref.currentBoundingRect.height + expandY) / (float)(_ref.currentBoundingRect.height);
		  int expandX = max(0, cvRound((float)(_ref.currentBoundingRect.width)*heightRatio - _ref.currentBoundingRect.width));
		  cv::Rect expRect = expandRect(_ref.currentBoundingRect, expandX, expandY, _preImg.cols, _preImg.rows);

		  m_predictionRect = expRect;//existingBlob.currentBoundingRect;//expRect;			
		  
		  if (!_ref.m_tracker_psr || _ref.m_tracker_psr.empty()) {
			  _ref.CreateExternalTracker();
		  }
		  if (!bisSubImgTracking) { // global approach
			  cv::Mat preImg3, srcImg3;
			  if (_preImg.channels() < 3) {
				  cv::cvtColor(_preImg, preImg3, CV_GRAY2BGR);
				  cv::cvtColor(_srcImg, srcImg3, CV_GRAY2BGR);
			  }
			  newRoi = expRect;
			  if (!_ref.m_tracker_initialized) {		// do it only once																				  
				  success = (_preImg.channels() < 3) ? _ref.m_tracker_psr->reinit(preImg3, newRoi) : _ref.m_tracker_psr->reinit(_preImg, newRoi);
				  _ref.m_tracker_initialized = true;
			  }
			  else {
				  success = (_preImg.channels() < 3) ? _ref.m_tracker_psr->updateAt(srcImg3, newRoi) : _ref.m_tracker_psr->updateAt(_srcImg, newRoi);
				  //success = (_preImg.channels() < 3) ? _ref.m_tracker_psr->update(srcImg3, newRoi) : _ref.m_tracker_psr->update(_srcImg, newRoi);
			  }

			  if (!success) { // retry with new rect 
				  // need to verify if the given object is not continous, the following code enforces the following the algorithm
				  newRoi = expRect;
				  success = (_preImg.channels() < 3) ? _ref.m_tracker_psr->reinit(preImg3, newRoi) : _ref.m_tracker_psr->reinit(_preImg, newRoi);
				  success = (_preImg.channels() < 3) ? _ref.m_tracker_psr->update(srcImg3, newRoi) : _ref.m_tracker_psr->update(_srcImg, newRoi);
			  }			  

			  _new_rect = (success)? newRoi: expRect;		
			  if (_conf.debugShowImages&&_conf.debugShowImagesDetail) { // full image debug
				  cv::Mat tmp2 = _srcImg.clone();
				  cv::Rect prect = _new_rect;

				  if (tmp2.channels() < 3)
					  cvtColor(tmp2, tmp2, CV_GRAY2BGR);
				  cv::rectangle(tmp2, expRect, SCALAR_CYAN, 1);
				  if(success)
					  cv::rectangle(tmp2, prect, SCALAR_MAGENTA, 2);
				  else
					  cv::rectangle(tmp2, prect, SCALAR_RED, 2);
				  cv::imshow("Full Image Tracking Global location", tmp2);
				  //cv::waitKey(1);
			  }

		  }else{
			  // partial local tracking using FDSST 2019. 01. 17
			  cv::Size roiSize(max(2 * m_predictionRect.width, _srcImg.cols / 8), std::max(2 * m_predictionRect.height, _srcImg.rows / 8)); // origin: max small subImage selection, I need to check if we can reduce more
	  
			  if (roiSize.width > _srcImg.cols)
			  {
				  roiSize.width = _srcImg.cols;
			  }
			  if (roiSize.height > _srcImg.rows)
			  {
				  roiSize.height = _srcImg.rows;
			  }
			  cv::Point roiTL(m_predictionRect.x + m_predictionRect.width / 2 - roiSize.width / 2, m_predictionRect.y + m_predictionRect.height / 2 - roiSize.height / 2);
			  cv::Rect roiRect(roiTL, roiSize);			// absolute full image coordinates
			  Clamp(roiRect.x, roiRect.width, _srcImg.cols);
			  Clamp(roiRect.y, roiRect.height, _srcImg.rows);

			  cv::Rect2d lastRect(m_predictionRect.x - roiRect.x, m_predictionRect.y - roiRect.y, m_predictionRect.width, m_predictionRect.height);
			  // relative subImage coordinates
		  
			  cv::Rect2d newsubRoi = lastRect;  // local window rect
			  if (lastRect.x >= 0 &&
				  lastRect.y >= 0 &&
				  lastRect.x + lastRect.width < roiRect.width &&
				  lastRect.y + lastRect.height < roiRect.height &&
				  lastRect.area() > 0) {
				  cv::Mat preImg3, srcImg3;
				  if (_preImg.channels() < 3) {
					  cv::cvtColor(_preImg, preImg3, CV_GRAY2BGR);
					  cv::cvtColor(_srcImg, srcImg3, CV_GRAY2BGR);
				  }

				  if (!_ref.m_tracker_initialized) {
					  success = (_preImg.channels() < 3) ?
						  _ref.m_tracker_psr->reinit(cv::Mat(preImg3, roiRect), newsubRoi) :
						  _ref.m_tracker_psr->reinit(cv::Mat(_preImg, roiRect), newsubRoi);
					  _ref.m_tracker_initialized = true; // ??????????????						
				  }
				  else {
					  success = (_preImg.channels() < 3) ?
						  _ref.m_tracker_psr->updateAt(cv::Mat(srcImg3, roiRect), newsubRoi) :
						  _ref.m_tracker_psr->updateAt(cv::Mat(_srcImg, roiRect), newsubRoi);
				  }

				  // update position and update the current frame because we not do this sometime if necessary						
				  if (!success) { // lastRect is not new, and old one is used								
					  newsubRoi = lastRect;
					  success = (_preImg.channels() < 3) ?
						  _ref.m_tracker_psr->reinit(cv::Mat(preImg3, roiRect), newsubRoi) :
						  _ref.m_tracker_psr->reinit(cv::Mat(_preImg, roiRect), newsubRoi);

					  success = (_preImg.channels() < 3) ?
						  _ref.m_tracker_psr->update(cv::Mat(srcImg3, roiRect), newsubRoi) :
						  _ref.m_tracker_psr->update(cv::Mat(_srcImg, roiRect), newsubRoi);
				  } // else is not required because if updateAt is successful, the new location after update is same							

				  cv::Rect prect;
				  if (success) {
					  prect = cv::Rect(cvRound(newsubRoi.x) + roiRect.x, cvRound(newsubRoi.y) + roiRect.y, cvRound(newsubRoi.width), cvRound(newsubRoi.height)); // new global location 			  
				  }
				  else {
					  prect = cv::Rect(cvRound(lastRect.x) + roiRect.x, cvRound(lastRect.y) + roiRect.y, cvRound(lastRect.width), cvRound(lastRect.height)); //  keep the old one			  
				  }
				  _new_rect = prect;

				  if (_conf.debugShowImages&&_conf.debugShowImagesDetail) { // local image debug
					  cv::Mat tmp2 = cv::Mat(_srcImg, roiRect).clone();
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
					  cv::imshow("local track", tmp2);
					  //cv::waitKey(1);
				  }
				  if (_conf.debugShowImages&&_conf.debugShowImagesDetail) { // full image debug
					  cv::Mat tmp2 = _srcImg.clone();
					  if (tmp2.channels() < 3)
						  cvtColor(tmp2, tmp2, CV_GRAY2BGR);
					  cv::rectangle(tmp2, expRect, SCALAR_CYAN, 1);
					  cv::rectangle(tmp2, prect, SCALAR_MAGENTA, 2);
					  cv::imshow("Full Image Tracking location", tmp2);
					 // cv::waitKey(1);
				  }

			  }
		  }
	  }
	  else {
		  if (_ref.m_tracker_psr || !_ref.m_tracker_psr.empty()) {
			  _ref.m_tracker_psr.release();
			  _ref.m_tracker_initialized = false;
		  }
	  }

	  return success;
  }
  bool trackNewLocation(const itms::Config& _conf, const cv::Mat& _preImg, const cv::Mat& _srcImg, itms::Blob& _ref, cv::Rect& _new_rect, const int _expandY) {

	  bool bexternalTrackerForLost = _conf.m_externalTrackerForLost;  // does use an external tracker?
	  bool bisSubImgTracking = _conf.isSubImgTracking;                // is it subImgTracking (local tracker) ? 
	  bool success = false;

	  if (bexternalTrackerForLost/*_conf.m_externalTrackerForLost*/) {
		  // main routine -- common part for global and local region-based tracker 
		  cv::Rect newRoi, m_predictionRect;
		  int expandY = _expandY;
		  float heightRatio = (float)(_ref.currentBoundingRect.height + expandY) / (float)(_ref.currentBoundingRect.height);
		  int expandX = max(0, cvRound((float)(_ref.currentBoundingRect.width)*heightRatio - _ref.currentBoundingRect.width));
		  cv::Rect expRect = expandRect(_ref.currentBoundingRect, expandX, expandY, _preImg.cols, _preImg.rows);

		  m_predictionRect = expRect;//existingBlob.currentBoundingRect;//expRect;			

		  if (!_ref.m_tracker_psr || _ref.m_tracker_psr.empty()) {
			  _ref.CreateExternalTracker();
		  }
		  if (!bisSubImgTracking) { // global approach
			  cv::Mat preImg3, srcImg3;
			  if (_preImg.channels() < 3) {
				  cv::cvtColor(_preImg, preImg3, CV_GRAY2BGR);
				  cv::cvtColor(_srcImg, srcImg3, CV_GRAY2BGR);
			  }
			  newRoi = expRect;
			  if (!_ref.m_tracker_initialized) {		// do it only once																				  
				  success = (_preImg.channels() < 3) ? _ref.m_tracker_psr->reinit(preImg3, newRoi) : _ref.m_tracker_psr->reinit(_preImg, newRoi);
				  _ref.m_tracker_initialized = true;
			  }
			  else {
				  //success = (_preImg.channels() < 3) ? _ref.m_tracker_psr->updateAt(srcImg3, newRoi) : _ref.m_tracker_psr->updateAt(_srcImg, newRoi);
				  success = (_preImg.channels() < 3) ? _ref.m_tracker_psr->update(srcImg3, newRoi) : _ref.m_tracker_psr->update(_srcImg, newRoi);
			  }

			  //if (!success) { // retry with new rect 
					//		  // need to verify if the given object is not continous, the following code enforces the following the algorithm
				 // newRoi = expRect;
				 // success = (_preImg.channels() < 3) ? _ref.m_tracker_psr->reinit(preImg3, newRoi) : _ref.m_tracker_psr->reinit(_preImg, newRoi);
				 // success = (_preImg.channels() < 3) ? _ref.m_tracker_psr->update(srcImg3, newRoi) : _ref.m_tracker_psr->update(_srcImg, newRoi);
			  //}

			  _new_rect = (success) ? newRoi : expRect;
			  if (_conf.debugShowImages&&_conf.debugShowImagesDetail) { // full image debug
				  cv::Mat tmp2 = _srcImg.clone();
				  cv::Rect prect = _new_rect;

				  if (tmp2.channels() < 3)
					  cvtColor(tmp2, tmp2, CV_GRAY2BGR);
				  cv::rectangle(tmp2, expRect, SCALAR_CYAN, 1);
				  if (success)
					  cv::rectangle(tmp2, prect, SCALAR_MAGENTA, 2);
				  else
					  cv::rectangle(tmp2, prect, SCALAR_RED, 2);
				  cv::imshow("Full Image Tracking Global location", tmp2);
				  //cv::waitKey(1);
			  }

		  }
		  else {
			  // partial local tracking using FDSST 2019. 01. 17
			  cv::Size roiSize(max(2 * m_predictionRect.width, _srcImg.cols / 8), std::max(2 * m_predictionRect.height, _srcImg.rows / 8)); // origin: max small subImage selection, I need to check if we can reduce more

			  if (roiSize.width > _srcImg.cols)
			  {
				  roiSize.width = _srcImg.cols;
			  }
			  if (roiSize.height > _srcImg.rows)
			  {
				  roiSize.height = _srcImg.rows;
			  }
			  cv::Point roiTL(m_predictionRect.x + m_predictionRect.width / 2 - roiSize.width / 2, m_predictionRect.y + m_predictionRect.height / 2 - roiSize.height / 2);
			  cv::Rect roiRect(roiTL, roiSize);			// absolute full image coordinates
			  Clamp(roiRect.x, roiRect.width, _srcImg.cols);
			  Clamp(roiRect.y, roiRect.height, _srcImg.rows);

			  cv::Rect2d lastRect(m_predictionRect.x - roiRect.x, m_predictionRect.y - roiRect.y, m_predictionRect.width, m_predictionRect.height);
			  // relative subImage coordinates

			  cv::Rect2d newsubRoi = lastRect;  // local window rect
			  if (lastRect.x >= 0 &&
				  lastRect.y >= 0 &&
				  lastRect.x + lastRect.width < roiRect.width &&
				  lastRect.y + lastRect.height < roiRect.height &&
				  lastRect.area() > 0) {
				  cv::Mat preImg3, srcImg3;
				  if (_preImg.channels() < 3) {
					  cv::cvtColor(_preImg, preImg3, CV_GRAY2BGR);
					  cv::cvtColor(_srcImg, srcImg3, CV_GRAY2BGR);
				  }

				  if (!_ref.m_tracker_initialized) {
					  success = (_preImg.channels() < 3) ?
						  _ref.m_tracker_psr->reinit(cv::Mat(preImg3, roiRect), newsubRoi) :
						  _ref.m_tracker_psr->reinit(cv::Mat(_preImg, roiRect), newsubRoi);
					  _ref.m_tracker_initialized = true; // ??????????????						
				  }
				  else {
					  success = (_preImg.channels() < 3) ?
						  _ref.m_tracker_psr->update(cv::Mat(srcImg3, roiRect), newsubRoi) :
						  _ref.m_tracker_psr->update(cv::Mat(_srcImg, roiRect), newsubRoi); // it is different from newLocationFromprevBlob
				  }				  

				  cv::Rect prect;
				  if (success) {
					  prect = cv::Rect(cvRound(newsubRoi.x) + roiRect.x, cvRound(newsubRoi.y) + roiRect.y, cvRound(newsubRoi.width), cvRound(newsubRoi.height)); // new global location 			  
				  }
				  else {
					  prect = cv::Rect(cvRound(lastRect.x) + roiRect.x, cvRound(lastRect.y) + roiRect.y, cvRound(lastRect.width), cvRound(lastRect.height)); //  keep the old one			  
				  }
				  _new_rect = prect;

				  if (_conf.debugShowImages && _conf.debugSpecial) { // local image debug
					  cv::Mat tmp2 = cv::Mat(_srcImg, roiRect).clone();
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
					  cv::imshow("local track", tmp2);
					  //cv::waitKey(1);
				  }
				  if (_conf.debugShowImages&&_conf.debugShowImagesDetail) { // full image debug
					  cv::Mat tmp2 = _srcImg.clone();
					  if (tmp2.channels() < 3)
						  cvtColor(tmp2, tmp2, CV_GRAY2BGR);
					  cv::rectangle(tmp2, expRect, SCALAR_CYAN, 1);
					  cv::rectangle(tmp2, prect, SCALAR_MAGENTA, 2);
					  cv::imshow("Full Image Tracking location", tmp2);
					  //cv::waitKey(1);
				  }

			  }
		  }
	  }
	  else {
		  if (_ref.m_tracker_psr || !_ref.m_tracker_psr.empty()) {
			  _ref.m_tracker_psr.release();
			  _ref.m_tracker_initialized = false;
		  }
	  }

	  return success;
  }

  bool compareContourAreasDes(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	  double i = fabs(contourArea(cv::Mat(contour1)));
	  double j = fabs(contourArea(cv::Mat(contour2)));
	  return (i > j); // descending order
  }

  bool compareContourAreasAsc(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	  double i = fabs(contourArea(cv::Mat(contour1)));
	  double j = fabs(contourArea(cv::Mat(contour2)));
	  return (i < j); // ascending order
  }

  float itmsFunctions::getNCC(itms::Config& _conf, cv::Mat &bgimg, cv::Mat &fgtempl, cv::Mat &fgmask, int match_method/* cv::TM_CCOEFF_NORMED*/, bool use_mask/*false*/) {
	  //// template matching algorithm implementation, demo	
	  if (1 && _conf.debugGeneralDetail) {
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
	  if (1 && _conf.debugShowImages && _conf.debugShowImagesDetail) {
		  imshowBeforeAndAfter(bgimg_gray, fgtempl_gray, "NCC cur/template img", 2);
		  /*imshow("img", bgimg_gray);
		  imshow("template image", fgtempl_gray);*/
		  //cv::waitKey(1);
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

// -------------- previous status 
	int os_nt_center = updateBlob.os_notdetermined_cnter;
	int os_numCSStoppted_center =updateBlob.os_NumOfConsecutiveStopped_cnter;  
	int os_numCSForward_center = updateBlob.os_NumOfConsecutivemvForward_cnter;
	int os_numCSBackward_center = updateBlob.os_NumOfConsecutivemvBackward_cnter;
// ------------------------------

	  int minConsecutiveFramesForOS = _conf.minConsecutiveFramesForOS;	// minConsecutiveFrames For OS // sangkny 20190404 according to the distance of the object

	  std::vector<cv::Point2f> blob_ntPts;
	  blob_ntPts.push_back(Point2f(updateBlob.centerPositions.back()));
	  if ((_conf.scaleFactor < 0.75)&&(updateBlob.oc != ObjectClass::OC_HUMAN) && (prevOS == OS_STOPPED || prevOS == OS_MOVING_BACKWARD)) {
		  float realDistance = getDistanceInMeterFromPixels(blob_ntPts, _conf.transmtxH, _conf.lane_length, false);
		  if (realDistance / 100 >= 150.)
			  minConsecutiveFramesForOS *= 2;
		  else if (realDistance / 100 >= 100.) {
			  minConsecutiveFramesForOS = (int) ((float)minConsecutiveFramesForOS * 1.5);
		  }
		  else if (realDistance / 100 >= 75) {
			  minConsecutiveFramesForOS = (int)((float)minConsecutiveFramesForOS * 1.25);
		  }
	  }


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
		  if (updateBlob.oc != OC_HUMAN && updateBlob.fos == OS_STOPPED) {
			  if (!doubleCheckStopObject(_conf, updateBlob)) { // restore the previous settings
				  updateBlob.fos = OS_NOTDETERMINED;
				  updateBlob.os_stopped_cnter--;
				  updateBlob.os_NumOfConsecutiveStopped_cnter = 0; // reset
				  updateBlob.os_NumOfConsecutivemvBackward_cnter = os_numCSBackward_center; 				  
				  updateBlob.os_NumOfConsecutivemvForward_cnter = os_numCSForward_center;
				  updateBlob.os_notdetermined_cnter += 1;
			  }
			  else { // sangkny 20190517 to check the moving obeckt from the stopped condition because of BGround image
				  if (updateBlob.os_stopped_cnter > std::max(updateBlob.os_mvBackward_cnter,updateBlob.os_mvForward_cnter)) {
					  updateBlob.startPoint = updateBlob.centerPositions.back(); // sangkny 20190404 reset the stop position to determine the WWR
				  }
			  }
		  }
		  break;	  

	  case OS_MOVING_BACKWARD:
		  updateBlob.os_mvBackward_cnter++;
		  updateBlob.os_NumOfConsecutivemvBackward_cnter = (prevOS == OS_MOVING_BACKWARD) ? updateBlob.os_NumOfConsecutivemvBackward_cnter + 1 : 1;
		  updateBlob.os_NumOfConsecutiveStopped_cnter = 0;  // reset the consecutive counter
		  updateBlob.os_NumOfConsecutivemvForward_cnter = 0;
		  //blob.os_NumOfConsecutivemvBackward_cnter = 1;
		  updateBlob.fos = (updateBlob.os_NumOfConsecutivemvBackward_cnter >= minConsecutiveFramesForOS) ? OS_MOVING_BACKWARD : OS_NOTDETERMINED;
		  if ((updateBlob.oc != OC_HUMAN) && updateBlob.fos == OS_MOVING_BACKWARD) { // vehicle only
			  if (!doubleCheckBackwardMoving(_conf, updateBlob)) { // restore the previous settings
				  updateBlob.fos = OS_NOTDETERMINED;
				  updateBlob.os_mvBackward_cnter--;
				  updateBlob.os_NumOfConsecutivemvBackward_cnter = 0; // reset 
				  updateBlob.os_NumOfConsecutiveStopped_cnter =  os_numCSStoppted_center;
				  updateBlob.os_NumOfConsecutivemvForward_cnter = os_numCSForward_center;
				  updateBlob.os_notdetermined_cnter += 1;				  
			  }

		  }

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
		  perc_Thres = 0.8; // should be bigger

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

  bool itmsFunctions::checkObjectStatus(itms::Config & _conf, const cv::Mat& _curImg, std::vector<Blob>& _Blobs, itms::ITMSResult & _itmsRes)
  {   
	  bool checkStatus = false;	  

	  std::vector<Blob>::iterator curBlob = _Blobs.begin();
	  while (curBlob != _Blobs.end()) {
		  std::vector<cv::Point2f> blob_ntPts;
		  blob_ntPts.push_back(Point2f(curBlob->centerPositions.back()));
		  
		  if (curBlob->bNotifyMessage || (_conf.bStrictObjEvent && curBlob->fos == ObjectStatus::OS_NOTDETERMINED) || (!checkIfPointInBoundary(_conf, blob_ntPts.back(), _conf.Boundary_ROI_Pts)))
		  {   // notified, then skip, if bStrictObjEvent, strict determination is conducted according to ObjectStatus
			  // sangkny 20190404
			  ++curBlob;
			  continue;
		  }
		  if (curBlob->oc == ObjectClass::OC_HUMAN ) { // 무조건 찾는다.
			  if(1 || curBlob->oc_prob <= 0.99){
				  std::vector<cv::Rect> _people;
				  detectCascadeRoiHuman(_conf, _curImg, curBlob->currentBoundingRect,_people); // sangkny 20190331 check the human with original sized image 
				  if(_people.size()==0){
					   //한번 더 원래 이미지로
					   _people.clear();
					   cv::Rect _roi_rect = curBlob->currentBoundingRect; //상대좌표가 들어와야 한다.
					   _roi_rect.x = (float)curBlob->currentBoundingRect.x/_config->scaleFactor;
					   _roi_rect.y = (float)curBlob->currentBoundingRect.y / _config->scaleFactor;
					   _roi_rect.width = (float)curBlob->currentBoundingRect.width / _config->scaleFactor;
					   _roi_rect.height = (float)curBlob->currentBoundingRect.height / _config->scaleFactor;
					   _roi_rect = expandRect(_roi_rect, 8, 8, orgImage.cols,orgImage.rows);

					   detectCascadeRoiHuman(_conf, orgImage, _roi_rect, _people); // sangkny 20190331 check the human with original sized image 
					   if (_people.size() == 0) {
						   curBlob->oc = itms::ObjectClass::OC_OTHER;
						   ++curBlob;
						   continue;
					   }
				  }
			  }
			  if(curBlob->totalVisibleCount<_conf.minVisibleCountHuman){ // check if the given the object has enough visible counts
				  if(_conf.debugGeneralDetail)
					  std::cout << "The visible count is not enough : " << curBlob->totalVisibleCount << std::endl;
				  ++curBlob;
				  continue;
			  }
			  //else{
			//	  ; // do nothing
			 // }
			  //// only human confirm the its class with ML-based approach
			  //std::vector<cv::Rect> _people;
			  //cv::Rect _roi_rect = curBlob->currentBoundingRect; //상대좌표가 들어와야 한다.
			  //_roi_rect.x = (float)curBlob->currentBoundingRect.x/_config->scaleFactor;
			  //_roi_rect.y = (float)curBlob->currentBoundingRect.y / _config->scaleFactor;
			  //_roi_rect.width = (float)curBlob->currentBoundingRect.width / _config->scaleFactor;
			  //_roi_rect.height = (float)curBlob->currentBoundingRect.height / _config->scaleFactor;
			  //expandRect(_roi_rect, 8, 8, orgImage.cols,orgImage.rows);

			  //detectCascadeRoiHuman(_conf, orgImage, _roi_rect, _people); // sangkny 20190331 check the human with original sized image 
			  //if (_people.size() == 0) {
				 // curBlob->oc =itms::ObjectClass::OC_OTHER;
				 // if(_conf.debugShowImagesDetail)
					//  std::cout<< "Final numan confirmation is not satisfied !!! in detectCascadeRoiHuman id:"<< std::to_string(curBlob->id) << endl;

				 // ++curBlob;
				 // continue;
			  //}
			  _itmsRes.objClass.push_back(std::pair<int, int>(curBlob->id, ObjectClass::OC_HUMAN));
			  _itmsRes.objStatus.push_back(std::pair<int, int>(curBlob->id, curBlob->fos));
			  _itmsRes.objRect.push_back(curBlob->currentBoundingRect);
			  _itmsRes.objSpeed.push_back(curBlob->speed);
			  checkStatus = true;
			  curBlob->bNotifyMessage = (_conf.bNoitifyEventOnce )? true: false;
			  curBlob->oc_notified = curBlob->oc;
			  curBlob->os_notified = curBlob->fos;
		  }
		  else { // vehicle class
			  // WWD
			  // STOP			  

			  if (curBlob->oc!= ObjectClass::OC_OTHER &&
			  ((curBlob->fos == ObjectStatus::OS_MOVING_BACKWARD && curBlob->speed >= _conf.speedLimitForstopping) || 
			  curBlob->fos == ObjectStatus::OS_STOPPED)) {
			  // 야간모드에서는 밝기와 variance를 체크한다. 
				std::vector<cv::Point2f> blob_ntPts;
				blob_ntPts.push_back(Point2f(curBlob->centerPositions.back()));
				float realDistance = getDistanceInMeterFromPixels(blob_ntPts, _config->transmtxH, _config->lane_length, false);
				cv::Rect roi_rect = curBlob->currentBoundingRect;

				if (realDistance / 100.f > 100.f) {
					roi_rect = expandRect(roi_rect, 4, 4, BGImage.cols, BGImage.rows);
				}
				else {
					roi_rect = expandRect(roi_rect, 8, 8, BGImage.cols, BGImage.rows);
				}
				cv::Rect roi_rect_Ex = expandRect(roi_rect, 2, 2, BGImage.cols, BGImage.rows);
				//// 마지막으로 현재 ROI의 variance 가 0에서 멀어져야 한다. 초기 bg image가 제대로 만들어 지지 않으면 오검지 될 수 있음..
				// 아래 부분은 너무 밝거나 어두운 부분에서도 밝은 곳 (200 <mean and 40> std, 그리고, 120 > mean  and 20 > std 곳에서는 물체라고 보기는 어렵다. 나중에 활용해 보도록
				// 하지만 현재는 너무 제한을 하면 미검지가 많을 수 있으므로 일단 보류 
				// 2019. 05. 22
				if (_config->img_dif_th > _config->nightBrightness_Th ) { // only when at night
					cv::Scalar mean, dev;
					cv::meanStdDev(_curImg(roi_rect_Ex), mean, dev);
					double _curROI_var = dev.val[0];
					double _curROI_mean = mean.val[0];					
					if (_config->debugGeneralDetail) {
						std::cout << "mean: " << std::to_string(_curROI_mean) << ", std dev in STOP ROI: " << std::to_string(_curROI_var) << endl;
						cv::imshow("WWR/Stop Object ROI", cv::Mat(_curImg(roi_rect_Ex)));
					}
					if ((_curROI_mean > 200 || _curROI_var < 50) && curBlob->fos == ObjectStatus::OS_MOVING_BACKWARD/*|| (_curROI_mean < 100 && _curROI_var < 30)*/) {
						++curBlob;
						continue;
					}
					if ((_curROI_mean > 200 || _curROI_var < 40) && curBlob->fos == ObjectStatus::OS_STOPPED/* because of break light, variance will be decreased */) {
						++curBlob;
						continue;
					}
				}

			  // sangkny 20190331 check NCC when stopped 
			  if(curBlob->fos == ObjectStatus::OS_STOPPED){
				  
				  // blob correlation debug
				  // roi_rect_Ex 와 일부러 차이가 있게 하였음 
				  if (_config->debugShowImagesDetail) {					  					  
					  itms::imshowBeforeAndAfter(BGImage(roi_rect_Ex), _curImg(roi_rect), "STOP<<NCC BG / Target>> ", 2);					  					  
				  }
				  float blobncc = getNCC(_conf, BGImage(roi_rect_Ex), _curImg(roi_rect)/*roi_rect_Ex 와 일부러 차이가 있게 하였음 */, Mat(), _config->match_method, _config->use_mask);
				  if (abs(blobncc) > _config->BlobNCC_Th) // background 와 템플릿 매칭에서 비슷한 결과가 나오면
				  {
					  if(_conf.debugGeneralDetail)
						  std::cout<< " Stopped Object and NCC is not satisfiend !!!!!!!!!!!!!!: "<< (blobncc)<< std::endl;
					  
					  ++curBlob;
					  continue;
				  }else if ((abs(blobncc) > _config->BlobNCC_Th/2.)) { // 백그라운드와 유사시...stop 구분  (양쪽에 차량이 있으면 안됨)
					  std::vector<cv::Rect> _cars, _allcars;					  					  
					  detectCascadeRoiVehicle(_conf, BGImage, roi_rect_Ex, _cars, _allcars);
					  if (_cars.size()!=0) { //  백그라운드에 오프젝트가 있으면 안된다.
							curBlob->os = itms::ObjectStatus::OS_NOTDETERMINED;
							if (_conf.debugShowImagesDetail)
								  std::cout << "Vehicle is in background !!! in detectCascadeRoiVehicle id:" << std::to_string(curBlob->id) << endl;
							++curBlob;
							continue;
					 } // 여기까지 배경에 차가 없으면 stop 이라고 보는 것이 좋을 것 같다. 그렇지 않으면 너무 강한 제한조건이 생긴다.
					  // 다음의 #if 1은 강한 조건이다. 현재는 0으로 없애고 본다. 
#if 0                 // 114411 에서 진행후 정지, 그리고 사람이 나오는 부분에서 stop이 안잡힐 수 있다.
					  _cars.clear();
					  detectCascadeRoiVehicle(_conf, _curImg, roi_rect_Ex, _cars); // resized image first
					  if (_cars.size() == 0) { //  백그라운드에 오프젝트가 있으면 안된다.
						  // one more time  in original image
						  cv::Rect _roi_rect_org;
						  float scaleBack = 1./_conf.scaleFactor;
						  _roi_rect_org.x = roi_rect_Ex.x *scaleBack;
						  _roi_rect_org.y = roi_rect_Ex.y *scaleBack;
						  _roi_rect_org.width = roi_rect_Ex.width*scaleBack;
						  _roi_rect_org.height = roi_rect_Ex.height*scaleBack;
						  detectCascadeRoiVehicle(_conf, orgImage, _roi_rect_org, _cars);
						  if(_cars.size() == 0){
							  curBlob->os = itms::ObjectStatus::OS_NOTDETERMINED;
							  if (_conf.debugShowImagesDetail)
								  std::cout << "Vehicle is not in the curBlob !!! in detectCascadeRoiVehicle id:" << std::to_string(curBlob->id) << endl;
							  ++curBlob;
							  continue;
						  }
					  }
#endif

				}
				
				  //detectCascadeRoiVehicle(_conf, orgImage, _roi_rect, _cars); // sangkny 20190331 check the vehicle with original sized image 
				  //if (_cars.size() == 0) {
					 // //curBlob->oc = itms::ObjectClass::OC_OTHER;
					 // if (_conf.debugShowImagesDetail)
						//  std::cout << "Final vehicle confirmation is not satisfied !!! in detectCascadeRoiVehicle id:" << std::to_string(curBlob->id) << endl;
					 // ++curBlob;
					 // continue;
				  //}
				  //// 마지막으로 현재 ROI의 variance 가 0에서 멀어져야 한다. 초기 bg image가 제대로 만들어 지지 않으면 오검지 될 수 있음..
				  // 아래 부분은 너무 밝거나 어두운 부분에서도 밝은 곳 (200 <mean and 40> std, 그리고, 120 > mean  and 20 > std 곳에서는 물체라고 보기는 어렵다. 나중에 활용해 보도록
				  // 하지만 현재는 너무 제한을 하면 미검지가 많을 수 있으므로 일단 보류 
				  // 2019. 05. 22
				  //cv::Scalar mean, dev;
				  //cv::meanStdDev(_curImg(roi_rect_Ex), mean, dev);
				  //double _curROI_var = dev.val[0];
				  //double _curROI_mean = mean.val[0];
				  //std::cout << "mean: "<< std::to_string(_curROI_mean)<<", std dev in STOP ROI: " << std::to_string(_curROI_var) << endl;
				  //cv::imshow("Stop Object ROI", cv::Mat(_curImg(roi_rect_Ex)));				  

				  if (_conf.debugShowImages && _conf.debugSpecial) {
					  cv::Mat debugImg = _curImg.clone();
					  if (debugImg.channels() < 3)
						  cvtColor(debugImg, debugImg, CV_GRAY2BGR);
					  cv::rectangle(debugImg, roi_rect_Ex, Scalar(0, 0, 255), 3);
					  cv::imshow("Stopped Object", debugImg);
					  //cv::waitKey(1);
				  }
					  
			  }
			_itmsRes.objClass.push_back(std::pair<int, int>(curBlob->id, curBlob->oc));
			_itmsRes.objStatus.push_back(std::pair<int, int>(curBlob->id, curBlob->fos));
			_itmsRes.objRect.push_back(curBlob->currentBoundingRect);
			_itmsRes.objSpeed.push_back(curBlob->speed);
			checkStatus = true;
			curBlob->bNotifyMessage = (_conf.bNoitifyEventOnce) ? true : false;
			curBlob->oc_notified = curBlob->oc;
			curBlob->os_notified = curBlob->fos;
				  
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
		  //cv::waitKey(1);
	  }

  }
  void detectCascadeRoiVehicle(itms::Config& _conf,/* put config file */cv::Mat img, cv::Rect& rect, std::vector<cv::Rect>& _cars, std::vector<cv::Rect>&object)
  { /* please see more details in Object_Detector_Cascade Project */
	  Mat roiImg = img(rect).clone();

	  int casWidth = 128; // ratio is 1: 1 for width to height
	  /*if (_conf.bgsubtype == BgSubType::BGS_CNT)
		  casWidth = (int)((float)casWidth *1.5);*/

	  // adjust cascade window image
	  float casRatio = (float)casWidth / roiImg.cols;

	  resize(roiImg, roiImg, Size(), casRatio, casRatio);	  

	  Size img_size = roiImg.size();
	  //vector<Rect> object;
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
		  //cv::waitKey(1);
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
	  if (((RefRatio - minMax_threshold-0.4) > h_w_ratio) 
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
		  //cv::waitKey(1);
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
		  //cv::waitKey(1);
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
				  //cv::waitKey(1);
			  }
		  }
	  }
  }

  int ResetAndProcessFrame(int iCh, unsigned char * pImage, int lSize)
  {
	  return 0;
  }

  std::vector<cv::Point> getBlobUnderRect(const Config &_conf, const cv::Mat& _curImg, const cv::Rect& _prect, const itms::Blob& _curBlob) {
	  // it will do when it is not human
	  std::vector<cv::Point> _contour;
	  if (_curBlob.oc == itms::ObjectClass::OC_HUMAN)
		  return _contour;

	  bool bdebug = true;
	  
	  cv::Mat roi_Img = _curImg(_prect);
	  cv::Mat roi_Thresh;

	  //adaptiveThreshold(roi_Img, roi_Thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 3);
	  cv::Canny(roi_Img, roi_Thresh, 50, 150);
	  if (bdebug && _conf.debugShowImagesDetail) {
		  imshowBeforeAndAfter(roi_Img, roi_Thresh, "traker roi image/ threshold", 2);		  
	  }

	  std::vector<std::vector<cv::Point> > contours;

	  /*Mat element = getStructuringElement(1,
		  Size(2 + 1, 2 + 1),
		  Point(1, 1));
	  dilate(roi_Thresh, roi_Thresh, element);*/ // because the current image has been blurred in the preprocessing 

	  cv::findContours(roi_Thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	  if (bdebug && _conf.debugShowImages && _conf.debugShowImagesDetail) {
		  drawAndShowContours(roi_Thresh.size(), contours, "roi_imgContours");
	  }

	  std::vector<std::vector<cv::Point> > convexHulls(contours.size());

	  // get the biggest one
	  double largest_area = 0;
	  int  largest_index =0;
	  for (unsigned int i = 0; i < contours.size(); i++) {
		  cv::convexHull(contours[i], convexHulls[i]);
		  double area = cv::boundingRect(convexHulls[i]).area();
		  if (area > largest_area) {
			  largest_area = area;
			  largest_index = i;			  
		  }
	  }

	  if (bdebug && _conf.debugShowImages && _conf.debugShowImagesDetail) {
		  drawAndShowContours(roi_Thresh.size(), convexHulls, "roi_imgConvexHulls");
		  if (_conf.debugGeneralDetail && contours.size()) {			  
				  std::cout << "largest index: " << largest_index << endl;
				  std::cout << "largest area: " << largest_area << endl;			  
		  }
	  }

	  /*std::vector<Blob> _currentBlobs;

	  
	  for (auto &convexHull : convexHulls) {
		  Blob possibleBlob(convexHull);

		  if (possibleBlob.currentBoundingRect.area() >= _curBlob.currentBoundingRect.area()) {
			  _currentBlobs.push_back(possibleBlob);
		  }
	  }*/
	  if (contours.size()) {
		  _contour = contours[largest_index];	  
	  }	  
	  return _contour;
  }
  bool doubleCheckStopObject(const Config & _conf, itms::Blob & _curBlob) {	  

	  double refdist = (double)(_conf.minDistanceForBackwardMoving); // same condition with BackwardMoving in cm
	  cv::Point2f sPt = cvtPx2RealPx(static_cast<cv::Point2f>(_curBlob.startPoint), _conf.transmtxH); // starting pt
	  cv::Point2f ePt = cvtPx2RealPx(static_cast<cv::Point2f>(_curBlob.centerPositions.back()), _conf.transmtxH);                                  // end pt

	  double dist = distanceBetweenPoints(sPt, ePt); // real distance (cm) in the ROI 	  

	  return (dist >= refdist);
  }
  bool doubleCheckBackwardMoving(const Config & _conf, itms::Blob & _curBlob)
  {	
	  // check the condition with the starting points
	  bool bbackwardcondition = false;
	  switch (_conf.ldirection) {
	  case LD_NORTH:
		  if (_curBlob.centerPositions.back().y > _curBlob.startPoint.y)
			  bbackwardcondition = true;
		  break;
	  case LD_SOUTH:
		  if (_curBlob.centerPositions.back().y < _curBlob.startPoint.y)
			  bbackwardcondition = true;
		  break;
	  case LD_EAST:
		  if (_curBlob.centerPositions.back().x < _curBlob.startPoint.x)
			  bbackwardcondition = true;
		  break;
	  case LD_WEST:
		  if (_curBlob.centerPositions.back().x > _curBlob.startPoint.x)
			  bbackwardcondition = true;
		  break;
	  case LD_NORTHEAST:
		  if (_curBlob.centerPositions.back().y > _curBlob.startPoint.y || _curBlob.centerPositions.back().x < _curBlob.startPoint.x)
			  bbackwardcondition = true;
		  break;
	  case LD_SOUTHWEST:
		  if (_curBlob.centerPositions.back().y < _curBlob.startPoint.y || _curBlob.centerPositions.back().x > _curBlob.startPoint.x)
			  bbackwardcondition = true;
		  break;
	  case LD_SOUTHEAST:
		  if (_curBlob.centerPositions.back().y < _curBlob.startPoint.y || _curBlob.centerPositions.back().x < _curBlob.startPoint.x)
			  bbackwardcondition = true;
		  break;
	  case LD_NORTHWEST:
		  if (_curBlob.centerPositions.back().y > _curBlob.startPoint.y || _curBlob.centerPositions.back().x > _curBlob.startPoint.x)
			  bbackwardcondition = true;
		  break;

	  default:
		  bbackwardcondition = false;
		  break;
	  }
	  // check the minimumdistance ~10meters( 1000 cm)
	  if (bbackwardcondition) { // refer to minDistanceForBackwardMoving in config file
		  double refdist = (double)(_conf.minDistanceForBackwardMoving); // in cm
		  cv::Point2f sPt = cvtPx2RealPx(static_cast<cv::Point2f>(_curBlob.startPoint), _conf.transmtxH); // starting pt
		  cv::Point2f ePt = cvtPx2RealPx(static_cast<cv::Point2f>(_curBlob.centerPositions.back()), _conf.transmtxH);                                  // end pt

		  double dist = distanceBetweenPoints(sPt, ePt); // real distance (cm) in the ROI 
		  bbackwardcondition = (bbackwardcondition && (dist >= refdist));
	  }
	  return bbackwardcondition;
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
	  pBgSub = createBackgroundSubtractorMOG2(_config->intNumBGRefresh, _config->dblMOGVariance, true);
	  pBgOrgSub = createBackgroundSubtractorMOG2(_config->intNumBGRefresh, _config->dblMOGVariance*3./4., true);
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
			  cv::GaussianBlur(BGImage, BGImage, cv::Size(5, 5), 0);
			  accmImage = BGImage; // copy the background as an initialization
			  // road_mask define
			  road_mask = cv::Mat::zeros(BGImage.size(), BGImage.type());			 
			  for (int ir = 0; ir < _config->Road_ROI_Pts.size(); ir++) 
				  fillConvexPoly(road_mask, _config->Road_ROI_Pts.at(ir).data(), _config->Road_ROI_Pts.at(ir).size(), Scalar(255, 255, 255), 8);				  

			  // determine the ROI for zoomming region			  
			  if( _config->Boundary_ROI_Pts.size()==4) {
				  cv::Point ptl, ptr, lh, rh;

				  ptl = _config->Boundary_ROI_Pts.at(0);
				  ptr = _config->Boundary_ROI_Pts.at(1);
				  
				  itms::LineSegment l1(_config->Boundary_ROI_Pts.at(0), _config->Boundary_ROI_Pts.at(3)), l2(_config->Boundary_ROI_Pts.at(1), _config->Boundary_ROI_Pts.at(2));
				  lh = l1.midpoint();
				  rh = l2.midpoint();

				  this->zPmin = cv::Point2f(min(ptl.x, lh.x), min(ptl.y, lh.y));
				  this->zPmax = cv::Point2f(max(ptr.x, rh.x), max(ptr.y, rh.y)/2);
				  this->ozPmin = zPmin*(1./_config->scaleFactor);
				  this->ozPmax = zPmax*(1./_config->scaleFactor);
			  }			  

			  if (_config->debugShowImages && _config->debugShowImagesDetail) {
				  cv::Mat debugImg = road_mask.clone();
				  if (debugImg.channels() < 3)
					  cvtColor(debugImg, debugImg, CV_GRAY2BGR);
				  for (int i = 0; i < _config->Boundary_ROI_Pts.size(); i++)
					  line(debugImg, _config->Boundary_ROI_Pts.at(i% _config->Boundary_ROI_Pts.size()), _config->Boundary_ROI_Pts.at((i + 1) % _config->Boundary_ROI_Pts.size()), SCALAR_BLUE, 2);
				  imshow("road mask", debugImg);
				  //cv::waitKey(1);
			  }
		  }
	  }
	  else {
		  if (_config->debugGeneral) {
			  cout << "Please check the background file : " << _config->BGImagePath << endl;
			  cout << "The background will be current frame <!><!>.\n";
		  }
		  // alternative frame will be the previous frame at processing
		  
	  }
	  if (_config->debugSaveFile) {
		  std::string str(_config->VideoPath), from = ".avi", to = ".txt";
		  size_t start_pos = 0; // 

		  while ((start_pos = str.find(from, start_pos)) != std::string::npos)
		  {
			  str.replace(start_pos, from.length(), to);
			  start_pos += to.length();
		  } // retrun str;
		  out_object_class = ofstream(str);		
		  out_object_class << "scale factor \t"<< to_string(_config->scaleFactor)<< endl;
		  out_object_class << "class \t Width \t Height \t realDistance \t Probability\n";
	  }

	  return isInitialized = true;
  }

  bool itmsFunctions::process(const cv::Mat& curImg1, ITMSResult& _itmsRes) {
	  Mat curImg = curImg1.clone();
	  this->orgImage = curImg1.clone();
	  bool bDayMode = true; // 
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

	  // At first compute the brightness to determine the given environment is in day or night
	  // and it needs to adjust img_dif_th
	  if (_config->isAutoBrightness) {
		  //compute the roi brightness and then adjust the img_dif_th withe the past max_past_frames 
		  float roiMean = mean(curImg(brightnessRoi)/*currentGray roi*/)[0];
		  if (pastBrightnessLevels.size() >= _config->max_past_frames_autoBrightness)									// the size of vector is max_past_frames
			  pop_front(pastBrightnessLevels, pastBrightnessLevels.size() - _config->max_past_frames_autoBrightness + 1); // keep the number of max_past_frames
																														  //pop_front(pastBrightnessLevels); // remove an elemnt from the front of 			
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
	  
	  // determine day or night
	  bDayMode = (_config->img_dif_th<_config->nightBrightness_Th)? true : false;

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
			  //draw lanes
			  for (int i = 0; i < _config->Boundary_ROI_Pts.size(); i++)
				  line(debugImg, _config->Boundary_ROI_Pts.at(i% _config->Boundary_ROI_Pts.size()), _config->Boundary_ROI_Pts.at((i + 1) % _config->Boundary_ROI_Pts.size()), SCALAR_BLUE, 2);			  

			  imshow("road mask inside BGImage.empty loop", debugImg);
			  //cv::waitKey(1);
		  }
	  }
	  // we have two blurred images, now process the two image 
	  cv::Mat imgDifference, imgDifferenceBg; // difference images from  a previous and a background images (small image)
	  cv::Mat imgThresh, imgThreshBg;          // the corresponding thresholding images

	  cv::absdiff(preImg, curImg, imgDifference);

	  if (_config->bgsubtype == itms::BgSubType::BGS_CNT){
		  //pBgSub->setVarThreshold(10);
		  //int shadowValue = pBgSub->getShadowValue();		  
		  pBgSub->apply(curImg, imgDifferenceBg,1./(double)(_config->intNumBGRefresh));
		  pBgSub->getBackgroundImage(BGImage);		// 이것을 하면 뒷부분의 addWeight 을 해제해야 함 (즉, 메모리 아낄 수 있음) // 2019. 04. 30.
		  if (_config->debugShowImages && _config->debugShowImagesDetail) {
			  Mat bgImage = Mat::zeros(curImg.size(), curImg.type());
			  bgImage = getBGImage();			  // it shoud be same as pBgSub->getBackgroundImage(bgImage);			  
			  itms::imshowBeforeAndAfter(bgImage, imgDifferenceBg, "bg image and imgdiffBg", 2);
			  
			  /*if (isWriteToFile && frameCount == 200) {
				  string filename = conf.VideoPath;
				  filename.append("_" + to_string(conf.scaleFactor) + "x.jpg");
				  cv::imwrite(filename, bgImage);
				  std::cout << " background image has been generated (!!)\n";
			  }*/
		  }		 
	  }
	  else {		  
		  //std::cout <<" File BG type: " << type2str(BGImage.type()) << endl; // ==> 8UC1
		  cv::absdiff(BGImage, curImg, imgDifferenceBg);			
	  }
	  
	  if (!road_mask.empty()) {
		  cv::bitwise_and(road_mask, imgDifference, imgDifference);	  		  
		  cv::bitwise_and(road_mask, imgDifferenceBg, imgDifferenceBg);
		  cv::medianBlur(imgDifferenceBg, imgDifferenceBg, 3); // 20190404 -> median blur is requred for BackGroundSubstract		  
		  
		  if (_config->debugShowImages && _config->debugShowImagesDetail) {
			  itms::imshowBeforeAndAfter(imgDifference, imgDifferenceBg, " rmask& imgdiff / Bg before thresholding",2);			  
		  }

		  float roi_bg_mean = mean(BGImage(brightnessRoi))[0];
		  float roi_Mean = mean(curImg(brightnessRoi))[0];
		  float roi_brightness_difference = abs(roi_bg_mean - roi_Mean);
		  if (_config->debugGeneralDetail) {
			  cout << "BG-Cur Brightness ---->: " << to_string(roi_brightness_difference) << " <<----------------->> \n\n\n";
		  }
		  if (_config->bgsubtype == itms::BgSubType::BGS_CNT) {
			  cv::threshold(imgDifferenceBg, imgThreshBg, _config->dblMOGShadowThr, 255.0, CV_THRESH_BINARY); // 90 부터 낮게
		  }
		  else { // BGS_DIF
			  cv::threshold(imgDifferenceBg, imgThreshBg, max((double)_config->img_dif_th, min(150., roi_brightness_difference*5. / 4. + 15/*50*/ + _config->img_dif_th)), 255.0, CV_THRESH_BINARY);
		  }
		  if (_config->zoomBG) { // sangkny 2019. 04. 30 -> 05. 09 => 05. 19 ==>
			  // sangkny 2019. 04. 30 zooming and threshold for longDistance regions. For efficiency, I used resized imgDiffernceBG image			  
			  cv::Rect zROI(zPmin, zPmax);
			  cv::Rect ozRoi(ozPmin, ozPmax);// (zPmin*(1. / _config->scaleFactor), zPmax*(1. / _config->scaleFactor)); // original roi	=> init로		  
			  cv::Mat tmp = imgDifferenceBg(zROI).clone();	// 관심영역 절대좌표 축소된 이미지	=> 현재는 상관없네...	  
			  if (orgPreImage.empty())
				  orgPreImage = orgImage.clone();

			  cv::Mat orgPart = orgImage(ozRoi), orgPrePart = orgPreImage(ozRoi);
			  //cv::Mat tmpZoom;
			  
			  if (orgPart.channels() > 1) {
				  cv::cvtColor(orgPart, orgPart, CV_BGR2GRAY);
				  cv::cvtColor(orgPrePart, orgPrePart, CV_BGR2GRAY);
			  }
			  // background generation with original part image
			  cv::Mat imgOrgDifferenceBg /* bg generation*/, 
			  imgOrgDifThres /* bg generation thresholded*/, 
			  imgOrgPreDif /* pre-cur org*/, 
			  imgOrgPreDifThres /* pre-cur org difference thresholded */;

			  if (_config->bgsubtype == itms::BgSubType::BGS_CNT) {
				  pBgOrgSub->apply(orgPart, imgOrgDifferenceBg, 1. / (double)(_config->intNumBGRefresh));
				  cv::threshold(imgOrgDifferenceBg, imgOrgDifThres, _config->dblMOGShadowThr, 255.0, CV_THRESH_BINARY); // 90 부터 낮게
				  medianBlur(imgOrgDifThres, imgOrgDifThres, 3);
				  absdiff(orgPart, orgPrePart, imgOrgPreDif);
				  cv::threshold(imgOrgPreDif, imgOrgPreDifThres, _config->img_dif_th, 255.0, CV_THRESH_BINARY); // 
				  medianBlur(imgOrgPreDifThres, imgOrgPreDifThres, 3); // consider more if neccessary 
				  cv::dilate(imgOrgPreDifThres, imgOrgPreDifThres, structuringElement5x5);
				  //if (!bDayMode) {
					 // cv::dilate(imgOrgPreDifThres, imgOrgPreDifThres, structuringElement5x5); // sangkny 2019. 05. 22
				  //}
				  cv::erode(imgOrgPreDifThres, imgOrgPreDifThres, structuringElement5x5);
				  if (_config->debugShowImagesDetail && _config->debugSpecial) {
					  imshowBeforeAndAfter(orgPart, imgOrgDifferenceBg, "orgPart/ imgOrgDifBg before thresholding !!", 2);	
					  imshowBeforeAndAfter(imgOrgDifThres, imgOrgPreDifThres, " bg / pre-cur diff after thresholding !!", 2);
				  }
				  // birany image selection process // sangkny 20190519 to eliminated noisy binary image using contour 
				  // compute ROI 
				  cv::Mat org_mask(orgImage.size(), CV_8UC1, cv::Scalar(0, 0, 0)); // original size
				  cv::Mat org_road_mask_intersection = org_mask.clone();
				  cv::rectangle(org_mask, cv::Rect(this->ozPmin, this->ozPmax), 255, -1); // 흑생 바탕에 흰색으로 사각형을 채운다
				  cv::Mat org_road_mask;
				  cv::resize(road_mask, org_road_mask, org_mask.size());                  // original size 로 재 설정
				  cv::bitwise_and(org_mask, org_road_mask, org_road_mask_intersection);
				  if(0 && _config->debugShowImagesDetail){
					  imshowBeforeAndAfter(org_mask, org_road_mask_intersection, "org mask / its intersection mask", 2);
				  }
				  // now apply it to the ROI image using org_mask_intersection
				  cv::Mat org_ROI_mask = org_road_mask_intersection(ozRoi).clone();
				  assert(org_ROI_mask.size() == imgOrgDifThres.size());
				  cv::bitwise_and(imgOrgDifThres, org_ROI_mask, imgOrgDifThres);
				  cv::bitwise_and(imgOrgPreDifThres, org_ROI_mask, imgOrgPreDifThres);
				 
				  if (_config->debugShowImagesDetail && _config->debugSpecial) {					  
					  imshowBeforeAndAfter(imgOrgDifThres, imgOrgPreDifThres, " road_ROI_mask && bg / pre-cur diff after thresholding !!", 2);
				  }				  
#if 0
				  // select better one (less nosy foreground image using computing the complexity of the given foreground iamge)
				  cv::Mat imgOrgDifThresCopy = imgOrgDifThres.clone();
				  cv::Mat imgOrgPreDifThresCopy = imgOrgPreDifThres.clone();

				  std::vector<std::vector<cv::Point> > imgOrgcontours, imgOrgPrecontours;

				  cv::findContours(imgOrgDifThresCopy, imgOrgcontours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
				  cv::findContours(imgOrgPreDifThresCopy, imgOrgPrecontours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

				  if (0||(_config->debugShowImages && _config->debugShowImagesDetail)) {
					  drawAndShowContours(imgOrgDifThresCopy.size(), imgOrgcontours, "imgOrgcontours");
					  drawAndShowContours(imgOrgPreDifThresCopy.size(), imgOrgPrecontours, "imgOrgPrecontours");
					  cv::imshow("Org_ROI_mask", org_ROI_mask);
					  cv::waitKey(1);
				  }
				  // sort the contours according to their areas
				  std::sort(imgOrgcontours.begin(), imgOrgcontours.end(), compareContourAreasDes); // sorting descending order
				  std::sort(imgOrgPrecontours.begin(), imgOrgPrecontours.end(), compareContourAreasDes); 
				  // area sum upto the 4 largest ones
				  double imgOrgArea = 0, imgOrgPreArea = 0;

				  for(unsigned cnt_i=0; cnt_i<std::min((int)imgOrgcontours.size(), 4); cnt_i++){
					  imgOrgArea += cv::contourArea(imgOrgcontours.at(cnt_i));
				  }
				  for (unsigned cnt_i = 0; cnt_i<std::min((int)imgOrgPrecontours.size(), 4); cnt_i++) {
					  imgOrgPreArea += cv::contourArea(imgOrgPrecontours.at(cnt_i));
				  }
				  bool bselectBG = false;
				  if(imgOrgArea> imgOrgPreArea){
					  bselectBG = true;
				  }
				  else {
					  bselectBG = false;
				  }
				  // end select one using contours and its area
#endif
#if 0             // need to check it again, because it is not working in 20180911_114811_cam_0.avi (WWR vor vehicle
				  // bitwise_and and see if imgOrgDifThres is too big compared to imgOrgPreDifThres, then it is replaced with imgOrgDifThres
				  cv::Mat imgOrgAndPreDifThres;
				  cv::bitwise_and(imgOrgDifThres, imgOrgPreDifThres, imgOrgAndPreDifThres);
				  int imgOrgAndPreNonZero = cv::countNonZero(imgOrgAndPreDifThres);
				  if (imgOrgAndPreNonZero > 0) {
					  int imgOrgDifThresNonZero = cv::countNonZero(imgOrgDifThres);
					  int imgOrgPreDifThresNonZero = cv::countNonZero(imgOrgPreDifThres);
					  int imgOrgMaskNonZero = cv::countNonZero(org_ROI_mask);
					  float toOrgMaskThresPerc = 0.1, toBitAndThresTimes = 100,
						  orgTomaskRatio = (float)imgOrgDifThresNonZero / (float)imgOrgMaskNonZero;
					  if ((orgTomaskRatio > toOrgMaskThresPerc) && (imgOrgDifThresNonZero > toBitAndThresTimes * imgOrgAndPreNonZero)
					  &&(imgOrgDifThresNonZero > imgOrgPreDifThresNonZero)) {
						  imgOrgDifThres = imgOrgAndPreDifThres.clone();
					  }
				  }
#endif            // end selecting less noisy
				   
				  // 밤이면 전프레임과의 차이로만 된 영상을 전달해줌, later if(_config->img_dif_th > _config->nightBrightness_Th || BGS_DIF)으로 외부에서 대체
				  if(!bDayMode /* _config->img_dif_th > _config->nightBrightness_Th */){
					  imgOrgDifThres = imgOrgPreDifThres.clone();
				  }
				  //at day time, imgOrgDifThres will be preserved !!!
			  }
			  else { // BGS_DIF, this mode does not have imgOrgDifferenceBg 
				  absdiff(orgPart, orgPrePart, imgOrgPreDif);
				  cv::threshold(imgOrgPreDif, imgOrgPreDifThres, _config->img_dif_th, 255.0, CV_THRESH_BINARY); // 
				  medianBlur(imgOrgPreDifThres, imgOrgPreDifThres, 3); // consider more if neccessary 
				  if (0 && _config->debugShowImagesDetail && _config->debugSpecial) {
					  cv::imshow("tmpZoom before dilation", imgOrgPreDifThres);
				  }
				  //cv::dilate(tmpZoom, tmpZoom, structuringElement3x3); // MORP_CLOSE
				  //cv::erode(tmpZoom, tmpZoom, structuringElement3x3);  // MORP_CLOSE
				  if (_config->debugShowImagesDetail && _config->debugSpecial) {
					  imshowBeforeAndAfter(imgOrgPreDif, imgOrgPreDifThres, "imgOrgPre Dif / Thres", 2);
				  }
				  // imgOrgDifThres 는 원래 배경과의 차이인데 이 옵션은 전프레임과의 차이 모드 이므로
				  imgOrgDifThres = imgOrgPreDifThres.clone(); // 2019. 05. 22
			  }

			  cv::resize(imgOrgDifThres, tmp, cv::Size(), _config->scaleFactor, _config->scaleFactor); // 축소된 크기로 변환
			  // mix with imgThresBg only when daytime //낮에만 한다 2019. 05. 21			  
			  cv::Mat mask(imgDifferenceBg.size(), CV_8UC1, cv::Scalar(0, 0, 0));      // 축소된 크기 마스크 만듬
			  //mask(cv::Rect(static_cast<cv::Point>(zPmin), cv::Size(tmp.cols, tmp.rows))) = tmp; // it is not working, 이렇게 하면 동작 안함
			  cv::Rect copyRoi = cv::Rect(static_cast<cv::Point>(zPmin), cv::Size(tmp.cols, tmp.rows));
			  tmp.copyTo(mask(copyRoi));  // mask(축소된 전체영상)의 copyRoi 위치로 tmp(축소된 관심영역) 카피
			  cv::bitwise_and(mask, road_mask, mask); // road_mask 마스킹
			  // bitwise_or or bitwise_and 는 CNT or DIF 에 따라 결정을 해야 한다. 아래는  dif 만 고려한 형태, 그리고 낮과 밤일때 고려
			  // 밤일때는 DIF CNT모두 dif로 만들어진 mask만 고려, 낮에는 Zoom 상태 고려 해야함
			  if (!bDayMode) {
				  imgThreshBg = mask.clone();  // mask image will be the BG thresholded image 
			  }
			  else {
				  bitwise_or(imgThreshBg, mask, imgThreshBg);
			  }
			  // 축소된 bground 영상과 현재프레임 차이로 생긴 영상에 zoom(원래크기) 옵션에 의한 영상을 논리합을 하면 
			  // 축소된 imgThreshBg 영상의 잘안보이는 상단부분을 잘보이게 하는 효과가 있으나 자칫하면 노이즈가 더 낄 수 있으니 주의 바람.			  		  
			  if (_config->debugShowImagesDetail && _config->debugSpecial) {
				  cv::imshow("bitwise_or mask", mask);
				  //cv::waitKey(1);
				  imshowBeforeAndAfter(mask, tmp,"mask image(scaled size) tmp(zRoi)",2);				  
			  }
		  }
		  
	  }
	  // _config->zoomBG 에서 처리된 영상을 반영하기 위해서는 -->> imgThreshBg <<-- 영상을 참고해야 함.
	  cv::threshold(imgDifference, imgThresh, _config->img_dif_th/* +13(night) -3(day) */, 255.0, CV_THRESH_BINARY); // 축소된 영상 전후 차이	  	  
	  if (_config->debugShowImages && _config->debugShowImagesDetail) {
		  imshowBeforeAndAfter(imgThreshBg, imgThresh, "BG <<-- >> imgThresh before combine both", 2);
		  imshow("before erode dilation on imgThresh when DIF ", imgThresh);
	  }
     // then combine both at daytime and night time // 낮 밤 구분없이 섞어야 하는 것 같은데... 그것도 위 루푸 앞에서...
	  //if (bDayMode /*_config->img_dif_th < _config->nightBrightness_Th*//* 26 as of 20190510 */) {
		  // under daytime condition, please refer night threshold in matchCurFrameBlobsToExistingBlobs
		  cv::bitwise_or(imgThresh, imgThreshBg, imgThresh);
	  //}
	  //else { // under dark condition, only difference between pre and cur is used
		 // ; // doing for night condition
	  //}

	  
	  for (unsigned int i = 0; i < 1; i++) { // we need to this for the near distance regions, so before bit_wise_or with long distance regions
											 //if (_config->bgsubtype == BgSubType::BGS_CNT)
											 //	  cv::erode(imgThresh, imgThresh, structuringElement3x3);
		  if (_config->bgsubtype == BgSubType::BGS_DIF || (_config->bgsubtype == BgSubType::BGS_CNT /*&& !bDayMode*//*_config->img_dif_th<=_config->nightBrightness_Th*//* night time */)) {
			  if (_config->scaleFactor > 0.75) {
				  //cv::dilate(imgThresh, imgThresh, structuringElement3x3);
				  cv::dilate(imgThresh, imgThresh, structuringElement5x5);
				  cv::erode(imgThresh, imgThresh, structuringElement3x3);
			  }
			  else if (_config->scaleFactor > 0.5) {
				  //cv::dilate(imgThresh, imgThresh, structuringElement3x3);
				  cv::dilate(imgThresh, imgThresh, structuringElement7x7);
				  cv::erode(imgThresh, imgThresh, structuringElement5x5);
			  }
			  else {				  
				  if (!bDayMode) {
					  cv::dilate(imgThresh, imgThresh, structuringElement5x5);					  
				  }				  
				  cv::dilate(imgThresh, imgThresh, structuringElement5x5);
				  cv::erode(imgThresh, imgThresh, structuringElement5x5);
			  }
		  }
	  }
	  // after morphology we deed to threshold again otherwise, contours will have some connection with neighbors
	  cv::threshold(imgThresh, imgThresh, 200, 255, CV_THRESH_BINARY); // sangkny 2019. 05. 22
	  	  
	  if (_config->debugShowImages && _config->debugShowImagesDetail) {
		  imshow("after erode dilation and combine", imgThresh);
		  //cv::waitKey(1);
		  itms::imshowBeforeAndAfter(imgThreshBg, imgThresh, " imgThreshBg (after morphology on imgThresh and combine (bit_OR)", 2);
		  //itms::imshowBeforeAndAfter(imgXor, imgXor, " (bit_XOR)", 2);
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
	  float f_scale_factor = _config->scaleFactor;
	  float fmin_area = 10 * f_scale_factor, 
		  fmin_obj_width = 5* f_scale_factor, 
		  fmin_obj_height = 5*f_scale_factor, 
		  fmin_obj_diagSize = 8*f_scale_factor;
	  for (auto &convexHull : convexHulls) {
		  Blob possibleBlob(convexHull);

		  if (possibleBlob.currentBoundingRect.area() > (int)fmin_area &&
			  possibleBlob.dblCurrentAspectRatio > 0.02 &&
			  possibleBlob.dblCurrentAspectRatio < 6.0 &&
			  possibleBlob.currentBoundingRect.width > (int)fmin_obj_width &&
			  possibleBlob.currentBoundingRect.height > (int)fmin_obj_height &&
			  possibleBlob.dblCurrentDiagonalSize > (double)fmin_obj_diagSize &&
			  (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.30) {
			  //  new approach according to 
			  // 1. distance, 2. correlation within certain range
			  std::vector<cv::Point2f> blob_ntPts;
			  blob_ntPts.push_back(Point2f(possibleBlob.centerPositions.back()));			  
			  cv::Rect roi_rect = possibleBlob.currentBoundingRect;
			  float blobncc = 0;			  
			  // bg image
			  // currnt image
			  // blob correlation
			  //blobncc = getNCC(*_config, BGImage(roi_rect), curImg(roi_rect), Mat(), _config->match_method, _config->use_mask); 
			  // background image need to be updated periodically 
			  // option double d3 = matchShapes(BGImage(roi_rect), imgFrame2Copy(roi_rect), CONTOURS_MATCH_I3, 0);
			  float realDistance = 0;
			  if (checkIfBlobInBoundaryAndDistance(*_config, possibleBlob, _config->Boundary_ROI_Pts, realDistance) 
			  &&((blobncc = getNCC(*_config, BGImage(roi_rect), curImg(roi_rect), Mat(), _config->match_method, _config->use_mask)) <= abs(_config->BlobNCC_Th))
			   )
			  {// check the correlation with bgground, object detection/classification				  
				  if (_config->debugGeneral && _config->debugGeneralDetail) {
					  cout << "Candidate object:" << blob_ntPts.back() << "(W,H)" << cv::Size(roi_rect.width, roi_rect.height) << " is in(" << to_string(realDistance / 100.) << ") Meters ~(**)\n";					  
				  }
				  ObjectClass objclass;
				  float classProb = 0.f;
				  classifyObjectWithDistanceRatio(*_config, possibleBlob, realDistance / 100, objclass, classProb);
				  
				  if (_config->debugSaveFile) {
					  out_object_class << to_string(possibleBlob.oc) << "\t" << to_string(possibleBlob.currentBoundingRect.width) << "\t" << to_string(possibleBlob.currentBoundingRect.height) << "\t" << to_string(realDistance / 100.) << "\t" << to_string(possibleBlob.oc_prob) << endl;
				  }

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
					  if(objclass != ObjectClass::OC_OTHER) // sankgny 2019. 03. 30
						  currentFrameBlobs.push_back(possibleBlob);
				  }

			  }
		  }
		  //else {
			 // // cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()
			 // cout << "---------- > contourArea/rectArea:" << cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()<<"<<---------------" << endl;
		  //}
	  }

	  if (_config->debugShowImages && _config->debugShowImagesDetail) {
		  // all of the currentFrameBlobs at this stage have 1 visible count yet. 
		  drawAndShowContours(*_config, imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");
	  }
	  // merge assuming
	  // blobs are in the ROI because of ROI map
	  // 남북 이동시는 가로가 세로보다 커야 한다.	  
	  mergeBlobsInCurrentFrameBlobs(*_config, currentFrameBlobs);			// need to consider the distance	  
	  if (m_collectPoints) {
		  if (_config->debugShowImages && _config->debugShowImagesDetail) {
			  drawAndShowContours(*_config, imgThresh.size(), currentFrameBlobs, "before merging predictedBlobs into currentFrameBlobs");			  
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
			  mergeBlobsInCurrentFrameBlobsWithPredictedBlobs(currentFrameBlobs, predictedBlobs);
		  }
	  }
      if (_config->split_blob) {
          SplitBlobsInCurrentFrameBlobs(*_config, currentFrameBlobs, curImg);
      }
	  if (_config->debugShowImages && _config->debugShowImagesDetail) {
		  drawAndShowContours(*_config, imgThresh.size(), currentFrameBlobs, "after merging and spliting currentFrameBlobs");		  
	  }
	  if (blnFirstFrame == true) {
		  for (auto &currentFrameBlob : currentFrameBlobs) {
			  blobs.push_back(currentFrameBlob);
		  }
	  }
	  else {
		  _config->trackid = _config->trackid % _config->maxTrackIds;
		  matchCurrentFrameBlobsToExistingBlobs(*_config, orgImage, preImg/* imgFrame1 */, curImg/* imgFrame2 */, blobs, currentFrameBlobs, _config->trackid);
		  //matchExistingBlobsToCurrentFrameBlobs(*_config, preImg/* imgFrame1 */, curImg/* imgFrame2 */, blobs, currentFrameBlobs, _config->trackid); // 실패..
	  }
	  //imgFrame2Copy = imgFrame2.clone();          // color get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

	  bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheBoundary(*_config, blobs,/* debugImg,*/ _config->ldirection, _config->Boundary_ROI_Pts);

	  if (_config->debugShowImages) {
		  cv::Mat debugImg = curImg.clone();
		  if (debugImg.channels() < 3)
			  cvtColor(debugImg, debugImg, cv::COLOR_GRAY2BGR);
		  if (_config->debugShowImagesDetail)
			  drawAndShowContours(*_config, imgThresh.size(), blobs, "All imgBlobs");

		  // draw blob information
		  drawBlobInfoOnImage(*_config, blobs, debugImg);  // blob(tracked) information
		  // draw lanes
		  drawRoadRoiOnImage(_config->Road_ROI_Pts, debugImg);
		  if (_config->debugShowImagesDetail) {
			  //draw brightness checking area
			  if (brightnessRoi.area() > 0) {
				  cv::rectangle(debugImg, brightnessRoi, SCALAR_CYAN, 2, 8, 0);
			  }
		  }

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
		  //cv::waitKey(1);
	  }

	  // now we prepare for the next iteration
	  currentFrameBlobs.clear();

	  preImg = curImg.clone();           // move frame 1 up to where frame 2 is	  
	  orgPreImage = curImg1.clone();	 // sangkny 2019/05/07
	  // generate background image
	  if(_config->bGenerateBG && _config->intNumBGRefresh > 0 && (_config->bgsubtype == BgSubType::BGS_DIF)){
		  
		  double learningRate = (double)1./(double)(_config->intNumBGRefresh);
		  Mat curTmp = curImg.clone();
		  //cout << "accType before apply: " + type2str(accmImage.type()) << endl;
		  //curTmp.convertTo(curTmp, CV_8UC1);
		  //accmImage.convertTo(accmImage, CV_8UC1);
		  //itms::imshowBeforeAndAfter(curTmp, accmImage, "cur/ acc image", 2);
		  //pBgSub->apply(curTmp, accmImage, learningRate); // slower than the below method, accmImage will be the difference between bgimage and current image
		  //cout << "accType after apply: " + type2str(accmImage.type()) << endl;
		  //cv::waitKey(1);
		  //pBgSub->getBackgroundImage(accmImage);
		  //cv::Mat tmp;//(curImg.size(), CV_32FC1);
		  //curTmp.convertTo(curTmp, CV_32FC1);
		  //accmImage.convertTo(accmImage, CV_32FC1);
		  //accumulateWeighted(curTmp, accmImage, learningRate, road_mask);// 같은 타입으로 컨버전 필요
		  
		  addWeighted(curImg, learningRate, accmImage, 1.-learningRate, 0, accmImage); // 같은 타입으로 컨버전 없어도 됨..
		  //convertScaleAbs(accmImage, accmImage);
		  //accmImage.convertTo(accmImage, CV_8UC1);
		  //convertScaleAbs(accmImage, accmImage,255.0);
		  
		  setBGImage(accmImage);
		  if(_config->debugShowImagesDetail){
			  cv::imshow("generating BG", getBGImage());			
		  }
	  }

	  // end generate backgroudimage 
	  blnFirstFrame = false;
	  if(checkObjectStatus(*_config, curImg, blobs, _itmsRes)){
		  if(_config->debugGeneralDetail)
			  cout<< "# of events: " << _itmsRes.objRect.size() <<" has been occurred (!)<!>" << endl;
	  }	  
	  return true;
  }// end process


  // ITMS API NATIVE CLASS
  ITMSAPINativeClass::ITMSAPINativeClass()
	  :isInitialized(false) {
	  //Init(); // shoud not initialize here because it can not be used in CLI or C# class
  }


  ITMSAPINativeClass::~ITMSAPINativeClass()
  {
	  std::cout <<" Leaing ITMSAPI Native Class \r\n";
  }
  
  int ITMSAPINativeClass::Init()
  {
	  int ref = -1;
	  if (isInitialized) {
		  std::cout << "ITMS API configuration has been already done and thus skipped !!\n";
		  return 1;
	  }

	  std::cout << "Using OpenCV" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << " in ITMSAPINativeClass. "<<std::endl;

	  int trackId = conf.trackid;          // unique object id, you can set or get the track id
	  int maxTrackId = conf.maxTrackIds;   // now two variable tracId and maxTrackId are not used 

	  std::cout << ".... configurating ...\n";
	  if (loadConfig(conf)) {
		  std::cout << " configuration is done !!\n\n";

		  ref = 0;
	  }
	  else {
		  std::cout << " configuration is not finished. However, default values are used instead!! Please double check the all files and values correctly (!)(!)\n";
	  }


	  // -------------------------------------- HOW TO USE THE API --------------------------------------------
	  // construct an instance 
	  itmsFncs = std::make_unique<itmsFunctions>(&conf); // create instance and initialize
	  itmsres = std::make_unique<ITMSResult>();
	  // --------------------------------------------------------------------------------------------------------

	  pFrame = Mat::zeros(Size(1920, 1080), CV_8UC3);
	  //pFrame = Mat::zeros(Size(1280, 720), CV_8UC3);
	  isInitialized = true;
	  return ref;
  }
  
  int ITMSAPINativeClass::ResetAndProcessFrame(int iCh, unsigned char * pImage, int lSize) // reset and process Frame
  {
	  int ref = -1;

	  if (!isInitialized)
		  Init();

	  memcpy(pFrame.data, pImage, lSize); // converting to mat 
	  //cv::imshow("main window", pFrame);
	  //cv::waitKey(1);


	  itmsres->reset();
	  itmsFncs->process(pFrame, *itmsres); // with current Frame 

	  ref = 0;
	  return ref;
  }
  int ITMSAPINativeClass::ResetAndProcessFrame(const cv::Mat& curImg1) // reset and process Frame
  {
	  int ref = -1;

	  if (!isInitialized)
	  		  Init();			  

	  itmsres->reset();
	  itmsFncs->process(curImg1, *itmsres); // with current Frame 

	  ref = 0;
	  return ref;
  }
  /*unique_ptr<itms::ITMSResult> ITMSAPINativeClass::getResult(void)
  { 
	  return std::move(this->itmsres);
  }*/
  std::vector<std::pair<int, int>> ITMSAPINativeClass::getObjectStatus(void)
  {
	  return (this->itmsres->objStatus);
  }
  std::vector<std::pair<int, int>> ITMSAPINativeClass::getObjectClass(void)
  {
	  return (this->itmsres->objClass);
  }
  std::vector<cv::Rect> ITMSAPINativeClass::getObjectRect(void)
  {
	  return (this->itmsres->objRect);
  }
  std::vector<track_t> ITMSAPINativeClass::getObjectSpeed(void)
  {
	  return (this->itmsres->objSpeed);
  }
  
  // linesegment class related 

  LineSegment::LineSegment()
  {
	  init(0, 0, 0, 0);
  }

  LineSegment::LineSegment(Point p1, Point p2)
  {
	  init(p1.x, p1.y, p2.x, p2.y);
  }

  LineSegment::LineSegment(int x1, int y1, int x2, int y2)
  {
	  init(x1, y1, x2, y2);
  }

  void LineSegment::init(int x1, int y1, int x2, int y2)
  {
	  this->p1 = Point(x1, y1);
	  this->p2 = Point(x2, y2);

	  if (p2.x - p1.x == 0)
		  this->slope = 0.00000000001;
	  else
		  this->slope = (float)(p2.y - p1.y) / (float)(p2.x - p1.x);

	  this->length = distanceBetweenPoints(p1, p2);

	  this->angle = angleBetweenPoints(p1, p2);
  }

  bool LineSegment::isPointBelowLine(Point tp)
  {
	  return ((p2.x - p1.x)*(tp.y - p1.y) - (p2.y - p1.y)*(tp.x - p1.x)) > 0;
  }

  float LineSegment::getPointAt(float x)
  {
	  return slope * (x - p2.x) + p2.y;
  }

  float LineSegment::getXPointAt(float y)
  {
	  float y_intercept = getPointAt(0);
	  return (y - y_intercept) / slope;
  }

  Point LineSegment::closestPointOnSegmentTo(Point p)
  { // inner product (dot product)
	  float top = (p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y)*(p2.y - p1.y);

	  float bottom = distanceBetweenPoints(p2, p1);
	  bottom = bottom * bottom;

	  float u = top / bottom;

	  float x = p1.x + u * (p2.x - p1.x);
	  float y = p1.y + u * (p2.y - p1.y);

	  return Point(x, y);
  }

  Point LineSegment::intersection(LineSegment line)
  {
	  float c1, c2;
	  float intersection_X = -1, intersection_Y = -1;

	  c1 = p1.y - slope * p1.x; // which is same as y2 - slope * x2

	  c2 = line.p2.y - line.slope * line.p2.x; // which is same as y2 - slope * x2

	  if ((slope - line.slope) == 0)
	  {
		  //std::cout << "No Intersection between the lines" << endl;
	  }
	  else if (p1.x == p2.x)
	  {
		  // Line1 is vertical
		  return Point(p1.x, line.getPointAt(p1.x));
	  }
	  else if (line.p1.x == line.p2.x)
	  {
		  // Line2 is vertical
		  return Point(line.p1.x, getPointAt(line.p1.x));
	  }
	  else
	  {
		  intersection_X = (c2 - c1) / (slope - line.slope);
		  intersection_Y = slope * intersection_X + c1;
	  }

	  return Point(intersection_X, intersection_Y);
  }

  Point LineSegment::midpoint()
  {
	  // Handle the case where the line is vertical
	  if (p1.x == p2.x)
	  {
		  float ydiff = p2.y - p1.y;
		  float y = p1.y + (ydiff / 2);
		  return Point(p1.x, y);
	  }
	  float diff = p2.x - p1.x;
	  float midX = ((float)p1.x) + (diff / 2);
	  int midY = getPointAt(midX);

	  return Point(midX, midY);
  }

  LineSegment LineSegment::getParallelLine(float distance)
  {
	  float diff_x = p2.x - p1.x;
	  float diff_y = p2.y - p1.y;
	  float angle = atan2(diff_x, diff_y);
	  float dist_x = distance * cos(angle);
	  float dist_y = -distance * sin(angle);

	  int offsetX = (int)round(dist_x);
	  int offsetY = (int)round(dist_y);

	  LineSegment result(p1.x + offsetX, p1.y + offsetY,
		  p2.x + offsetX, p2.y + offsetY);

	  return result;
  }
  
  
  // car counting related function implementation
  ///
  /// \brief CarsCounting::CarsCounting
  /// \param parser
  ///
  CarsCounting::CarsCounting(const cv::CommandLineParser& parser)
	  :
	  m_showLogs(true),
	  m_fps(25),
	  m_useLocalTracking(false), // local tracking control
	  m_isTrackerInitialized(false),
	  m_startFrame(0),
	  m_endFrame(0),
	  m_finishDelay(0)
  {
	  m_inFile = parser.get<std::string>(0);
	  m_outFile = parser.get<std::string>("out");
	  m_showLogs = parser.get<int>("show_logs") != 0;
	  m_startFrame = parser.get<int>("start_frame");
	  m_endFrame = parser.get<int>("end_frame");
	  m_finishDelay = parser.get<int>("end_delay");

	  m_colors.push_back(cv::Scalar(255, 0, 0));
	  m_colors.push_back(cv::Scalar(0, 255, 0));
	  m_colors.push_back(cv::Scalar(0, 0, 255));
	  m_colors.push_back(cv::Scalar(255, 255, 0));
	  m_colors.push_back(cv::Scalar(0, 255, 255));
	  m_colors.push_back(cv::Scalar(255, 0, 255));
	  m_colors.push_back(cv::Scalar(255, 127, 255));
	  m_colors.push_back(cv::Scalar(127, 0, 255));
	  m_colors.push_back(cv::Scalar(127, 0, 127));
  }

  CarsCounting::CarsCounting(Config * config)
	  :
	  m_showLogs(true),
	  m_fps(25),
	  m_useLocalTracking(false), // local tracking control
	  m_isTrackerInitialized(false),
	  m_startFrame(0),
	  m_endFrame(0),
	  m_finishDelay(0)
  {
	  m_inFile = "";// parser.get<std::string>(0);
	  m_outFile = "";// parser.get<std::string>("out");
	  m_showLogs = true;// parser.get<int>("show_logs") != 0;
	  m_startFrame = 0; // parser.get<int>("start_frame");
	  m_endFrame = 0;// parser.get<int>("end_frame");
	  m_finishDelay = 0;// parser.get<int>("end_delay");

	  m_colors.push_back(cv::Scalar(255, 0, 0));
	  m_colors.push_back(cv::Scalar(0, 255, 0));
	  m_colors.push_back(cv::Scalar(0, 0, 255));
	  m_colors.push_back(cv::Scalar(255, 255, 0));
	  m_colors.push_back(cv::Scalar(0, 255, 255));
	  m_colors.push_back(cv::Scalar(255, 0, 255));
	  m_colors.push_back(cv::Scalar(255, 127, 255));
	  m_colors.push_back(cv::Scalar(127, 0, 255));
	  m_colors.push_back(cv::Scalar(127, 0, 127));

	  if (!config->isLoaded) {
		  isConfigFileLoaded = false;
	  }
	  else {
		  _config = config;
		  isConfigFileLoaded = true;
		  Init();
	  }

  }
 
  ///
  /// \brief CarsCounting::~CarsCounting
  ///
  CarsCounting::~CarsCounting()
  {
	  /*if (m_detector) {
	  m_detector.release();
	  m_detector = nullptr;
	  }
	  if (m_tracker) {
	  m_tracker.release();
	  m_tracker = nullptr;
	  }*/
  }

  bool CarsCounting::Init(void) {
	  //pBgSub = cv::bgsubcnt::createBackgroundSubtractorCNT(fps, true, fps * 60);
	  //pBgSub = createBackgroundSubtractorMOG2();
	  //blobs.clear();
	  pastBrightnessLevels.clear();

	  brightnessRoi = _config->AutoBrightness_Rect;
	  m_collectPoints = _config->m_useLocalTracking;
	  blnFirstFrame = true;
	  if (existFileTest(_config->BGImagePath)) {
		  BGImage = cv::imread(_config->BGImagePath);
		  if (!BGImage.empty()) {
			  if (BGImage.channels() > 1)
				  cv::cvtColor(BGImage, BGImage, cv::COLOR_BGR2GRAY);
			  cv::resize(BGImage, BGImage, cv::Size(), _config->scaleFactor, _config->scaleFactor);
			  // sangkny 2019. 02. 09
			  cv::GaussianBlur(BGImage, BGImage, cv::Size(5, 5), 0);
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
				  //cv::waitKey(1);
			  }
		  }
	  }
	  else {
		  if (_config->debugGeneral) {
			  cout << "Please check the background file : " << _config->BGImagePath << endl;
			  cout << "The background will be current frame <!><!>.\n";
		  }
		  // alternative frame will be the previous frame at processing

	  }
	  return isInitialized = true;
  }
  ///
  /// \brief CarsCounting::Process
  ///
  void CarsCounting::Process()
  {
	  cv::VideoWriter writer;

	  cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

	  int k = 0;

	  double freq = cv::getTickFrequency();

	  int64 allTime = 0;

	  bool manualMode = false;
	  int framesCounter = m_startFrame + 1;

	  cv::VideoCapture capture;
	  if (m_inFile.size() == 1)
	  {
		  capture.open(atoi(m_inFile.c_str()));
	  }
	  else
	  {
		  capture.open(m_inFile);
	  }
	  if (!capture.isOpened())
	  {
		  std::cerr << "Can't open " << m_inFile << std::endl;
		  return;
	  }
	  capture.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);

	  m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));

	  //m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS)); // is it required for 2 times ?

	  float scaleFactor = 0.5;
	  using namespace cv;
	  std::vector<cv::Point> road_roi_pts;
	  std::vector<std::vector<cv::Point>> Road_ROI_Pts; // sidewalks and carlanes

														//20180912_112338
														// side walk1
	  road_roi_pts.push_back(Point(932.75, 100.25)*scaleFactor);
	  road_roi_pts.push_back(Point(952.25, 106.25)*scaleFactor);
	  road_roi_pts.push_back(Point(434.75, 1055.75)*scaleFactor);
	  road_roi_pts.push_back(Point(235.25, 1054.25)*scaleFactor);
	  Road_ROI_Pts.push_back(road_roi_pts);
	  road_roi_pts.clear();
	  // car lane
	  road_roi_pts.push_back(Point(949.25, 104.75)*scaleFactor);
	  road_roi_pts.push_back(Point(1015.25, 103.25)*scaleFactor);
	  road_roi_pts.push_back(Point(1105.25, 1048.25)*scaleFactor);
	  road_roi_pts.push_back(Point(416.75, 1057.25)*scaleFactor);
	  Road_ROI_Pts.push_back(road_roi_pts);
	  road_roi_pts.clear();
	  // side walk2
	  road_roi_pts.push_back(Point(1009.25, 101.75)*scaleFactor);
	  road_roi_pts.push_back(Point(1045.25, 98.75)*scaleFactor);
	  road_roi_pts.push_back(Point(1397.75, 1052.75)*scaleFactor);
	  road_roi_pts.push_back(Point(1087.25, 1049.75)*scaleFactor);
	  Road_ROI_Pts.push_back(road_roi_pts);
	  road_roi_pts.clear();


	  cv::Mat colorFrame;
	  cv::UMat grayFrame;

	  cv::Mat road_mask;

	  for (;;)
	  {
		  capture >> colorFrame;
		  if (colorFrame.empty())
		  {
			  std::cerr << "Frame is empty!" << std::endl;
			  break;
		  }
		  resize(colorFrame, colorFrame, cv::Size(), scaleFactor, scaleFactor);
		  cv::cvtColor(colorFrame, grayFrame, cv::COLOR_BGR2GRAY);
		  if (framesCounter == m_startFrame + 1) {
			  // road mask generation
			  road_mask = cv::Mat::zeros(grayFrame.size(), grayFrame.type());
			  for (int ir = 0; ir < Road_ROI_Pts.size(); ir++)
				  fillConvexPoly(road_mask, Road_ROI_Pts.at(ir).data(), Road_ROI_Pts.at(ir).size(), Scalar(255, 255, 255), 8);

			  if (road_mask.channels() > 1)
				  cvtColor(road_mask, road_mask, CV_BGR2GRAY);
			  if (0) {
				  imshow("road mask", road_mask);
				  //cv::waitKey(1);
			  }
		  }
		  if (!road_mask.empty()) // only one time setting
			  bitwise_and(road_mask, grayFrame, grayFrame);

		  if (!m_isTrackerInitialized)
		  {
			  m_isTrackerInitialized = InitTracker(grayFrame);
			  if (!m_isTrackerInitialized)
			  {
				  std::cerr << "Tracker initilize error!!!" << std::endl;
				  break;
			  }
		  }

		  if (!writer.isOpened())
		  {
			  writer.open(m_outFile, cv::VideoWriter::fourcc('H', 'F', 'Y', 'U'), m_fps, colorFrame.size(), true);
		  }

		  int64 t1 = cv::getTickCount();

		  cv::UMat clFrame;
		  if (!GrayProcessing() || !m_tracker->GrayFrameToTrack())
		  {
			  clFrame = colorFrame.getUMat(cv::ACCESS_READ);
		  }

		  m_detector->Detect(GrayProcessing() ? grayFrame : clFrame);

		  const regions_t& regions = m_detector->GetDetects();
		  // 여기서 걸러내면 됨...

		  m_tracker->Update(regions, m_tracker->GrayFrameToTrack() ? grayFrame : clFrame, m_fps);


		  int64 t2 = cv::getTickCount();

		  allTime += t2 - t1;
		  int currTime = cvRound(1000 * (t2 - t1) / freq);

		  DrawData(colorFrame, framesCounter, currTime);

		  cv::imshow("Video", colorFrame);

		  int waitTime = manualMode ? 0 : std::max<int>(1, cvRound(1000 / m_fps - currTime));
		  k = cv::waitKey(waitTime);
		  if (k == 'm' || k == 'M')
		  {
			  manualMode = !manualMode;
		  }
		  else if (k == 27)
		  {
			  break;
		  }

		  if (writer.isOpened())
		  {
			  writer << colorFrame;
		  }

		  ++framesCounter;
		  if (m_endFrame && framesCounter > m_endFrame)
		  {
			  std::cout << "Process: riched last " << m_endFrame << " frame" << std::endl;
			  break;
		  }
	  }

	  std::cout << "work time = " << (allTime / freq) << std::endl;
	  cv::waitKey(m_finishDelay);
  }
  bool CarsCounting::process(const cv::Mat& colorFrame, ITMSResult& _itmsRes)
  {
	  cv::Mat _colorFrame = colorFrame.clone();

	  m_fps = (float)_config->fps;

	  //m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS)); // is it required for 2 times ?

	  float scaleFactor = _config->scaleFactor;
	  
	  cv::UMat grayFrame;

	  //cv::Mat road_mask;
	  if (!isInitialized) {
		  cout << "itmsFunctions is not initialized (!)(!)\n";
		  return false;
	  }
	  if (_colorFrame.empty())
		{
			std::cerr << "Frame is empty!" << std::endl;
			return false;
		
		}
		
	  resize(_colorFrame, _colorFrame, cv::Size(), scaleFactor, scaleFactor);
		
	  if (_colorFrame.channels() > 1) {
			cv::cvtColor(_colorFrame, grayFrame, cv::COLOR_BGR2GRAY);
		}else{
			_colorFrame.copyTo(grayFrame);
		}		
	  
	  if (preImg.empty()) {
		  preImg = _colorFrame.clone(); // resized frame
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
			  //cv::waitKey(1);
		  }
	  }

	  // if needs to adjust img_dif_th
	  if (_config->isAutoBrightness) {
		  //compute the roi brightness and then adjust the img_dif_th withe the past max_past_frames 
		  float roiMean = mean(_colorFrame(brightnessRoi)/*currentGray roi*/)[0];
		  if (pastBrightnessLevels.size() >= _config->max_past_frames_autoBrightness) // the size of vector is max_past_frames
			  pop_front(pastBrightnessLevels, pastBrightnessLevels.size() - _config->max_past_frames_autoBrightness + 1); // keep the number of max_past_frames
																														  //pop_front(pastBrightnessLevels); // remove an elemnt from the front of 			
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
	  		
	  if (!road_mask.empty()) { // only one time setting
		  bitwise_and(road_mask, grayFrame, grayFrame);
	  }
	  		
	  if (!m_isTrackerInitialized)	
	  {
			m_isTrackerInitialized = InitTracker(grayFrame);
			if (!m_isTrackerInitialized)
			{
				std::cerr << "Tracker initilize error!!!" << std::endl;
				return false;
			}
		}		

		cv::UMat clFrame;
		if (!GrayProcessing() || !m_tracker->GrayFrameToTrack())
		{
			clFrame = _colorFrame.getUMat(cv::ACCESS_READ);
		}

		m_detector->Detect(GrayProcessing() ? grayFrame : clFrame);
		BGImage = m_detector->GetBackGround().getMat(cv::ACCESS_READ).clone();
		
		regions_t& regions = m_detector->GetDetects();

		// ------------------- 여기서 걸러내면 됨...
		regions_t::iterator region = regions.begin();
		while(region != regions.end()){
			// 1. distance, 2. correlation within certain range			
			std::vector<cv::Point2f> blob_ntPts;
			cv::Point2f center(region->m_rect.x + 0.5f * region->m_rect.width, region->m_rect.y + 0.5f * region->m_rect.height);
			blob_ntPts.push_back(center);
			cv::Rect roi_rect = region->m_rect;
			float blobncc = 0;
			// bg image
			// currnt image
			// blob correlation
			//cv::Mat img = grayFrame.getMat(cv::ACCESS_READ).clone();
			BGImage;
		
			blobncc = getNCC(*_config, BGImage(roi_rect), grayFrame.getMat(cv::ACCESS_READ)(roi_rect), Mat(), _config->match_method, _config->use_mask);
			// background image need to be updated periodically 
			// option double d3 = matchShapes(BGImage(roi_rect), imgFrame2Copy(roi_rect), CONTOURS_MATCH_I3, 0);
			if (blobncc <= abs(_config->BlobNCC_Th)
				&& checkIfPointInBoundary(*_config, blob_ntPts.back(), _config->Boundary_ROI_Pts)
				/*realDistance >= 100 && realDistance <= 19900*//* distance constraint */)
			{// check the correlation with bgground, object detection/classification
				float realDistance = getDistanceInMeterFromPixels(blob_ntPts, _config->transmtxH, _config->lane_length, false);
				if (_config->debugGeneral && _config->debugGeneralDetail) {
					cout << "Candidate object:" << blob_ntPts.back() << "(W,H)" << cv::Size(roi_rect.width, roi_rect.height) << " is in(" << to_string(realDistance / 100.) << ") Meters ~(**)\n";
				}

				//ObjectClass objclass;
				//float classProb = 0.f;
				//classifyObjectWithDistanceRatio(*_config, possibleBlob, realDistance / 100, objclass, classProb);
				//// update the blob info and add to the existing blobs according to the classifyObjectWithDistanceRatio function output
				//// verify the object with cascade object detection
				//if (classProb > 0.79 /* 1.0 */) {
				//	currentFrameBlobs.push_back(possibleBlob);
				//}
				//else if (classProb > 0.5f) {

				//	// check with a ML-based approach
				//	//float scaleRect = 1.5;
				//	//Rect expRect = expandRect(roi_rect, scaleRect*roi_rect.width, scaleRect*roi_rect.height, imgFrame2Copy.cols, imgFrame2Copy.rows);

				//	//if (possibleBlob.oc == itms::ObjectClass::OC_VEHICLE) {
				//	//	// verify it
				//	//	std::vector<cv::Rect> cars;
				//	//	detectCascadeRoiVehicle(imgFrame2Copy, expRect, cars);
				//	//	if (cars.size())
				//	//		possibleBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
				//	//	//else													// commented  :  put the all candidates commented out : it does not put the object in the candidates
				//	//	//	continue;
				//	//}
				//	//else if (possibleBlob.oc == itms::ObjectClass::OC_HUMAN) {
				//	//	// verify it
				//	//	std::vector<cv::Rect> people;
				//	//	detectCascadeRoiHuman(imgFrame2Copy, expRect, people);
				//	//	if (people.size())
				//	//		possibleBlob.oc_prob = 1.0;							// set the probability to 1, and it goes forever after.
				//	//	//else
				//	//	//	continue;
				//	//}
				//	//else {// should not com in this loop (OC_OTHER)
				//	//	int kkk = 0;
				//	//}				
				//	if (objclass != ObjectClass::OC_OTHER) // sankgny 2019. 03. 30
				//		currentFrameBlobs.push_back(possibleBlob);
				//}

			}
			else { // delete
				region = regions.erase(region);
				continue;
			}


			++region;
		} // end while loop
		// -----------------------------------
		m_tracker->Update(regions, m_tracker->GrayFrameToTrack() ? grayFrame : clFrame, m_fps);

		
		if (_config->debugGeneralDetail) {
			DrawData(_colorFrame, 0/*framesCounter*/, 0/*currTime*/);
		}
		if (_config->debugShowImagesDetail) {
			cv::imshow("Video", _colorFrame);
			//cv::waitKey(1);
		}
		
		preImg = _colorFrame.clone();// resized current Image

	  return true;
  }
  ///
  /// \brief CarsCounting::GrayProcessing
  /// \return
  ///
  bool CarsCounting::GrayProcessing() const
  {
	  return true;
  }

  ///
  /// \brief CarsCounting::DrawTrack
  /// \param frame
  /// \param resizeCoeff
  /// \param track
  /// \param drawTrajectory
  /// \param isStatic
  ///
  void CarsCounting::DrawTrack(cv::Mat frame,
	  int resizeCoeff,
	  const CTrack& track,
	  bool drawTrajectory,
	  bool isStatic
  )
  {
	  auto ResizeRect = [&](const cv::Rect& r) -> cv::Rect
	  {
		  return cv::Rect(resizeCoeff * r.x, resizeCoeff * r.y, resizeCoeff * r.width, resizeCoeff * r.height);
	  };
	  auto ResizePoint = [&](const cv::Point& pt) -> cv::Point
	  {
		  return cv::Point(resizeCoeff * pt.x, resizeCoeff * pt.y);
	  };

	  if (isStatic)
	  {
		  std::ostringstream os;
		  os << int(track.m_trackID);
#if (CV_VERSION_MAJOR >= 4)
		  cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
		  putText(frame, "s: " + os.str(), cv::Point(ResizeRect(track.GetLastRect()).x, ResizeRect(track.GetLastRect()).y - 1), cv::FONT_HERSHEY_SIMPLEX,
			  0.5, cv::Scalar(255, 0, 255), 1);
#else
		  cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(255, 0, 255), 2, CV_AA);
		  putText(frame, "s: " + os.str(), cv::Point(ResizeRect(track.GetLastRect()).x, ResizeRect(track.GetLastRect()).y - 1), cv::FONT_HERSHEY_SIMPLEX,
			  0.5, cv::Scalar(255, 0, 255), 1);
#endif
	  }
	  else
	  {
		  std::ostringstream os;
		  os << int(track.m_trackID);
#if (CV_VERSION_MAJOR >= 4)
		  cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
		  putText(frame, "a: " + os.str(), cv::Point(ResizeRect(track.GetLastRect()).x, ResizeRect(track.GetLastRect()).y - 1), cv::FONT_HERSHEY_SIMPLEX,
			  0.5, cv::Scalar(0, 255, 0), 1);
#else
		  cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(0, 255, 0), 1, CV_AA);

		  putText(frame, "a: " + os.str(), cv::Point(ResizeRect(track.GetLastRect()).x, ResizeRect(track.GetLastRect()).y - 1), cv::FONT_HERSHEY_SIMPLEX,
			  0.5, cv::Scalar(0, 255, 0), 1);
#endif
	  }

	  if (drawTrajectory)
	  {
		  cv::Scalar cl = m_colors[track.m_trackID % m_colors.size()];

		  for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
		  {
			  const TrajectoryPoint& pt1 = track.m_trace.at(j);
			  const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
#if (CV_VERSION_MAJOR >= 4)
			  cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, cv::LINE_AA);
#else
			  cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, CV_AA);
#endif
			  if (!pt2.m_hasRaw)
			  {
#if (CV_VERSION_MAJOR >= 4)
				  cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, cv::LINE_AA);
#else
				  cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, CV_AA);
#endif
			  }
		  }
	  }

	  if (m_useLocalTracking)
	  {
		  cv::Scalar cl = m_colors[track.m_trackID % m_colors.size()];

		  for (auto pt : track.m_lastRegion.m_points)
		  {
#if (CV_VERSION_MAJOR >= 4)
			  cv::circle(frame, cv::Point(cvRound(pt.x), cvRound(pt.y)), 1, cl, -1, cv::LINE_AA);
#else
			  cv::circle(frame, cv::Point(cvRound(pt.x), cvRound(pt.y)), 1, cl, -1, CV_AA);
#endif
		  }
	  }
  }

  ///
  /// \brief CarsCounting::InitTracker
  /// \param grayFrame
  ///
  bool CarsCounting::InitTracker(cv::UMat frame)
  {
	  m_minObjWidth = 5;//frame.cols / 50;

	  const int minStaticTime = 2;

	  config_t config;
#if 1
	  config["history"] = std::to_string(cvRound(10 * minStaticTime * m_fps));
	  config["varThreshold"] = "16";
	  config["detectShadows"] = "1";
	  m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_MOG2, config, m_useLocalTracking, frame));
#else
	  config["minPixelStability"] = "15";
	  config["maxPixelStability"] = "900";
	  config["useHistory"] = "1";
	  config["isParallel"] = "1";
	  m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Motion_CNT, config, m_useLocalTracking, frame));
#endif
	  m_detector->SetMinObjectSize(cv::Size(m_minObjWidth, m_minObjWidth));

	  TrackerSettings settings;
	  settings.m_useLocalTracking = m_useLocalTracking;
	  settings.m_distType = tracking::DistCenters;
	  settings.m_kalmanType = tracking::KalmanLinear;
	  settings.m_filterGoal = tracking::FilterRect;
	  settings.m_lostTrackType = tracking::TrackKCF;// original TrackerCSRT; // Use KCF tracker for collisions resolving
	  settings.m_matchType = tracking::MatchHungrian;
	  settings.m_dt = 0.5f;                             // Delta time for Kalman filter
	  settings.m_accelNoiseMag = 0.5f;                  // Accel noise magnitude for Kalman filter
	  settings.m_distThres = frame.rows / 15.f;         // Distance threshold between region and object on two frames

	  settings.m_useAbandonedDetection = true;		 // check the given region is static or not (abandoned)
	  if (settings.m_useAbandonedDetection)
	  {
		  settings.m_minStaticTime = minStaticTime;
		  settings.m_maxStaticTime = 6;
		  settings.m_maximumAllowedSkippedFrames = cvRound(settings.m_minStaticTime * m_fps); // Maximum allowed skipped frames
		  settings.m_maxTraceLength = 2 * settings.m_maximumAllowedSkippedFrames;        // Maximum trace length
	  }
	  else
	  {
		  settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
		  settings.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
	  }

	  m_tracker = std::make_unique<CTracker>(settings);

	  return true;
  }

  ///
  /// \brief CarsCounting::DrawData
  /// \param frame
  ///
  void CarsCounting::DrawData(cv::Mat frame, int framesCounter, int currTime)
  { // 여기서 각종 event/ property update를 하면 됨...
	  if (m_showLogs)
	  {
		  std::cout << "Frame " << framesCounter << ": tracks = " << m_tracker->tracks.size() << ", time = " << currTime << std::endl;
	  }

	  std::set<size_t> currIntersections;

	  for (const auto& track : m_tracker->tracks)
	  {
		  if (track->IsStatic())
		  {
			  DrawTrack(frame, 1, *track, true, true);
		  }
		  else
		  {
			  if (track->IsRobust(cvRound(m_fps / 4),          // Minimal trajectory size
				  0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				  cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
				  )
			  {
				  DrawTrack(frame, 1, *track, true);

				  CheckLinesIntersection(*track, static_cast<float>(frame.cols), static_cast<float>(frame.rows), currIntersections);
			  }
		  }
	  }

	  m_lastIntersections.clear();
	  m_lastIntersections = currIntersections;

	  //m_detector->CalcMotionMap(frame);

	  for (const auto& rl : m_lines)
	  {
		  rl.Draw(frame);
	  }
  }

  ///
  /// \brief CarsCounting::AddLine
  /// \param newLine
  ///
  void CarsCounting::AddLine(const RoadLine& newLine)
  {
	  m_lines.push_back(newLine);
  }

  ///
  /// \brief CarsCounting::GetLine
  /// \param lineUid
  /// \return
  ///
  bool CarsCounting::GetLine(unsigned int lineUid, RoadLine& line)
  {
	  for (const auto& rl : m_lines)
	  {
		  if (rl.m_uid == lineUid)
		  {
			  line = rl;
			  return true;
		  }
	  }
	  return false;
  }

  ///
  /// \brief CarsCounting::RemoveLine
  /// \param lineUid
  /// \return
  ///
  bool CarsCounting::RemoveLine(unsigned int lineUid)
  {
	  for (auto it = std::begin(m_lines); it != std::end(m_lines);)
	  {
		  if (it->m_uid == lineUid)
		  {
			  it = m_lines.erase(it);
		  }
		  else
		  {
			  ++it;
		  }
	  }
	  return false;
  }

  ///
  /// \brief CarsCounting::CheckLinesIntersection
  /// \param track
  ///
  void CarsCounting::CheckLinesIntersection(const CTrack& track, float xMax, float yMax, std::set<size_t>& currIntersections)
  {
	  auto Pti2f = [&](cv::Point pt) -> cv::Point2f
	  {
		  return cv::Point2f(pt.x / xMax, pt.y / yMax);
	  };

	  for (auto& rl : m_lines)
	  {
		  if (m_lastIntersections.find(track.m_trackID) == m_lastIntersections.end())
		  {
			  if (rl.IsIntersect(Pti2f(track.m_trace[track.m_trace.size() - 3]), Pti2f(track.m_trace[track.m_trace.size() - 1])))
			  {
				  currIntersections.insert(track.m_trackID);
			  }
		  }
	  }
  }  

} // itms namespace