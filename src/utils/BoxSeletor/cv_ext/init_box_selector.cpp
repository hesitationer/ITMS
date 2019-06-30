

#include "utils/BoxSelector/cv_ext/init_box_selector.hpp"

void InitBoxSelector::onMouse(int event, int x, int y, int, void*)
{
    if (!selectObject)
    {
        switch (event)
        {
        case cv::EVENT_LBUTTONDOWN:
            //set origin of the bounding box
            startSelection = true;
            initBox.x = x;
            initBox.y = y;
            break;
        case cv::EVENT_LBUTTONUP:
            //set width and height of the bounding box
            initBox.width = std::abs(x - initBox.x);
            initBox.height = std::abs(y - initBox.y);
            startSelection = false;
            selectObject = true;
            break;
        case cv::EVENT_MOUSEMOVE:
            if (startSelection && !selectObject)
            {
                //draw the bounding box
                cv::Mat currentFrame;
                image.copyTo(currentFrame);
                cv::rectangle(currentFrame, cv::Point((int)initBox.x, (int)initBox.y), cv::Point(x, y), cv::Scalar(255, 0, 0), 2, 1);				
                cv::imshow(windowTitle.c_str(), currentFrame);
            }
            break;
        }
    }
}

bool InitBoxSelector::selectBox(cv::Mat& frame, cv::Rect& initBox)
{
    frame.copyTo(image);
    startSelection = false;
    selectObject = false;
	cv::namedWindow(windowTitle.c_str(), cv::WINDOW_NORMAL);
    cv::imshow(windowTitle.c_str(), image);
    cv::setMouseCallback(windowTitle.c_str(), onMouse, 0);

    while (selectObject == false)
    {
        char c = (char)cv::waitKey(10);

        if (c == 27)
            return false;
    }

    initBox = InitBoxSelector::initBox;
    cv::setMouseCallback(windowTitle.c_str(), 0, 0);
    cv::destroyWindow(windowTitle.c_str());
    return true;
}

const std::string InitBoxSelector::windowTitle = "Draw Bounding Box";
bool InitBoxSelector::startSelection = false;
bool InitBoxSelector::selectObject = false;
cv::Mat InitBoxSelector::image;
cv::Rect InitBoxSelector::initBox;


// InitPointSelector Functions
void InitPointSelector::onMouse(int event, int x, int y, int, void*)
{
	if (!selectObject)
	{
		cv::Point curPt;
		switch (event)
		{
		case cv::EVENT_LBUTTONDOWN:
			//set origin of the bounding box
			startSelection = true;
			initPt = cv::Point(x,y);
			break;
		case cv::EVENT_LBUTTONUP:
			// check the location and put the point untile the max points
			curPt=cv::Point(x, y);
			if(curPt==initPt){
				initBoxPts.push_back(curPt);
				numPoints = initBoxPts.size();
			}
			if(initBoxPts.size()==max_numPoints){
				startSelection = false;
				selectObject = true;
			}
			break;
		case cv::EVENT_MOUSEMOVE:
			if (startSelection && !selectObject)
			{
				//draw the bounding box
				cv::Mat currentFrame;
				image.copyTo(currentFrame);
				if (initBoxPts.size() ){ // more than 1 point in the vector
					for (int i = 0; i < initBoxPts.size() - 1; i++) {
						line(currentFrame, initBoxPts.at(i), initBoxPts.at((i + 1)), cv::Scalar(255, 0, 0), 2, 1);
					}
					cv::line(currentFrame, initBoxPts.back(), cv::Point(x, y), cv::Scalar(255, 0, 0), 2, 1); // draw from the last point
				}								
				cv::imshow(windowTitle.c_str(), currentFrame);
			}
			break;
		}
	}
}

bool InitPointSelector::selectPoints(cv::Mat& frame, std::vector<cv::Point>& initPoints)
{
	frame.copyTo(image);
	numPoints = 0;
	max_numPoints = 4;
	if(initBoxPts.size())
		initBoxPts.clear();

	startSelection = false;
	selectObject = false;
	cv::namedWindow(windowTitle.c_str(), cv::WINDOW_NORMAL);
	cv::imshow(windowTitle.c_str(), image);
	cv::setMouseCallback(windowTitle.c_str(), onMouse, 0);

	while (selectObject == false)
	{
		char c = (char)cv::waitKey(10);

		if (c == 27)
			return false;
	}

	initPoints = InitPointSelector::initBoxPts;
	cv::setMouseCallback(windowTitle.c_str(), 0, 0);
	cv::destroyWindow(windowTitle.c_str());
	return true;
}

const std::string InitPointSelector::windowTitle = "Draw Point Lines";
bool InitPointSelector::startSelection = false;
bool InitPointSelector::selectObject = false;
cv::Mat InitPointSelector::image;
std::vector<cv::Point> InitPointSelector::initBoxPts;
cv::Point InitPointSelector::initPt;
int InitPointSelector::numPoints;
int InitPointSelector::max_numPoints;
