/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
// License Agreement
// For Open Source Computer Vision Library
// (3-clause BSD License)
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//M*/

/*
// Original file: https://github.com/Itseez/opencv_contrib/blob/292b8fa6aa403fb7ad6d2afadf4484e39d8ca2f1/modules/tracking/samples/tracker.cpp
// + Author: Klaus Haag
// * Refactor file: Move target selection to separate class/file
*/
/*
// modified by sangkny
// last updated on 2019. 01. 20
*/


#include "init_box_selector.hpp"

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
