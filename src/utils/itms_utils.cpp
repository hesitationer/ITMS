#include "itms_utils.h"

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
} // itms namespace