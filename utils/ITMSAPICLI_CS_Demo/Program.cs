using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.IO; // File Existence Check !!

// ----------------------------- How to Use API -- 1 -------------------------------------
using OpenCvSharp; 
using ITMSAPI_CLIWrap;
// ---------------------------------------------------------------------------------------

namespace ITMSAPICLI_CS_Demo
{
    class Program
    {
        static void Main(string[] args)
        {

            string filename = "..\\TrafficVideo\\20180911_113611_cam_0.avi"; //20180912_112338_cam_0
                        
            bool isFileExist = File.Exists(filename);
            Console.WriteLine( isFileExist ? " File exists." : "File does not exist.");
            if (!isFileExist)
            {
                Console.WriteLine("File does not exist.Please double check the file you choose !! \n");
                Console.Read();
                return;
            }

            using (VideoCapture capture = new VideoCapture(filename))
            {
                // ------------------------------------- How to use API -- 2 ------------------------------------
                ITMSAPI_CLIWrap.ITMS_CLIWrap obj = new ITMSAPI_CLIWrap.ITMS_CLIWrap();
                obj.Init();
                // ---------------------------------------------------------------------------------------------- 

                int sleepTime = (int)Math.Round(1000 / capture.Fps);
                bool debugShowImage = true;
                int ESCKEY = 0;

                using (Window window = new Window("Video Playing", WindowMode.AutoSize) )
                {
                    using (Mat image = new Mat()) // Frame image buffer
                    {
                        // When the movie playback reaches end, Mat.data becomes NULL.
                        while (true && ((char)ESCKEY != 27) )
                        {
                            capture.Read(image); // read data
                            if (image.Empty())
                                break;
                            unsafe
                            {
                                // ------------------------------- How to use API -- 3 ----------------------
                                obj.ResetAndProcessFrame(0, image.DataPointer, image.Cols * image.Rows * image.Channels());
                                if (obj.GetObjectNumber() > 0)
                                    Console.WriteLine(" Event happened!!\n");
                                // --------------------------------------------------------------------------
                            }
                            if (debugShowImage)
                                window.ShowImage(image);
                            ESCKEY = Window.WaitKey(1);
                        }
                    }
                }
                // ---------------------------------- How to USE API -- 4 -----------------------------------
                if (obj != null)
                    ((IDisposable)obj).Dispose();
                // ------------------------------------------------------------------------------------------
            }
        }        
    }
}
