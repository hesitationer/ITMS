using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using System.Runtime.InteropServices;
using System.Drawing.Imaging;


using System.Diagnostics;

using System.IO;

//using TestManageCalProc;
// ------------ how to use API -- 1 ---------
using OpenCvSharp;
using ITMSAPI_CLIWrap;

namespace ITMS_Demo
{   
    public partial class Main : Form
    {

        //[DllImport(@"ITMS_APICLI.dll", CallingConvention = CallingConvention.Cdecl)]
        //extern public static void TestCall(int i);


        //[DllImport(@"ITMS_APICLI.dll", CallingConvention = CallingConvention.Cdecl)]
        //extern public static void SetFrame(int Ch, int width, int height, byte[] pFrame, int frameType);


        //public delegate void ResultCallback(ref DefineStruct.ResultInfo data);
        //[DllImport(@"ITMS_APICLI.dll", CallingConvention = CallingConvention.Cdecl)]
        //public static extern int DoWorkResultData(ResultCallback callbackPointer);
        //private ResultCallback resultDataCallBack;
        

        #region Var

        private string[] MjpgFileNames = null;
        private int FileIndex = 0;

        byte[] buffer = new byte[1920*1080*3];

        //private string debugFolderPath = @"Q:\Camera1\20180911\주간\주간 - 일반 - 복합 - 정지(1) 보행(1) 정지(1)  --  1415-1510\20180911";
        private string debugFolderPath = @"";

		// ------------ how to use API -- 2 ---------
		private ITMSAPI_CLIWrap.ITMS_CLIWrap obj = new ITMSAPI_CLIWrap.ITMS_CLIWrap();
		private Mat image = new Mat();

        #endregion
  

        public Main()
        {
            InitializeComponent();
			// ------------ how to use API -- 3 ---------
			obj.Init();

		}
        

        private void Main_Load(object sender, EventArgs e)
        {

        }


        // 이미지 전송 타이머
        private void timerPlay_Tick(object sender, EventArgs e)
        {
            try
            {          
                this.Invoke((MethodInvoker)delegate ()
                {
                    if (mjpgControl1.Position + 1 >= mjpgControl1.Length)
                    {
                        //mjpgControl1.Position = 0;

                        if(MjpgFileNames.Length -1 <= FileIndex)
                        {
                            FileIndex = 0;
                        }

                        mjpgControl1.FilePath = MjpgFileNames[++FileIndex];
                        Console.WriteLine(" File path: {0} \n", mjpgControl1.FilePath);
                    }                    
                    mjpgControl1.Position = mjpgControl1.Position + 1;

                    unsafe
                    {

                        // 시뮬레이션 때문에 사용      --- jpg 이미지에서 raw 데이터로 변환시 시간 소요
                        Bitmap bt = (Bitmap)Image.FromStream(mjpgControl1.DataStream);
						image = OpenCvSharp.Extensions.BitmapConverter.ToMat(bt);						      
						// ------------ how to use API -- 4 ---------     
						if (!image.Empty())
						{
							obj.ResetAndProcessFrame(0, image.DataPointer, bt.Width * bt.Height * 3);
                            if (obj.GetObjectNumber() > 0)
                            {
                                Console.WriteLine(" Event happened!!\n ");
                                for (int i = 0; i < obj.GetObjectNumber(); i++) {                                    
                                    Console.WriteLine(" Object ID:{0}, Class:{1}, Status:{2}, Speed:{3} \r \n ", obj.GetObjectIDAt(i),
                                        obj.GetObjectClassAt(i), obj.GetObjectStatusAt(i), obj.GetObjectSpeedAt(i));
                                }
                            }
                        }
					}
				});

       
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

        }

        public string[] FolderOpen()
        {
            try
            {
                if (debugFolderPath != "")
                {
                    string searchPatterns = "*.mjpg";
                    string[] files = Directory.GetFiles(debugFolderPath, searchPatterns);
                    return files;
                }
                else
                { 
                    FolderBrowserDialog OpenFolder = new FolderBrowserDialog();

                    if (OpenFolder.ShowDialog() == DialogResult.OK)
                    {

                        string path = OpenFolder.SelectedPath + @"\";
                        string searchPatterns = "*.mjpg";
                        string[] files = Directory.GetFiles(path, searchPatterns);
                        //string[] files = Directory.GetFiles(@"C:\", "*.*");

                        return files;
                    }
                }

            }
            catch (Exception ex)
            {

                throw;
            }

            return null;
        }
        

        private void OnResultData(ref DefineStruct.ResultInfo data)
        {
            string s = Convert.ToString(data.objClassId);
        }


        // 영상 전송 시작
        private void button1_Click(object sender, EventArgs e)
        {
            try
            {           
                MjpgFileNames = FolderOpen();


                if (MjpgFileNames == null)
                    Application.Exit();
                

                mjpgControl1.FilePath = MjpgFileNames[FileIndex++];

                timerPlay.Interval = 10;
                timerPlay.Start();
            }
            catch (Exception ex)
            {
        
            }        
        }

        //
        private void button2_Click(object sender, EventArgs e)
        {

        }

        // 이전 파일 재생
        private void button3_Click(object sender, EventArgs e)
        {
            if (FileIndex > 1)
            {
                mjpgControl1.FilePath = MjpgFileNames[--FileIndex];                
                Console.WriteLine(" File path: {0} \n", mjpgControl1.FilePath);

            }

        }

        // 다음 파일 재생
        private void button4_Click(object sender, EventArgs e)
        {
            if (FileIndex > 0 && (MjpgFileNames.Length - 1) > FileIndex)
            {
                mjpgControl1.FilePath = MjpgFileNames[++FileIndex];
                Console.WriteLine(" File path: {0} \n", mjpgControl1.FilePath);
            }
        }

        // 시뮬레이션 영상 표출 안함.
        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            mjpgControl1.Display = (bool)checkBox1.Checked;
        }


        // 콜백 포인터가 등록되었는지 테스트
        int idex = 0;
        private void button5_Click(object sender, EventArgs e)
        {
            //TestCall(idex++);
        }


        // 콜백 등록
        private void button6_Click(object sender, EventArgs e)
        {
            //resultDataCallBack = new ResultCallback(OnResultData);
            //DoWorkResultData(resultDataCallBack);
        }
    }

    // 검지 결과값 구조체
    public class DefineStruct
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct RECT
        {
            public Int32 Left, Top, Right, Bottom;

            public RECT(int left, int top, int right, int bottom)
            {
                Left = left;
                Top = top;
                Right = right;
                Bottom = bottom;
            }

            public RECT(System.Drawing.Rectangle r) : this(r.Left, r.Top, r.Right, r.Bottom) { }

            public int X
            {
                get { return Left; }
                set { Right -= (Left - value); Left = value; }
            }

            public int Y
            {
                get { return Top; }
                set { Bottom -= (Top - value); Top = value; }
            }

            public int Height
            {
                get { return Bottom - Top; }
                set { Bottom = value + Top; }
            }

            public int Width
            {
                get { return Right - Left; }
                set { Right = value + Left; }
            }

            public System.Drawing.Point Location
            {
                get { return new System.Drawing.Point(Left, Top); }
                set { X = value.X; Y = value.Y; }
            }

            public System.Drawing.Size Size
            {
                get { return new System.Drawing.Size(Width, Height); }
                set { Width = value.Width; Height = value.Height; }
            }

            public static implicit operator System.Drawing.Rectangle(RECT r)
            {
                return new System.Drawing.Rectangle(r.Left, r.Top, r.Width, r.Height);
            }

            public static implicit operator RECT(System.Drawing.Rectangle r)
            {
                return new RECT(r);
            }

            public static bool operator ==(RECT r1, RECT r2)
            {
                return r1.Equals(r2);
            }

            public static bool operator !=(RECT r1, RECT r2)
            {
                return !r1.Equals(r2);
            }

            public bool Equals(RECT r)
            {
                return r.Left == Left && r.Top == Top && r.Right == Right && r.Bottom == Bottom;
            }

            public override bool Equals(object obj)
            {
                if (obj is RECT)
                    return Equals((RECT)obj);
                else if (obj is System.Drawing.Rectangle)
                    return Equals(new RECT((System.Drawing.Rectangle)obj));
                return false;
            }

            public override int GetHashCode()
            {
                return ((System.Drawing.Rectangle)this).GetHashCode();
            }

            public override string ToString()
            {
                return string.Format(System.Globalization.CultureInfo.CurrentCulture, "{{Left={0},Top={1},Right={2},Bottom={3}}}", Left, Top, Right, Bottom);
            }
        }

        public struct ResultInfo
        {
            public UInt32 objStatusId;            
            public UInt32 objStatus;            
            public UInt32 objClassId;           
            public UInt32 objClass;           
            public RECT objRect;            
            public Single objSpeed;          
        }
    }
}

