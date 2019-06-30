using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using Topes;

using System.Drawing.Drawing2D;

namespace FrameControl
{
    public partial class MjpgControl : UserControl
    {
        MjpgFrame mjpg = new MjpgFrame();
        MidxProgress midxProgress = new MidxProgress();

        private bool display = true;
        
        public MjpgControl()
        {
            InitializeComponent();
        }

        // mjpg frame file path
        public string FilePath
        {
            get { return mjpg.FilePath; }
            set
            {
                if (value == null || value == "")
                {
                    mjpg.FilePath = "";
                    return;
                }
                try
                {
                    mjpg.MakeIndexPercentChanged += new MjpgFrame.MakeIndexPercentChangedHandler(MakeIndexPercentChangedHandler);
                    mjpg.FilePath = value;         

                    if(Display)
                        pictureBox1.Image = (Bitmap)Image.FromStream(mjpg.DataStream); 
                }
                catch (System.Exception e)
                {
                    MessageBox.Show(e.ToString());
                }
            }
        }

        // frame count
        public int Length
        {
            get { return mjpg.Length; }
        }

        // [0,Length-1]
        public int Position
        {
            get { return mjpg.Position; }
            set
            {
                if (mjpg.FilePath == null || mjpg.FilePath.Length < 1)
                {
                    return;
                }

                try
                {
                    mjpg.Position = value;   
                    if(Display)                
                         pictureBox1.Image = (Bitmap)Image.FromStream(mjpg.DataStream);

                }
                catch (System.Exception ex)
                {

                }
            }
        }
        
  
        public bool Display
        {
            get { return display; }
            set { display = value; }
        }



        // frame header
        public byte[] Header
        {
            get { return mjpg.Header; }
            set { mjpg.Header = value; }
        }

        // jpg data stream
        public Stream DataStream
        {
            get { return mjpg.DataStream; }
        }

        public int FrameWidth
        {
            get { return pictureBox1.Image.Width; }
        }

        public int FrameHeight
        {
            get { return pictureBox1.Image.Height; }
        }

        public string FrameBPP
        {
            get { return pictureBox1.Image.PixelFormat.ToString(); }
        }

        public void MakeIndexPercentChangedHandler(int percent)
        {
            //this.Text = percent.ToString();
            midxProgress.progressBar1.Value = percent;
            if (percent == 0)
            {
                midxProgress.Show();
            }
            else if (percent == 100)
            {
                midxProgress.Hide();
            }
        }

        public void UpdateHeader(int iSpd)
        {      
            mjpg.SetSpeed();
        }

    }
}
