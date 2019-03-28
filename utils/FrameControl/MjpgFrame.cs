using System;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Collections;

namespace Topes
{
    class MjpgFrame
    {
        private FileStream _fs;
        private MemoryStream _ms = new MemoryStream(100 * 1000); // 100k
        private BinaryReader _br;
        private int _length;        // count
        private int[] _idxs;        // idx1(0), idx2, ..., idxCount
        private int _framePosition; // [0,count-1]
        private string _filePath = "";
        private byte[] _header = new byte[256];

        private BinaryWriter _bw;


        // midx progress
        public delegate void MakeIndexPercentChangedHandler(int percent);
        public event MakeIndexPercentChangedHandler MakeIndexPercentChanged;

        public MjpgFrame()
        {
        }

        // mjpg frame file path
        public string FilePath
        {
            get { return _filePath; }
            set {
                if (value==null || value=="")
                {
                    _filePath = "";
                    return;
                }
                if (!Open(value))
                {
                    string info = String.Format("{0} 파일 열기에 실패했습니다.", value);
                    throw new Exception(info);
                }
            }
        }

        // frame count
        public int Length
        {
            get { return _length; }
        }

        // [0,Length-1]
        public int Position
        {
            get { return _framePosition; }
            set { SetPosition(value); }
        }

        // frame header
        public byte[] Header
        {
            get { return _header; }
            set { _header = value; }
        }

        // jpg data stream
        public Stream DataStream
        {
            get { return _ms; }
        }

        // make .midx file
        private bool MakeIndexFile(string mjpgFile, string idxFile)
        {
            // midx progress
            int progress = 0;
            if (MakeIndexPercentChanged != null) 
            {
                MakeIndexPercentChanged(progress);
            }

            ArrayList idxs = new ArrayList(216000);

            while (_br.PeekChar() != -1)
            {
                idxs.Add((int)_br.BaseStream.Position);
                _br.BaseStream.Position += 256;
                int len = _br.ReadInt32();
                if (len<1)
                {
                    idxs.RemoveAt(idxs.Count - 1);
                    break;
                }
                _br.BaseStream.Position += len;

                // midx progress
                progress = (int)(90 * _br.BaseStream.Position / _br.BaseStream.Length);
                if (MakeIndexPercentChanged != null)
                {
                    MakeIndexPercentChanged(progress);
                }
            }
            _idxs = new int[idxs.Count];
            idxs.CopyTo(_idxs);
            _length = idxs.Count;

            FileStream ifs = new FileStream(idxFile, FileMode.CreateNew);
            BinaryWriter ibw = new BinaryWriter(ifs);
            ibw.Write(_length);
            for (int i = 0; i < _idxs.Length; i++)
            {
                ibw.Write(_idxs[i]);

                // midx progress
                progress = 90 + (int)(10 * i / _idxs.Length);
                if (MakeIndexPercentChanged != null)
                {
                    MakeIndexPercentChanged(progress);
                }
            }
            ibw.Close();
            ifs.Close();

            // midx progress
            progress = 100;
            if (MakeIndexPercentChanged != null)
            {
                MakeIndexPercentChanged(progress);
            }

            return true;
        }

        public bool Open(string filePath)
        {
            if (filePath==null || !File.Exists(filePath))
            {
                return false;
            }

            if (_fs != null)
            {
                _fs.Close();
            }
            if (_br != null)
            {
                _br.Close();
            }
            _fs = new FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite);
            _br = new BinaryReader(_fs);
            _bw = new BinaryWriter(_fs);

            FileInfo fi = new FileInfo(filePath);
            string ext = fi.Extension;
            string idxFile = filePath.Replace(ext, ".midx");

            if (!File.Exists(idxFile))
            {
                MakeIndexFile(filePath, idxFile);
            }
            else
            {
                FileStream ifs = new FileStream(idxFile, FileMode.Open, FileAccess.Read, FileShare.Read);
                BinaryReader ibr = new BinaryReader(ifs);
                _length = ibr.ReadInt32();
                Debug.Assert(_length == ifs.Length / 4 - 1);
                _idxs = new int[_length];
                for (int i = 0; i < _length; i++)
                {
                    _idxs[i] = ibr.ReadInt32();
                }
                ibr.Close();
                ifs.Close();
            }

            _filePath = filePath;
            SetPosition(0);
            return true;
        }

        public bool SetPosition(int pos)
        {
            if (_filePath.Length < 1 || pos >= _idxs.Length)
            {
                return false;
            }
            _framePosition = pos;
            _br.BaseStream.Position = _idxs[_framePosition];
            _header = _br.ReadBytes(256); //_br.BaseStream.Seek(256, SeekOrigin.Current);
            int dataStreamLength = _br.ReadInt32();
            _ms.SetLength(dataStreamLength);
            _br.Read(_ms.GetBuffer(), 0, (int)dataStreamLength);
            return true;
        }

        public bool SetSpeed()
        {

            if (_filePath.Length < 1)// || pos >= _idxs.Length)
            {
                return false;
            }


            if (_fs != null)
            {
                _bw.BaseStream.Position = _idxs[_framePosition];

                _bw.Write(_header);

                //bw.Close();
            }

            return true;
        }
    }
}
