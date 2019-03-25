using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using ITMSAPI_CLIWrap;

namespace ITMSAPICLI_CS_Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            ITMSAPI_CLIWrap.ITMS_CLIWrap obj = new ITMSAPI_CLIWrap.ITMS_CLIWrap();

            Console.WriteLine(obj.addValue(1, 2));
            Console.Read();            
        }
    }
}
