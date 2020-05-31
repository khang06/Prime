using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Prime_MIDI
{
    class CudaFFT
    {
        [DllImport("cuda_fft")]
        public static extern IntPtr process_audio(IntPtr audio, ulong size, uint rate);
    }
}
