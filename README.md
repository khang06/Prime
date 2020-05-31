# Prime-CUDA
Prime-MIDI, but 40 or so times faster with GPU acceleration!

## Building
Prerequisites: Visual Studio 2019, CUDA Toolkit 10.2 (other versions might work too)
1. Build the solution as x64 Release **(it will not work correctly yet, trying to use it now will crash)**
2. Open up a VS 2019 Developer Command Prompt
3. Compile the DLL with `nvcc -O3 -Xptxas -O3,-v --shared cuda_fft.cu -o cuda_fft.dll`
4. Copy it into `Prime-MIDI\bin\x64\Release`
5. Run and enjoy!
