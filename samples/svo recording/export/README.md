# Stereolabs ZED - SVO Recording utilities

This sample demonstrates how to read a SVO file and convert it into an AVI file (LEFT + RIGHT) or (LEFT + DEPTH_VIEW).

It can also convert a SVO in the following png image sequences: LEFT+RIGHT, LEFT+DEPTH_VIEW, and LEFT+DEPTH_16Bit.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 7 64bits or later, Ubuntu 16.04
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

## Build the program

#### Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the Visual Studio `Win64` solution
- Open the resulting solution and change configuration to `Release`
- Build solution

#### Build for Linux

Open a terminal in the sample directory and execute the following command:

    mkdir build
    cd build
    cmake ..
    make

## Run the program

Open a command terminal in build directory and follow the instructions bellow:

```
Usage:

ZED_SVO_Export A B C

Please use the following parameters from the command line:
 A - SVO file path (input) : "path/to/file.svo"
 B - AVI file path (output) or image sequence folder(output) : "path/to/output/file.avi" or "path/to/output/folder/"
 C - Export mode:  0=Export LEFT+RIGHT AVI.
				   1=Export LEFT+DEPTH_VIEW AVI.
				   2=Export LEFT+RIGHT image sequence.
				   3=Export LEFT+DEPTH_VIEW image sequence.
				   4=Export LEFT+DEPTH_16Bit image sequence.
 A and B need to end with '/' or '\'

Examples:
  (AVI LEFT+RIGHT)              ZED_SVO_Export "path/to/file.svo" "path/to/output/file.avi" 0
  (AVI LEFT+DEPTH)              ZED_SVO_Export "path/to/file.svo" "path/to/output/file.avi" 1
  (SEQUENCE LEFT+RIGHT)         ZED_SVO_Export "path/to/file.svo" "path/to/output/folder/" 2
  (SEQUENCE LEFT+DEPTH)         ZED_SVO_Export "path/to/file.svo" "path/to/output/folder/" 3
  (SEQUENCE LEFT+DEPTH_16Bit)   ZED_SVO_Export "path/to/file.svo" "path/to/output/folder/" 4
```

## Troubleshooting

If you want to tweak the video file option in the sample code (for example recording a mp4 file), you may have to recompile OpenCV with the FFmpeg option (WITH_FFMPEG).
