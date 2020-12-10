# UltimateGoalCV
Vision pipeline for Ultimate Goal

Currently depends on C++20 and Python 3.6
Needs natively built OpenCV with Python 3.6 and C++ packages along with Eigen for some operations.

## Usage
1) Open up the project in CLion and set the respective CMake paths for proper compilation of the library
2) Start with the `ColorTuningHSV.py` and replace the path with the absolute path of one of the images from the `assets` folder
3) Slide the values till the rings are left
4) Store these values and redo with `ColorTuningYCrCb.py` and store these values as well.
5) Finally, replace the values in either `webcam.cpp` or `main.cpp` and it should work.
