//
// Created by tinku on 11/12/20.
//

#ifndef ULTIMATEGOALCV_PHONECAM_H
#define ULTIMATEGOALCV_PHONECAM_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include<tuple>

using namespace cv;
using namespace std;

Mat drawRectanglesImg(std::vector<vector<Point>> contours, const Mat& input);

tuple<Mat, Mat> preProcess(const Mat& input);

Mat postProcessImg(const Mat& maskedImageInput, const tuple<Mat, Mat>& images);
#endif //ULTIMATEGOALCV_PHONECAM_H
