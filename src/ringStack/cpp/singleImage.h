//
// Created by tinku on 11/12/20.
//

#ifndef ULTIMATEGOALCV_SINGLEIMAGE_H
#define ULTIMATEGOALCV_SINGLEIMAGE_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include<tuple>

using namespace cv;
using namespace std;

tuple<Mat, Mat> preProcess(const std::string& filePath);

Mat getHSVImage(const Mat& croppedImageInput);

Mat getYCrCbImage(const Mat& croppedImageInput);

std::vector<vector<Point>> getLargestContour(std::vector<vector<Point>> vec);

int numberOfRings(int width, int height);

void debug(const Mat& image);

std::vector<vector<Point>> findContours(const Mat& input);

tuple<Mat, int, int> drawRectangles(std::vector<vector<Point>> contours, const Mat& input);

void postProcess(const Mat& maskedImageInput, const tuple<Mat, Mat>& images);

#endif //ULTIMATEGOALCV_SINGLEIMAGE_H
