//#include <opencv2/opencv.hpp>
//#include <tuple>
//#include "main.h"
//
//using namespace cv;
//using namespace std;
//
//int main() {
//    tuple<Mat, Mat> images = preProcess((String) "/home/tinku/Desktop/UltimateGoalCV/src/assets/straightZeroStack.jpeg");
//    Mat imageHSVMasked = getHSVImage(get<0>(images));
//    Mat imageYCrCbMasked = getYCrCbImage(imageHSVMasked);
//    postProcess(imageYCrCbMasked, images);
//    return 0;
//}
//
//tuple<Mat, Mat> preProcess(const std::string& filePath) {
//    Mat image = imread(filePath);
//    int width = image.cols;
//    int height = image.rows;
//    Rect roi(lround(0.35* width), lround(0.35* height), lround(0.7* width - 0.35* width), lround(0.7* height - (0.35 *height)));
//    Mat croppedImage = image(roi);
//    return make_tuple(croppedImage, image);
//}
//
//Mat getHSVImage(const Mat& croppedImageInput){
//    Mat croppedImage = croppedImageInput.clone();
//    Mat croppedImageHSVMasked;
//    Mat mask;
//    Scalar lowerBoundHSV = Scalar(8, 92, 77);
//    Scalar upperBoundHSV = Scalar(90, 255, 255);
//    cv::cvtColor(croppedImage, croppedImage, COLOR_BGR2HSV);
//    cv::inRange(croppedImage, lowerBoundHSV, upperBoundHSV, mask);
//    cv::bitwise_and(croppedImage, croppedImage, croppedImageHSVMasked, mask);
//    cv::cvtColor(croppedImageHSVMasked, croppedImageHSVMasked, COLOR_HSV2BGR);
//    return croppedImageHSVMasked;
//}
//
//Mat getYCrCbImage(const Mat& croppedImageInput){
//    Mat croppedImage = croppedImageInput.clone();
//    Mat croppedImageYCrCbMasked;
//    Mat mask;
//    Scalar lowerBoundYCrCb = Scalar(0, 152, 64);
//    Scalar upperBoundYCrCb = Scalar(255, 255, 113);
//    cv::cvtColor(croppedImage, croppedImage, COLOR_BGR2YCrCb);
//    cv::inRange(croppedImage, lowerBoundYCrCb, upperBoundYCrCb, mask);
//    cv::bitwise_and(croppedImage, croppedImage, croppedImageYCrCbMasked, mask);
//    return croppedImageYCrCbMasked;
//}
//
//std::vector<vector<Point>> getLargestContour(std::vector<vector<Point>> vec){
//    double max = 0;
//    int index = 0;
//    for (auto & i : vec){
//        if (contourArea(i) > max){
//            max = contourArea(i);
//            index++;
//        }
//    }
//    vec.erase(vec.begin(), vec.begin() + index - 1);
//    vec.erase(vec.begin() + 1, vec.end());
//    return vec;
//}
//
//int numberOfRings(int width, int height){
//    double aspectRatio = (double) width/height;
//    if ((aspectRatio > 1.3) && (aspectRatio < 1.7)){
//        return 4;
//    }
//    else if (aspectRatio > 1.7){
//        return 1;
//    }
//    else{
//        return 0;
//    }
//}
//
//std::vector<vector<Point>> findContours(const Mat& input){
//    Mat maskedImage = input.clone();
//    Mat bilateralFiltered;
//    Mat thresHolded;
//    std::vector<vector<Point>> contours;
//    cv::GaussianBlur(maskedImage, maskedImage, Size(5, 5), 0);
//    cv::bilateralFilter(maskedImage, bilateralFiltered, 10, 150, 150);
//    cv::cvtColor(bilateralFiltered, bilateralFiltered, COLOR_BGR2GRAY);
//    cv::threshold(bilateralFiltered, thresHolded, 128, 255, THRESH_BINARY | THRESH_OTSU);
//    cv::findContours(thresHolded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//    contours = getLargestContour(contours);
//    return contours;
//}
//
//tuple<Mat, int, int> drawRectangles(std::vector<vector<Point>> contours, const Mat& input){
//    Mat originalImageCropped = input.clone();
//    Scalar color = Scalar(0, 0, 255);
//    std::vector<Rect> boundRect( contours.size() );
//    std::vector<vector<Point> > contours_poly( contours.size());
//    approxPolyDP( contours[0], contours_poly[0], 3, true );
//    boundRect[0] = boundingRect( contours_poly[0] );
//    rectangle(originalImageCropped, boundRect[0].tl(), boundRect[0].br(), color, 4);
//    return make_tuple(originalImageCropped, boundRect[0].width, boundRect[0].height);
//}
//
//void debug(const Mat& image){
//    String windowName = "Rings";
//    imshow(windowName, image);
//    waitKey(0);
//    destroyWindow(windowName);
//}
//
//void postProcess(const Mat& maskedImageInput, const tuple<Mat, Mat>& images){
//    Mat maskedImage = maskedImageInput.clone();
//    Mat originalImageCropped = get<0>(images).clone();
//    Mat fullImage = get<1>(images).clone();
//    std::vector<vector<Point>> contours = findContours(maskedImage);
//    tuple<Mat, int, int> rectsDrawn = drawRectangles(contours, originalImageCropped);
//    cout << numberOfRings(get<1>(rectsDrawn), get<2>(rectsDrawn));
//    Mat insetImage(fullImage, Rect((lround(0.35*fullImage.cols)), (lround(0.35*fullImage.rows)), get<0>(rectsDrawn).cols, get<0>(rectsDrawn).rows));
//    get<0>(rectsDrawn).copyTo(insetImage);
//    Mat fullResized;
//    cv::resize(fullImage, fullResized, Size(lround(0.3*fullImage.cols), lround(0.3*fullImage.rows)));
//    debug(fullResized);
//}
//
//
//
