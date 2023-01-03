#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "libai_core.hpp"


using namespace std;
using namespace ucloud;
using namespace cv;

#ifndef VAR
#define VAR private
#endif

#include <math.h>
#include <algorithm>
#include <string.h>

namespace unitest{

};

int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL)); 
    int W = 640, H = 480;
    int _W = W/10, _H = H/10;
    int N = 5; int B = 5;
    //generate random rects
    Mat cv_mask = Mat::zeros(Size(W,H), CV_8UC1);
    VecObjBBox bboxes;
    for(int n = 0; n < N; n++ ){
        int x = rand()%W;
        int y = rand()%H;
        int w = rand()%_W;
        int h = rand()%_H;
        w = (x+w >= W ) ? W-x:w;
        h = (y+h >= H ) ? H-y:h;
        TvaiRect t = TvaiRect{x,y,w,h};
        BBox box;
        box.rect = t;
        bboxes.push_back(box);
        cv_mask(Range(y,y+h), Range(x,x+w)) = 255;
        // rectangle(cv_mask, Rect(x,y,w,h), Scalar(255), 1);
    }
    // imwrite("result/tmp.jpg", cv_mask);
    Mat cv_kernel = Mat::ones(Size(3,3), CV_8UC1);
    Mat cv_dilate_mask, cv_t_mask;
    vector<vector<Point>> vec_cv_contours;
    dilate(cv_mask, cv_dilate_mask , cv_kernel, Point(1,1), 5);
    // imwrite("result/tmp.jpg", cv_dilate_mask);
    
    float auto_thresh = cv::threshold(cv_dilate_mask, cv_t_mask, 0, 255, THRESH_BINARY+THRESH_OTSU);
    cout << "auto threshold = " << auto_thresh << endl;
    imwrite("result/tmp.jpg", cv_t_mask);

    findContours(cv_t_mask,vec_cv_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    cout << "contours: #" << vec_cv_contours.size() << endl;
    for(auto iter=vec_cv_contours.begin(); iter!=vec_cv_contours.end(); iter++){
        Rect rect = boundingRect(*iter);
        BBox bbox;
        bbox.confidence = 1.0;
        bbox.rect.x = rect.x; bbox.rect.width = rect.width;
        bbox.rect.y = rect.y; bbox.rect.height = rect.height;
        cout << "contours:" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << endl;
        bboxes.push_back(bbox);
    }


}