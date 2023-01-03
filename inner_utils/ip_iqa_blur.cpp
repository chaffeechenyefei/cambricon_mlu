#include "ip_iqa_blur.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

Mat shiftDFT(Mat &cv_img){
    Mat tmp, q0, q1, q2, q3, dst;
    // first crop the image, if it has an odd number of rows or columns

    dst = cv_img(Rect(0, 0, cv_img.cols & -2, cv_img.rows & -2));

    int cx = dst.cols / 2;
    int cy = dst.rows / 2;

    // rearrange the quadrants of Fourier image
    // so that the origin is at the image center

    q0 = dst(Rect(0, 0, cx, cy));
    q1 = dst(Rect(cx, 0, cx, cy));
    q2 = dst(Rect(0, cy, cx, cy));
    q3 = dst(Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    return dst;
}

Mat create_butterworth_highpass_filter(int rows, int cols, int D, int n)
{
    Mat dft_Filter = Mat::zeros(rows, cols, CV_32F);

    Point center = Point(rows / 2, cols / 2);
    double radius;
    // based on the forumla in the IP notes (p. 130 of 2009/10 version)
    // see also HIPR2 on-line

    for (int i = 0; i < rows; i++)
    {   
        float *ptrR = dft_Filter.ptr<float>(i);
        for (int j = 0; j < cols; j++)
        {
            if(i==center.x&&j==center.y){
                radius = 1e-4;
            }else
                radius = (double) sqrt(pow((i - center.x), 2.0) + pow((double) (j - center.y), 2.0));
            ptrR[j] = (float)(1/ (1+ pow( (double)(D/radius) , double(2*n))) );
        }
    }

    Mat toMerge[] = {dft_Filter, dft_Filter};
    merge(toMerge, 2, dft_Filter);
    return dft_Filter;
}

/**
 * return  dft_Filter complex mat with (rows, cols)
 */
Mat create_butterworth_lowpass_filter(int rows, int cols, int D, int n)
{
    Mat dft_Filter = Mat::zeros(rows, cols, CV_32F);

    Point center = Point(rows / 2, cols / 2);
    double radius;
    // based on the forumla in the IP notes (p. 130 of 2009/10 version)
    // see also HIPR2 on-line

    for (int i = 0; i < rows; i++)
    {   
        float *ptrR = dft_Filter.ptr<float>(i);
        for (int j = 0; j < cols; j++)
        {
            radius = (double) sqrt(pow((i - center.x), 2.0) + pow((double) (j - center.y), 2.0));
            ptrR[j] = (float)(1/ (1+ pow( (double)(radius/D) , double(2*n))) );
        }
    }

    Mat toMerge[] = {dft_Filter, dft_Filter};
    merge(toMerge, 2, dft_Filter);
    return dft_Filter;
}

/**
 * D = radius
 * n = order
 * W = width
 * return complex images
 */
cv::Mat exec_butterworth_lowpass_filter(cv::Mat &cv_img, int D, int n){
    cv::Mat cv_gray, cv_padded, cv_complex, cv_filter;
    Mat planes[2];
    if(cv_img.channels()==3){
        cvtColor(cv_img, cv_gray, cv::COLOR_BGR2GRAY);
    } else {
        cv_img.copyTo(cv_gray);
    }

    int M = getOptimalDFTSize( cv_gray.rows );
    int N = getOptimalDFTSize( cv_gray.cols );

    copyMakeBorder(cv_gray, cv_padded, 0, M - cv_gray.rows, 0, N - cv_gray.cols, cv::BORDER_CONSTANT, Scalar::all(0));
    planes[0] = Mat_<float>(cv_padded);
    planes[1] = Mat::zeros(cv_padded.size(), CV_32F);
    merge(planes, 2, cv_complex);

    dft(cv_complex, cv_complex);

    cv_filter = create_butterworth_lowpass_filter(M, N, D, n);

    shiftDFT(cv_complex);
    mulSpectrums(cv_complex, cv_filter, cv_complex, 0);
    shiftDFT(cv_complex);

    return cv_complex;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// IQA_BLUR
////////////////////////////////////////////////////////////////////////////////////////////////
void IQA_BLUR::init(int rows, int cols, int D, int n){
    m_rows = getOptimalDFTSize(rows);
    m_cols = getOptimalDFTSize(cols);
    m_D = D;
    m_n = n;
    m_filter_mat = create_butterworth_highpass_filter(m_rows, m_cols, D, n);
}

/**
 * final: 0-1 low-high
 * given (2.x,5.x) -> (0,1)
 * when x = 0.211824, given coef=4, return 0.7
 */
float IQA_BLUR::coef_map(float coef){
    float x0 = 4;
    float y0 = 0.7;
    float x1 = 6;
    float y1 = 1.0;

    float k = (y1-y0)/(x1-x0);
    return k*(coef - x0)+y0;
}

float IQA_BLUR::run(cv::Mat &img_cv){
    if(m_filter_mat.empty()) return -1;
    Mat img_gray, img_padded, img_complex, img_mag;
    Mat planes[2];
    vector<Mat> vplanes;
    if(img_cv.channels()==4){
        cvtColor(img_cv, img_gray, COLOR_BGRA2GRAY);
    } else if (img_cv.channels()==3){
        cvtColor(img_cv, img_gray, COLOR_BGR2GRAY);
    } else {
        img_cv.copyTo(img_gray);
    }
    //如果图像大于filter, 则直接resize, 否则进行padding
    if(img_gray.rows<=m_rows&&img_gray.cols<=m_cols){
        copyMakeBorder(img_gray, img_padded, 0, m_rows - img_gray.rows, 0, m_cols - img_gray.cols, cv::BORDER_CONSTANT, Scalar::all(0));
    }
    else{
        resize(img_gray, img_padded, Size(m_cols,m_rows));
    }
    
    planes[0] = Mat_<float>(img_padded);
    planes[1] = Mat::zeros(img_padded.size(), CV_32F);
    merge(planes, 2, img_complex);
    dft(img_complex, img_complex);
    shiftDFT(img_complex);
    mulSpectrums(img_complex, m_filter_mat, img_complex, 0);
    shiftDFT(img_complex);
    
    split(img_complex, vplanes);
    magnitude(vplanes[0], vplanes[1],img_mag);
    img_mag += 1;
    log(img_mag, img_mag);
    Scalar _coef = mean(img_mag);
    float coef = _coef[0];
    return this->coef_map(coef);
}

////////////////////////////////////////////////////////////////////////////////////////////////
// IQA_BLUR END
////////////////////////////////////////////////////////////////////////////////////////////////