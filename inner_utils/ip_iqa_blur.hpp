/***
 * Image Processing DFT
 */
#ifndef _IP_IQA_BLUR_HPP_
#define _IP_IQA_BLUR_HPP_
#include <opencv2/opencv.hpp>

cv::Mat shiftDFT(cv::Mat &cv_img);

/**
 *创建低通/高通铝滤波器 
 */
cv::Mat create_butterworth_highpass_filter(int rows, int cols, int D, int n);
cv::Mat create_butterworth_lowpass_filter(int rows, int cols, int D, int n);

cv::Mat exec_butterworth_lowpass_filter(cv::Mat &cv_img, int D, int n);

//IQA图像质量评价之模糊度
class IQA_BLUR{
public:
    IQA_BLUR(){}
    IQA_BLUR(int rows, int cols, int D, int n){init(rows, cols, D, n);}
    ~IQA_BLUR(){}
    void init(int rows, int cols, int D, int n);
    float run(cv::Mat &img_cv);

protected:
    float coef_map(float coef);
    
private:
    cv::Mat m_filter_mat;
    int m_rows{0};
    int m_cols{0};
    int m_D{0};
    int m_n{0};
};

#endif