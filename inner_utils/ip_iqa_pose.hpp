/***
 * Image Processing IQA
 */
#ifndef _IP_IQA_POSE_HPP_
#define _IP_IQA_POSE_HPP_
#include <opencv2/opencv.hpp>
#include "../libai_core.hpp"
/**
 * 只支持人脸左右旋转的角度判断
 */
class IQA_POSE{
public:
    IQA_POSE(){}
    ~IQA_POSE(){}
    static float run(ucloud::BBox &bbox);
protected:
    static float coef_map(float coef);
};

class iqa_pose_tools{
public:
    iqa_pose_tools(){}
    ~iqa_pose_tools(){}
    static float dist_p_line(float x0,float y0, float A, float B, float C);
    static std::vector<float> get_mid_line(float x1,float y1, float x2, float y2);
    static float dist_p_mid_line(float x0, float y0, float x1,float y1, float x2, float y2);
    static float dist_p_p(float x1,float y1,float x2,float y2);
    static float ratio_p_mid_line(float x0, float y0, float x1,float y1, float x2, float y2);
};

#endif