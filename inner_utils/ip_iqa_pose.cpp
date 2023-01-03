#include "ip_iqa_pose.hpp"

//new method for head pose restriction
float iqa_pose_tools::dist_p_line(float x0,float y0, float A, float B, float C){
/*
:param p: [x0,y0]
:param line: Ax+By+C=0 [A,B,C]
:return: d = | Ax0+By0+C | / sqrt(A^2+B^2)
*/
    float dm = fabsf(A*x0+B*y0+C);
    float dn = sqrtf(A*A+B*B+1e-3);
    return dm/dn;
}

std::vector<float> iqa_pose_tools::get_mid_line(float x1,float y1, float x2, float y2){
    float A = x1 - x2;
    float B = y1 - y2;
    float C = -(y1*y1-y2*y2 + x1*x1 - x2*x2)/2;
    std::vector<float> res;
    res.push_back(A);
    res.push_back(B);
    res.push_back(C);
    return res;
}

float iqa_pose_tools::dist_p_mid_line(float x0, float y0, float x1,float y1, float x2, float y2){
    std::vector<float> mid_line = get_mid_line(x1,y1,x2,y2);
    float A = mid_line[0];
    float B = mid_line[1];
    float C = mid_line[2];
    float d = dist_p_line(x0,y0,A,B,C);
    return d;
}

float iqa_pose_tools::dist_p_p(float x1,float y1,float x2,float y2){
    float d = sqrtf( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
    return d;
}

float iqa_pose_tools::ratio_p_mid_line(float x0, float y0, float x1,float y1, float x2, float y2){
    float dp1p2 = dist_p_p(x1,y1,x2,y2);
    float dp0line = dist_p_mid_line(x0,y0,x1,y1,x2,y2);
    return dp0line/(dp1p2+1e-3);
}
/////////////////////////////////////////////////////////////////////
// End of Head Pose Restriction
/////////////////////////////////////////////////////////////////////

/**
 * coef --> score
 * 0.2 --> 0.7
 * 0   --> 1
 */
float IQA_POSE::coef_map(float coef){
    float x0 = 0.2;
    float y0 = 0.7;
    float x1 = 0;
    float y1 = 1.0;

    float k = (y1-y0)/(x1-x0);
    return k*(coef - x0)+y0;
}

float IQA_POSE::run(ucloud::BBox &bbox){
    float ratio = iqa_pose_tools::ratio_p_mid_line(bbox.Pts.pts[2].x, bbox.Pts.pts[2].y, 
                                    bbox.Pts.pts[0].x, bbox.Pts.pts[0].y,
                                    bbox.Pts.pts[1].x, bbox.Pts.pts[1].y);

    return coef_map(ratio);
}