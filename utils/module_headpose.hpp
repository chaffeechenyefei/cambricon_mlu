#ifndef _MODULE_HEADPOSE_HPP_
#define _MODULE_HEADPOSE_HPP_

#include "module_base.hpp"
// #include "../inner_utils/ip_iqa_blur.hpp"
#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
//人头角度
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

/*******************************************************************************
HeadPoseEvaluationV4 人头角度计算
itten.hu@2022-09
chaffee.chen@2022-10-08
*******************************************************************************/
class HeadPoseEvaluationV4: public AlgoAPI{
public:
    HeadPoseEvaluationV4(){
        m_net = std::make_shared<BaseModelV2>();
    }
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~HeadPoseEvaluationV4();
    RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
    RET_CODE set_valid_pose(float yaw_tp, float yaw_bt, float pitch_tp, float pitch_bt, float roll_tp, float roll_bt);
private:
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes);
    //后处理单元
    RET_CODE postprocess(float* model_output, BBox &bbox);

    CLS_TYPE _cls_ = CLS_TYPE::FACE;
    MLUNet_Ptr m_net = nullptr;

    float m_yaw_tp{30};
    float m_yaw_bt{-30};
    float m_pitch_tp{22};
    float m_pitch_bt{-32};
    float m_roll_tp{30};
    float m_roll_bt{-30};
};


/*******************************************************************************
HeadPoseEvaluation 人头角度计算 继承BaseModel
itten.hu@2022-09
chaffee.chen@2022-10-08
*******************************************************************************/
// class HeadPoseEvaluation: public BaseModel{
// public:
//     HeadPoseEvaluation(){};
//     RET_CODE init(const std::string &modelpath);
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     ~HeadPoseEvaluation();
//     RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
//     RET_CODE set_valid_pose(float yaw_tp, float yaw_bt, float pitch_tp, float pitch_bt, float roll_tp, float roll_bt);
// private:
//     RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes);
//     // RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes);

//     float blur_evaluate_mlu();
//     //后处理单元
//     RET_CODE postprocess(float* model_output, float blur_score, BBox &bbox);

//     CLS_TYPE _cls_ = CLS_TYPE::FACE;

//     float m_yaw_tp{30};
//     float m_yaw_bt{-30};
//     float m_pitch_tp{22};
//     float m_pitch_bt{-32};
//     float m_roll_tp{30};
//     float m_roll_bt{-30};

//     IQA_BLUR m_iqa_blur;
// };


#endif