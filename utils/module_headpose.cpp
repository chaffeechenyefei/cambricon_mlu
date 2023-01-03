#include "module_headpose.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>

using namespace ucloud;
using namespace cv;
using std::vector;

/*******************************************************************************
innser function
*******************************************************************************/
/**
map (v1,v2) -> (nv1, nv2) linearly
 */
static float coef_map(float coef, float v1, float nv1, float v2, float nv2){
    float k = (nv2-nv1)/(v1-v2);
    return k*(coef - v1)+nv1;
}

/*******************************************************************************
HeadPoseEvaluationV4 人头角度计算
itten.hu@2022-09
chaffee.chen@2022-10-08
*******************************************************************************/
RET_CODE HeadPoseEvaluationV4::init(const std::string &modelpath){
    LOGI <<  "-> HeadPoseEvaluationV4::init";
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    //Self param
    return ret;
}

RET_CODE HeadPoseEvaluationV4::init(std::map<InitParam, std::string> &modelpath){
    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath[InitParam::BASE_MODEL], config);
    //Self param
    return ret;
}

HeadPoseEvaluationV4::~HeadPoseEvaluationV4(){LOGI << "-> HeadPoseEvaluationV4::~HeadPoseEvaluationV4";}

RET_CODE HeadPoseEvaluationV4::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes){
    LOGI << "-> HeadPoseEvaluationV4::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    for(int i = 0; i < bboxes.size(); i++){
        if(bboxes[i].objtype !=CLS_TYPE::FACE) continue;
        TvaiRect roiRect = bboxes[i].rect;
        float aspect_ratio = 1.0;
        float aX, aY;
        float** model_output = nullptr;
        {
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roiRect, aspect_ratio, aX , aY);
            if(ret!=RET_CODE::SUCCESS) return ret;
            model_output = m_net->general_mlu_infer();
        }
        
        //TODO post process 
        ret = postprocess(model_output[0], bboxes[i]);
        m_net->cpu_free(model_output);
    }
    return RET_CODE::SUCCESS;
}



RET_CODE HeadPoseEvaluationV4::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        // ret = run_bgr_on_cpu(tvimage, bboxes);
    }
    else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu(tvimage,bboxes);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE HeadPoseEvaluationV4::postprocess(float* model_output, BBox &bbox){
    float yaw = model_output[0];
    float pitch = model_output[1];
    float roll = model_output[2];
    float yaw_score = 0.6; float pitch_score = 0.6; float roll_score = 0.6;
    if(yaw>=0) yaw_score = coef_map(yaw, m_yaw_tp, 0.6, 0, 1.0);
    else yaw_score = coef_map(yaw, m_yaw_bt, 0.6, 0, 1.0);

    if(pitch>=0) pitch_score = coef_map(pitch, m_pitch_tp, 0.6, 0, 1.0);
    else pitch_score = coef_map(pitch, m_pitch_bt, 0.6, 0, 1.0);

    if(roll>=0) roll_score = coef_map(roll, m_roll_tp, 0.6, 0, 1.0);
    else roll_score = coef_map(roll, m_roll_bt, 0.6, 0, 1.0);

    float pose_score = std::min(std::min(yaw_score, pitch_score), roll_score);

    bbox.quality = pose_score;

    return RET_CODE::SUCCESS;
}

RET_CODE HeadPoseEvaluationV4::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(_cls_);
    return RET_CODE::SUCCESS;
};

RET_CODE HeadPoseEvaluationV4::set_valid_pose(float yaw_tp, float yaw_bt, float pitch_tp, 
float pitch_bt, float roll_tp, float roll_bt){
    m_yaw_tp = yaw_tp;
    m_yaw_bt = yaw_bt;
    m_pitch_bt = pitch_bt;
    m_pitch_tp = pitch_tp;
    m_roll_bt = roll_bt;
    m_roll_tp = roll_tp;
    return RET_CODE::SUCCESS;
}

/*******************************************************************************
HeadPoseEvaluation 人头角度计算 继承BaseModel
itten.hu@2022-09
chaffee.chen@2022-10-08
*******************************************************************************/
// RET_CODE HeadPoseEvaluation::init(const std::string &modelpath){
//     LOGI <<  "-> HeadPoseEvaluation::init";
//     bool pad_both_side = true;//两边预留
//     bool keep_aspect_ratio = true;//保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);
//     //Self param
//     m_iqa_blur.init(_H, _W, _H/2+10, 2);
//     return ret;
// }

// RET_CODE HeadPoseEvaluation::init(std::map<InitParam, std::string> &modelpath){
//     if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
//     bool pad_both_side = true;//两边预留
//     bool keep_aspect_ratio = true;//保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath[InitParam::BASE_MODEL], config);
//     //Self param
//     m_iqa_blur.init(_H, _W, _H/2+10, 2);
//     return ret;
// }

// HeadPoseEvaluation::~HeadPoseEvaluation(){LOGI << "-> HeadPoseEvaluation::~HeadPoseEvaluation";}

// RET_CODE HeadPoseEvaluation::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> HeadPoseEvaluation::run_yuv_on_mlu_phyAddr";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         if(bboxes[i].objtype !=CLS_TYPE::FACE) continue;
//         TvaiRect roiRect = bboxes[i].rect;
//         float aspect_ratio = 1.0;
//         float aX, aY;
//         float* model_output = nullptr;
//         float blur_score = 0.7;
//         {
//             std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             blur_score = blur_evaluate_mlu();
//             model_output = BaseModel::general_mlu_infer();
//         }
        
//         //TODO post process 
//         ret = postprocess(model_output, blur_score,bboxes[i]);
//         free(model_output);
//     }
//     return RET_CODE::SUCCESS;
// }


// RET_CODE HeadPoseEvaluation::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> HeadPoseEvaluation::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         if(bboxes[i].objtype !=CLS_TYPE::FACE) continue;
//         TvaiRect roiRect = bboxes[i].rect;
//         float aspect_ratio = 1.0;
//         float aX,aY;
//         float* model_output = nullptr;
//         float blur_score = 0.7;
//         {
//             std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu(tvimage, roiRect, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             blur_score = blur_evaluate_mlu();
//             model_output = BaseModel::general_mlu_infer();
//         }
//         //TODO post process 
//         ret = postprocess(model_output, blur_score,bboxes[i]);
//         free(model_output);
//     }
//     return RET_CODE::SUCCESS;
// }

// // RET_CODE HeadPoseEvaluation::run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes){
// //     LOGI << "-> HeadPoseEvaluation::run_bgr_on_cpu";
// //     RET_CODE ret = RET_CODE::FAILED;

// //     std::vector<CLS_TYPE> valid_cls = {CLS_TYPE::FACE};
// //     std::vector<float*> model_outputs;
// //     std::vector<float> aspect_ratios;
// //     {
// //         std::lock_guard<std::mutex> lk(_mlu_mutex);
// //         ret = BaseModel::general_preprocess_infer_bgr_on_cpu(tvimage, bboxes,model_outputs, aspect_ratios, valid_cls);
// //     }
// //     if(ret!=RET_CODE::SUCCESS) return ret;
// //     //TODO post processing
// //     for(int i=0; i< model_outputs.size(); i++){
// //         if(model_outputs[i]==nullptr) continue;
// //         // ret = postprocess(model_outputs[i], bboxes[i]);
// //         free(model_outputs[i]);   
// //     }
// //     model_outputs.clear();
// //     return ret;
// // }



// RET_CODE HeadPoseEvaluation::run(TvaiImage& tvimage, VecObjBBox &bboxes){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//         // ret = run_bgr_on_cpu(tvimage, bboxes);
//     }
//     else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
//         if(tvimage.usePhyAddr)
//             ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes);
//         else
//             ret = run_yuv_on_mlu(tvimage,bboxes);
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     return ret;
// }

// float HeadPoseEvaluation::blur_evaluate_mlu(){
//     Mat cvimage(_H,_W,CV_8UC4);
//     cnrtMemcpy(cvimage.data, mlu_input_[0], _C*_H*_W , CNRT_MEM_TRANS_DIR_DEV2HOST);
//     float blur_score = m_iqa_blur.run(cvimage);
//     return blur_score;
// }

// RET_CODE HeadPoseEvaluation::postprocess(float* model_output, float blur_score, BBox &bbox){
//     float yaw = model_output[0];
//     float pitch = model_output[1];
//     float roll = model_output[2];
//     float yaw_score = 0.6; float pitch_score = 0.6; float roll_score = 0.6;
//     if(yaw>=0) yaw_score = coef_map(yaw, m_yaw_tp, 0.6, 0, 1.0);
//     else yaw_score = coef_map(yaw, m_yaw_bt, 0.6, 0, 1.0);

//     if(pitch>=0) pitch_score = coef_map(pitch, m_pitch_tp, 0.6, 0, 1.0);
//     else pitch_score = coef_map(pitch, m_pitch_bt, 0.6, 0, 1.0);

//     if(roll>=0) roll_score = coef_map(roll, m_roll_tp, 0.6, 0, 1.0);
//     else roll_score = coef_map(roll, m_roll_bt, 0.6, 0, 1.0);

//     float pose_score = std::min(std::min(yaw_score, pitch_score), roll_score);

//     bbox.quality = 0.4*pose_score + 0.6*blur_score;

//     return RET_CODE::SUCCESS;
// }

// RET_CODE HeadPoseEvaluation::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss.push_back(_cls_);
//     return RET_CODE::SUCCESS;
// };

// RET_CODE HeadPoseEvaluation::set_valid_pose(float yaw_tp, float yaw_bt, float pitch_tp, 
// float pitch_bt, float roll_tp, float roll_bt){
//     m_yaw_tp = yaw_tp;
//     m_yaw_bt = yaw_bt;
//     m_pitch_bt = pitch_bt;
//     m_pitch_tp = pitch_tp;
//     m_roll_bt = roll_bt;
//     m_roll_tp = roll_tp;
//     return RET_CODE::SUCCESS;
// }