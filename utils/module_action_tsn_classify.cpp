#include "module_action_tsn_classify.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include <fstream>

#include "cluster.hpp"


#ifdef DEBUG
#include <chrono>
#include <sys/time.h>
#include "../inner_utils/module.hpp"
#endif


// #include <future>
using namespace ucloud;
using namespace cv;

using std::vector;

//inner function

/*******************************************************************************
TSNActionClassifyV4 动作分类
chaffee.chen@2022-10-10
*******************************************************************************/

RET_CODE TSNActionClassifyV4::init(const std::string &modelpath){
    LOGI << "-> TSNActionClassifyV4::init";
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    //Self param
    return ret;
}

RET_CODE TSNActionClassifyV4::init(std::map<InitParam, std::string> &modelpath){
    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    return init(modelpath[InitParam::BASE_MODEL]);
}

//clear self param
TSNActionClassifyV4::~TSNActionClassifyV4(){LOGI << "-> TSNActionClassifyV4::~TSNActionClassifyV4";}

RET_CODE TSNActionClassifyV4::run_yuv_on_mlu(BatchImageIN &batch_tvimages, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> TSNActionClassifyV4::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    std::vector<float> batch_aspect_ratio;

    VecObjBBox bboxes_pre_filter;
    VecRect batch_roi;
    // int _N = m_net->m_inputShape[0].BatchSize();
    // if(!m_pAoiRect.empty()){
    //     for(int i=0; i < _N ; i++ ){
    //         batch_roi.push_back(m_pAoiRect[0]);
    //     }
    // } //use whole image instead
    float** batch_model_output = nullptr;
    {        
        ret = m_net->general_batch_preprocess_yuv_on_mlu(batch_tvimages, batch_roi, batch_aspect_ratio);
        if(ret!=RET_CODE::SUCCESS) return ret;
        batch_model_output = m_net->general_mlu_infer();
    }
    //ATT. TSNActionClassifyV4 IN_BATCH_SIZE=8 OUT_BATCH_SIZE=1
    //TODO
    BBox bbox;
    bbox.rect = TvaiRect{0,0,batch_tvimages[0].width, batch_tvimages[0].height};
    postprocess(batch_model_output[0], bbox);
    m_net->cpu_free(batch_model_output);
    bboxes_pre_filter.push_back(bbox);
    postfilter(bboxes_pre_filter, bboxes, threshold);
    return ret;
}

RET_CODE TSNActionClassifyV4::run_yuv_on_mlu(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> TSNActionClassifyV4::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    std::vector<float> batch_aspect_ratio;

    VecObjBBox bboxes_pre_filter;
    VecRect batch_rects; 
    merge_batch_bboxes_to_rect(batch_bboxes, batch_rects);
    for(auto iter = batch_rects.begin(); iter!=batch_rects.end(); iter++ ){
        VecRect batch_roi;
        for(int i = 0; i < batch_tvimages.size(); i++){
            batch_roi.push_back(*iter);
        }
        float** batch_model_output = nullptr;
        {            
            ret = m_net->general_batch_preprocess_yuv_on_mlu(batch_tvimages, batch_roi, batch_aspect_ratio);
            if(ret!=RET_CODE::SUCCESS) return ret;
            batch_model_output = m_net->general_mlu_infer();
        }
        //ATT. TSNActionClassify IN_BATCH_SIZE=8 OUT_BATCH_SIZE=1
        //TODO
        BBox bbox;
        bbox.rect = *iter;
        postprocess(batch_model_output[0], bbox);
        m_net->cpu_free(batch_model_output);
        bboxes_pre_filter.push_back(bbox);
    }
    postfilter(bboxes_pre_filter, bboxes, threshold);
    return ret;
}


RET_CODE TSNActionClassifyV4::run(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(batch_tvimages.empty()) return ret;//return FAILED if input empty
    if(batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV21 || batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV12 ){
        threshold = clip_threshold(threshold);
        ret = run_yuv_on_mlu(batch_tvimages, batch_bboxes ,bboxes, threshold, nms_threshold);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE TSNActionClassifyV4::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(batch_tvimages.empty()) return ret;//return FAILED if input empty
    if(batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV21 || batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV12 ){
        threshold = clip_threshold(threshold);
        ret = run_yuv_on_mlu(batch_tvimages ,bboxes, threshold, nms_threshold);//little trick, use nullptr to judge the condition
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE TSNActionClassifyV4::postprocess(float* model_output, BBox &bbox){
    int featDim = m_net->m_outputShape[0].DataCount();//1,1,1,5 [normal, fight, fall, smoke]
    bbox.objtype = _cls_;
    bbox.confidence = model_output[1];
    bbox.objectness = bbox.confidence;
    return RET_CODE::SUCCESS;
}

RET_CODE TSNActionClassifyV4::postfilter(VecObjBBox &ins, VecObjBBox &outs, float threshold){
    for(auto iter=ins.begin(); iter!=ins.end(); iter++){
        if(iter->confidence > threshold ){
            outs.push_back(*iter);
        }
    }
    return RET_CODE::SUCCESS;
}

RET_CODE TSNActionClassifyV4::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(_cls_);
    return RET_CODE::SUCCESS;
};


void TSNActionClassifyV4::merge_batch_bboxes_to_rect(BatchBBoxIN &batch_bboxes, VecRect &rects){
    //Step1. Cluster to rect in each image
    BatchBBoxOUT batch_bboxes_filter;
    for(auto biter=batch_bboxes.begin(); biter!=batch_bboxes.end(); biter++){
        VecObjBBox bboxes;
        for( auto iter=biter->begin(); iter!=biter->end(); iter++){
            if(iter->objtype == CLS_TYPE::PEDESTRIAN)
                bboxes.push_back(*iter);
        }
        batch_bboxes_filter.push_back(bboxes);
    }
    ClusterImageSetLevel clusterHandle(m_threshold_cluster);
    clusterHandle.insert(batch_bboxes_filter);
    clusterHandle.merge();
    rects = clusterHandle.getROI();
}

float TSNActionClassifyV4::clip_threshold(float x){
    if(x < 0) return m_default_threshold_fight;
    if(x > 1) return m_default_threshold_fight;
    return x;
}

/*******************************************************************************
TSNActionClassify 动作分类
chaffee.chen@2021
*******************************************************************************/
// RET_CODE TSNActionClassify::init(const std::string &modelpath){
//     LOGI << "-> TSNActionClassify::init";
//     bool pad_both_side = true;//两边预留
//     bool keep_aspect_ratio = true;//保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);
//     //Self param
//     return ret;
// }

// RET_CODE TSNActionClassify::init(std::map<InitParam, std::string> &modelpath){
//     if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
//     return init(modelpath[InitParam::BASE_MODEL]);
// }

// //clear self param
// TSNActionClassify::~TSNActionClassify(){LOGI << "-> TSNActionClassify::~TSNActionClassify";}

// RET_CODE TSNActionClassify::run_yuv_on_mlu(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
//     LOGI << "-> TSNActionClassify::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     std::vector<float> batch_aspect_ratio;

//     VecObjBBox bboxes_pre_filter;
//     VecRect batch_roi;
//     if(!m_pAoiRect.empty()){
//         for(int i=0; i < _N ; i++ ){
//             batch_roi.push_back(m_pAoiRect[0]);
//         }
//     }
//     float* batch_model_output = nullptr;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
        
//         ret = BaseModel::general_batch_preprocess_yuv_on_mlu(batch_tvimages, batch_roi, batch_aspect_ratio);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         batch_model_output = BaseModel::general_mlu_infer();
//     }
//     //ATT. TSNActionClassify IN_BATCH_SIZE=8 OUT_BATCH_SIZE=1
//     int outputDim = _oW*_oH*_oC;
//     //TODO
//     BBox bbox;
//     bbox.rect = TvaiRect{0,0,batch_tvimages[0].width, batch_tvimages[0].height};
//     postprocess(batch_model_output, bbox);
//     free(batch_model_output);
//     bboxes_pre_filter.push_back(bbox);
//     postfilter(bboxes_pre_filter, bboxes);
//     return ret;
// }

// RET_CODE TSNActionClassify::run_yuv_on_mlu(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes, VecObjBBox &bboxes){
//     LOGI << "-> TSNActionClassify::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     std::vector<float> batch_aspect_ratio;

//     VecObjBBox bboxes_pre_filter;
//     VecRect batch_rects; 
//     merge_batch_bboxes_to_rect(batch_bboxes, batch_rects);
//     for(auto iter = batch_rects.begin(); iter!=batch_rects.end(); iter++ ){
//         VecRect batch_roi;
//         for(int i = 0; i < batch_tvimages.size(); i++){
//             batch_roi.push_back(*iter);
//         }
//         float* batch_model_output = nullptr;
//         {
//             std::lock_guard<std::mutex> lk(_mlu_mutex);
            
//             ret = BaseModel::general_batch_preprocess_yuv_on_mlu(batch_tvimages, batch_roi, batch_aspect_ratio);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             batch_model_output = BaseModel::general_mlu_infer();
//         }
//         //ATT. TSNActionClassify IN_BATCH_SIZE=8 OUT_BATCH_SIZE=1
//         int outputDim = _oW*_oH*_oC;
//         //TODO
//         BBox bbox;
//         bbox.rect = *iter;
//         postprocess(batch_model_output, bbox);
//         free(batch_model_output);
//         bboxes_pre_filter.push_back(bbox);
//     }
//     postfilter(bboxes_pre_filter, bboxes);
//     return ret;
// }


// RET_CODE TSNActionClassify::run(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes, VecObjBBox &bboxes){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(batch_tvimages.empty()) return ret;//return FAILED if input empty
//     if(batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV21 || batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV12 ){
//         ret = run_yuv_on_mlu(batch_tvimages, batch_bboxes ,bboxes);
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     return ret;
// }

// RET_CODE TSNActionClassify::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes, float threshold, float nms_threshold){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(batch_tvimages.empty()) return ret;//return FAILED if input empty
//     if(batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV21 || batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV12 ){
//         ret = run_yuv_on_mlu(batch_tvimages ,bboxes);//little trick, use nullptr to judge the condition
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     return ret;
// }

// RET_CODE TSNActionClassify::postprocess(float* model_output, BBox &bbox){
//     int featDim = _oH*_oW*_oC;//1,1,1,5 [normal, fight, fall, smoke]
//     bbox.objtype = _cls_;
//     bbox.confidence = model_output[1];
//     bbox.objectness = bbox.confidence;
//     return RET_CODE::SUCCESS;
// }

// RET_CODE TSNActionClassify::postfilter(VecObjBBox &ins, VecObjBBox &outs){
//     for(auto iter=ins.begin(); iter!=ins.end(); iter++){
//         if(iter->confidence > m_threshold_fight ){
//             outs.push_back(*iter);
//         }
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE TSNActionClassify::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss.push_back(_cls_);
//     return RET_CODE::SUCCESS;
// };

// RET_CODE TSNActionClassify::set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, 
// std::vector<TvaiRect> &pAoiRect){
//     if(float_in_range(threshold,1,0))
//         m_threshold_fight = threshold;
//     else
//         return RET_CODE::ERR_INIT_PARAM_FAILED;
//     if(!pAoiRect.empty()){
//         m_pAoiRect = pAoiRect;
//     }
//     return RET_CODE::SUCCESS;    
// }


// void TSNActionClassify::merge_batch_bboxes_to_rect(BatchBBoxIN &batch_bboxes, VecRect &rects){
//     //Step1. Cluster to rect in each image
//     BatchBBoxOUT batch_bboxes_filter;
//     for(auto biter=batch_bboxes.begin(); biter!=batch_bboxes.end(); biter++){
//         VecObjBBox bboxes;
//         for( auto iter=biter->begin(); iter!=biter->end(); iter++){
//             if(iter->objtype == CLS_TYPE::PEDESTRIAN)
//                 bboxes.push_back(*iter);
//         }
//         batch_bboxes_filter.push_back(bboxes);
//     }
//     ClusterImageSetLevel clusterHandle(m_threshold_cluster);
//     clusterHandle.insert(batch_bboxes_filter);
//     clusterHandle.merge();
//     rects = clusterHandle.getROI();
// }