#include "module_obj_feature_extraction.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include <fstream>

// #include <future>
using namespace ucloud;
using namespace cv;


/*******************************************************************************
ObjFeatureExtractionV2 使用BaseModelV2
*******************************************************************************/
ObjFeatureExtractionV2::ObjFeatureExtractionV2(){
    m_net = std::make_shared<BaseModelV2>();
}

RET_CODE ObjFeatureExtractionV2::init(WeightData wData){
    LOGI << "-> ObjFeatureExtractionV2::init";
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::ARGB, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(wData, config);
    LOGI << "<- ObjFeatureExtractionV2::init";
    return ret;
}

RET_CODE ObjFeatureExtractionV2::init(const std::string &modelpath){
    LOGI << "-> ObjFeatureExtractionV2::init";
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::ARGB, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    LOGI << "<- ObjFeatureExtractionV2::init";
    return ret;
}

RET_CODE ObjFeatureExtractionV2::init(const std::string &modelpath, MODEL_INPUT_FORMAT inpFMT, bool keep_aspect_ratio, bool pad_both_side){
    LOGI << "-> ObjFeatureExtractionV2::init";
    BASE_CONFIG config(inpFMT, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    return ret;
}

RET_CODE ObjFeatureExtractionV2::run(TvaiImage& tvimage, VecObjBBox &bboxes){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu_batch(tvimage, bboxes);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE ObjFeatureExtractionV2::run_yuv_on_mlu_batch(TvaiImage &tvaiImage, VecObjBBox &bboxes){
    LOGI << "-> ObjFeatureExtraction::run_yuv_on_mlu_batch";
    RET_CODE ret = RET_CODE::SUCCESS;
    int batch_size = m_net->m_inputShape[0].BatchSize();
    for (int i = 0; i < bboxes.size();){
        VecObjBBox batch_bboxes;
        std::vector<float> aspect_ratios;
        int actual_batch_size = batch_size;
        if(i+batch_size>bboxes.size())
            actual_batch_size = bboxes.size()%batch_size;
        ret = m_net->general_batch_preprocess_yuv_on_mlu(tvaiImage, bboxes, aspect_ratios, i );
        if(ret!=RET_CODE::SUCCESS) return ret;
        float** model_batch_outputs = m_net->general_mlu_infer();
        float* model_batch_output = model_batch_outputs[0];
        int featDim = m_net->m_outputShape[0].DataCount();
        for(int j = 0 ; j < actual_batch_size ; j++ ){
            float* model_output = model_batch_output+j*featDim;
            postprocess(model_output, bboxes[i+j]);
        }
        m_net->cpu_free(model_batch_outputs);
        i += batch_size;
    }
    return ret;
}

RET_CODE ObjFeatureExtractionV2::postprocess(float* model_output, BBox &bbox){
    int featDim = m_net->m_outputShape[0].DataCount();
    normalize_l2_unit(model_output,featDim);
    bbox.trackfeat.resize(featDim);
    memcpy(&(bbox.trackfeat[0]), model_output , featDim*sizeof(float));
    return RET_CODE::SUCCESS;
}

/*******************************************************************************
ObjFeatureExtraction 继承 BaseModel
*******************************************************************************/
// RET_CODE ObjFeatureExtraction::init(const std::string &modelpath){
//     LOGI << "-> ObjFeatureExtraction::init";
//     bool pad_both_side = true;//两边预留
//     bool keep_aspect_ratio = true;//保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::ARGB, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);
//     //Self param
//     return ret;
// }

// RET_CODE ObjFeatureExtraction::init(const std::string &modelpath, MODEL_INPUT_FORMAT inpFMT, bool keep_aspect_ratio, bool pad_both_side){
//     LOGI << "-> ObjFeatureExtraction::init2";
//     BASE_CONFIG config(inpFMT, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);
//     //Self param
//     return ret;
// }

// //clear self param
// ObjFeatureExtraction::~ObjFeatureExtraction(){LOGI << "-> ObjFeatureExtraction::~ObjFeatureExtraction";}

// RET_CODE ObjFeatureExtraction::run_yuv_on_mlu_batch(TvaiImage &tvaiImage, VecObjBBox &bboxes){
//     LOGI << "-> ObjFeatureExtraction::run_yuv_on_mlu_batch";
//     RET_CODE ret = RET_CODE::SUCCESS;
//     int batch_size = _N;
//     for (int i = 0; i < bboxes.size();){
//         VecObjBBox batch_bboxes;
//         std::vector<float> aspect_ratios;
//         int actual_batch_size = batch_size;
//         if(i+batch_size>bboxes.size())
//             actual_batch_size = bboxes.size()%batch_size;
//         ret = BaseModel::general_batch_preprocess_yuv_on_mlu(tvaiImage, bboxes, aspect_ratios, i );
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         float* model_batch_output = BaseModel::general_mlu_infer();
//         int featDim = _oC*_oH*_oW;
//         for(int j = 0 ; j < actual_batch_size ; j++ ){
//             float* model_output = model_batch_output+j*featDim;
//             postprocess(model_output, bboxes[i+j]);
//         }
//         free(model_batch_output);
//         i += batch_size;
//     }
//     return ret;
// }

// RET_CODE ObjFeatureExtraction::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> ObjFeatureExtraction::run_yuv_on_mlu_phyAddr";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         TvaiRect roiRect = bboxes[i].rect;
//         float aspect_ratio = 1.0;
//         float aX, aY;
//         std::shared_ptr<float> model_output;
//         {
//             // std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             model_output = BaseModel::general_mlu_infer_share_ptr();
//         }
        
//         //TODO post process 
//         ret = postprocess(model_output.get(),bboxes[i]);
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE ObjFeatureExtraction::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> ObjFeatureExtraction::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         TvaiRect roiRect = bboxes[i].rect;
//         float aspect_ratio = 1.0;
//         float aX,aY;
//         std::shared_ptr<float> model_output;
//         {
//             // std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu(tvimage, roiRect, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             model_output = BaseModel::general_mlu_infer_share_ptr();
//         }
//         //TODO post process 
//         ret = postprocess(model_output.get(),bboxes[i]);
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE ObjFeatureExtraction::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
//     if(batch_tvimages.empty()) return RET_CODE::SUCCESS;
//     return run(batch_tvimages[0], bboxes);
// }

// RET_CODE ObjFeatureExtraction::run(TvaiImage& tvimage, VecObjBBox &bboxes){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
//         ret = run_yuv_on_mlu_batch(tvimage, bboxes);
//         // if(tvimage.usePhyAddr)
//         //     ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes);
//         // else
//         //     ret = run_yuv_on_mlu(tvimage,bboxes);
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     return ret;
// }

// RET_CODE ObjFeatureExtraction::postprocess(float* model_output, BBox &bbox){
//     int featDim = _oH*_oW*_oC;
//     normalize_l2_unit(model_output,featDim);
//     bbox.trackfeat.resize(featDim);
//     memcpy(&(bbox.trackfeat[0]), model_output , featDim*sizeof(float));
//     return RET_CODE::SUCCESS;
// }

// RET_CODE ObjFeatureExtraction::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss.push_back(_cls_);
//     return RET_CODE::SUCCESS;
// };


