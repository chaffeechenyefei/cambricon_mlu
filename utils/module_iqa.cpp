#include "module_iqa.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include "../inner_utils/ip_iqa_pose.hpp"
#include <fstream>

#ifdef DEBUG
#include <chrono>
#include <sys/time.h>
#include "../inner_utils/module.hpp"
#endif

#ifdef VERBOSE
#define LOGI LOG(INFO)
#else
#define LOGI 0 && LOG(INFO)
#endif

#define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

using namespace ucloud;
using namespace cv;

RET_CODE IQA_Face_Evaluator::init(int dst_W, int dst_H){
    bool pad_both_side = true;
    bool keep_aspect_ratio = true;
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = YuvCropResizeModel::base_init( dst_H, dst_W, config);
    m_H = dst_H;
    m_W = dst_W;
    m_iqa_blur.init(dst_H, dst_W, dst_H/2+10, 2);
    return ret;
}

IQA_Face_Evaluator::~IQA_Face_Evaluator(){
    LOGI << "~IQA_Face_Evaluator()";
}

RET_CODE IQA_Face_Evaluator::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.clear();
    valid_clss.push_back(m_output_clss);
    return RET_CODE::SUCCESS;
}

RET_CODE IQA_Face_Evaluator::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes){
    // LOGI << "-> IQA_Face_Evaluator::run_yuv_on_mlu_phyAddr";
    RET_CODE ret = RET_CODE::FAILED;
    for(int i = 0; i < bboxes.size(); i++){
        if(bboxes[i].objtype !=m_output_clss) continue;
        TvaiRect roiRect = bboxes[i].rect;
        float aspect_ratio = 1.0;
        float aX, aY;
        Mat cropped_img;
        {
            std::lock_guard<std::mutex> lk(_mlu_mutex);
            ret = YuvCropResizeModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, cropped_img, aspect_ratio, aX , aY);
            if(ret!=RET_CODE::SUCCESS) return ret;
        }
        
        //TODO post process 
        ret = postprocess(cropped_img, bboxes[i]);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE IQA_Face_Evaluator::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes){
    // LOGI << "-> IQA_Face_Evaluator::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    for(int i = 0; i < bboxes.size(); i++){
        if(bboxes[i].objtype !=CLS_TYPE::FACE) continue;
        TvaiRect roiRect = bboxes[i].rect;
        float aspect_ratio = 1.0;
        float aX,aY;
        Mat cropped_img;
        {
            std::lock_guard<std::mutex> lk(_mlu_mutex);
            ret = YuvCropResizeModel::general_preprocess_yuv_on_mlu(tvimage, roiRect, cropped_img,aspect_ratio, aX , aY);
            if(ret!=RET_CODE::SUCCESS) return ret;
        }
        //TODO post process 
        ret = postprocess(cropped_img, bboxes[i]);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE IQA_Face_Evaluator::run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes){
    // LOGI << "-> IQA_Face_Evaluator::run_bgr_on_cpu";
    RET_CODE ret = RET_CODE::FAILED;
    std::vector<CLS_TYPE> valid_cls = {m_output_clss};
    std::vector<Mat> cropped_imgs;
    std::vector<float> aspect_ratios;
    {
        ret = YuvCropResizeModel::general_preprocess_infer_bgr_on_cpu(tvimage, bboxes, cropped_imgs, aspect_ratios, valid_cls);
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    //TODO post processing
    for(int i=0; i< cropped_imgs.size(); i++){
        if(cropped_imgs[i].empty()) continue;
        ret = postprocess(cropped_imgs[i], bboxes[i]);   
    }
    cropped_imgs.clear();
    return ret;
}

RET_CODE IQA_Face_Evaluator::run(TvaiImage& tvimage, VecObjBBox &bboxes){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
        ret = run_bgr_on_cpu(tvimage, bboxes);
    }
    else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        if(tvimage.usePhyAddr)
            ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes);
        else
            ret = run_yuv_on_mlu(tvimage,bboxes);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}


RET_CODE IQA_Face_Evaluator::postprocess(cv::Mat cropped_img, BBox &bbox){
    // LOGI << "-> IQA_Face_Evaluator::postprocess";
    float coef_pose = IQA_POSE::run(bbox);
    // std::cout << "iqa_pose " << coef_pose << std::endl;
    float coef_blur = m_iqa_blur.run(cropped_img);
    // std::cout << "iqa_blur " << coef_blur << std::endl;
    float coef = 0.6*coef_blur + 0.4*coef_pose;
    bbox.quality = coef;
    return RET_CODE::SUCCESS;
}