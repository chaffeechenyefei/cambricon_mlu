#include "module_yolo_detection_v2.hpp"
#include "module_base.hpp"

#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "../inner_utils/inner_basic.hpp"
#include <fstream>
#include <iostream>

#ifdef VERBOSE
#define LOGI LOG(INFO)
#else
#define LOGI 0 && LOG(INFO)
#endif


YoloDetectionV4::YoloDetectionV4(){
    m_net = std::make_shared<BaseModelV2>();
}

YoloDetectionV4::~YoloDetectionV4(){
    LOGI << "~YoloDetectionV4()";
}

RET_CODE YoloDetectionV4::init(std::map<InitParam, WeightData> &weightConfig){
    LOGI << "-> YoloDetectionV4::init";
    if(weightConfig.find(InitParam::BASE_MODEL) == weightConfig.end())
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    bool pad_both_side = false;
    bool keep_aspect_ratio = true;
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);

    RET_CODE ret = m_net->base_init(weightConfig[InitParam::BASE_MODEL], config);
    if (ret!=RET_CODE::SUCCESS) {
        printf("RET_CODE ret = BaseModel::base_init( modelpath[InitParam::BASE_MODEL], config); return %d\n",ret);
        return ret;
    }

    if(_output_cls_num == 0)
        _output_cls_num = m_net->m_outputShape[0].H() - m_dimOffset;
    else{
        if(_output_cls_num!=m_net->m_outputShape[0].H() - m_dimOffset){
            printf("_output_cls_num [%d] != models output shape [%d]\n", _output_cls_num, m_net->m_outputShape[0].H() - 5);
            ret = RET_CODE::FAILED;
        }
    }
    LOGI << "<- YoloDetectionV4::init";
    return RET_CODE::SUCCESS;
}


RET_CODE YoloDetectionV4::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> YoloDetectionV4::init";
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) {
            printf("auto_model_file_search failed, return %d\n",ret);
            return ret;
        }
    }

    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end())
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    bool pad_both_side = false;
    bool keep_aspect_ratio = true;
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);

    RET_CODE ret = m_net->base_init(modelpath[InitParam::BASE_MODEL], config);
    if (ret!=RET_CODE::SUCCESS) {
        printf("RET_CODE ret = BaseModel::base_init( modelpath[InitParam::BASE_MODEL], config); return %d\n",ret);
        return ret;
    }

    if(_output_cls_num == 0)
        _output_cls_num = m_net->m_outputShape[0].H() - m_dimOffset;
    else{
        if(_output_cls_num!=m_net->m_outputShape[0].H() - m_dimOffset){
            printf("_output_cls_num [%d] != models output shape [%d]\n", _output_cls_num, m_net->m_outputShape[0].H() - 5);
            ret = RET_CODE::FAILED;
        }
    }

    

    LOGI << "<- YoloDetectionV4::init";
    return ret;
}

RET_CODE YoloDetectionV4::init(const std::string &modelpath){
    std::map<InitParam, std::string> modelConfig = {{InitParam::BASE_MODEL, modelpath}};
    return this->init(modelConfig);
}

RET_CODE YoloDetectionV4::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);
    // BYTETRACKPARM track_param = {threshold, threshold+0.1f};
    LOGI << "-> YoloDetectionV4::threshold =" << threshold;

    float preprocess_time{0}, npu_inference_time{0}, postprocess_time{0};

    float aspect_ratio = 1.0;
    float aX,aY;
    float** model_output = nullptr;
    {
        m_Tk.start();
        if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
            ret = m_net->general_preprocess_bgr_on_cpu(tvimage, aspect_ratio, aX , aY);
        } else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, TvaiRect{0,0,tvimage.width, tvimage.height}, aspect_ratio, aX, aY);
        } else
            ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;    
        preprocess_time = m_Tk.end("preprocess", false);
        if(ret!=RET_CODE::SUCCESS) return ret;
        m_Tk.start();
        model_output = m_net->general_mlu_infer();
        npu_inference_time = m_Tk.end("npu inference", false);
    }
    m_Tk.start();
    ret = postprocess(model_output[0], bboxes, threshold, nms_threshold, 1.0, aspect_ratio, tvimage.width, tvimage.height);
    postprocess_time = m_Tk.end("postprocess", false);
    m_net->cpu_free(model_output);

    if(!bboxes.empty()){
        bboxes[0].tmInfo = {preprocess_time, npu_inference_time, postprocess_time};
    }
    
    if(ret!=RET_CODE::SUCCESS) return ret;
    /**Tracking can be added here**/
    return ret;
}

RET_CODE YoloDetectionV4::run(TvaiImage &tvimage, TvaiRect tvrect, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);
    float aspect_ratio = 1.0;
    float aX,aY;
    float** model_output = nullptr;
    {
        if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
            ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        } else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, tvrect, aspect_ratio, aX, aY);
        } else
            ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;    
        if(ret!=RET_CODE::SUCCESS) return ret;
        model_output = m_net->general_mlu_infer();
    }
    ret = postprocess(model_output[0], bboxes, tvrect, threshold, nms_threshold, 1.0, aspect_ratio, tvimage.width, tvimage.height);
    m_net->cpu_free(model_output);
    if(ret!=RET_CODE::SUCCESS) return ret;

    return ret;
}

float YoloDetectionV4::clip_threshold(float x){
    if(x < 0) return m_default_threshold;
    if(x > 1) return m_default_threshold;
    return x;
}
float YoloDetectionV4::clip_nms_threshold(float x){
    if(x < 0) return m_default_nms_threshold;
    if(x > 1) return m_default_nms_threshold;
    return x;
}

RET_CODE YoloDetectionV4::postprocess(float* model_output, VecObjBBox &bboxes, float threshold, float nms_threshold ,
    float expand_ratio, float aspect_ratio, int imgW, int imgH)
{
    LOGI << "-> YoloDetectionV4::postprocess";
    int nBBox = m_net->m_outputShape[0].C();
    int featLen = m_net->m_outputShape[0].H();
    std::vector<VecObjBBox> vecBox;
    VecObjBBox vecBox_after_nms;
    if(!m_output_cls_order.empty())
        base_output2ObjBox_multiCls(model_output, vecBox, m_output_cls_order, _unique_cls_order , nBBox, featLen, threshold, m_dimOffset);
    else if(!m_output_cls_order_str.empty())
        base_output2ObjBox_multiCls(model_output, vecBox, m_output_cls_order_str, _unique_cls_order_str , nBBox, featLen, threshold, m_dimOffset);
    else {
        printf("**ERR[%s][%d]: The meaning of model output's is not defined.\n", __FILE__, __LINE__);
        return RET_CODE::FAILED;
    }        
    base_nmsBBox(vecBox,nms_threshold, NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aspect_ratio);
    std::vector<VecObjBBox>().swap(vecBox);
    bboxes.insert(bboxes.end(), vecBox_after_nms.begin(), vecBox_after_nms.end());
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    LOGI << "<- YoloDetectionV4::postprocess";
    return RET_CODE::SUCCESS;     
}

static void shift_box_from_roi_to_org(VecObjBBox &bboxes, TvaiRect &roirect){
    for(auto &&bbox: bboxes){
        bbox.rect.x += roirect.x;
        bbox.rect.y += roirect.y;
    }
}
RET_CODE YoloDetectionV4::postprocess(float* model_output, VecObjBBox &bboxes, TvaiRect tvrect, float threshold, float nms_threshold, 
    float expand_ratio, float aspect_ratio, int imgW, int imgH)
{
    LOGI << "-> YoloDetectionV2::postprocess";
    int nBBox = m_net->m_outputShape[0].C();
    int featLen = m_net->m_outputShape[0].H();
    std::vector<VecObjBBox> vecBox;
    VecObjBBox vecBox_after_nms;
    if(!m_output_cls_order.empty())
        base_output2ObjBox_multiCls(model_output, vecBox, m_output_cls_order, _unique_cls_order , nBBox, featLen, threshold, m_dimOffset);
    if(!m_output_cls_order_str.empty())
        base_output2ObjBox_multiCls(model_output, vecBox, m_output_cls_order_str, _unique_cls_order_str , nBBox, featLen, threshold, m_dimOffset);
    if(m_output_cls_order.empty() && m_output_cls_order_str.empty()) {
        printf("**ERR[%s][%d]: The meaning of model output's is not defined.\n", __FILE__, __LINE__);
        return RET_CODE::FAILED;
    }    
    base_nmsBBox(vecBox,nms_threshold, NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aspect_ratio);
    shift_box_from_roi_to_org(vecBox_after_nms, tvrect);
    bboxes.insert(bboxes.end(), vecBox_after_nms.begin(), vecBox_after_nms.end());
    std::vector<VecObjBBox>().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    return RET_CODE::SUCCESS;     
}

RET_CODE YoloDetectionV4::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    LOGI << "-> get_class_type: inner_class_num = " << _output_cls_num;
    if(_output_cls_num<=0) return RET_CODE::ERR_MODEL_NOT_INIT;
    if(m_output_cls_order.empty()) return RET_CODE::ERR_MODEL_NOT_INIT;
    for(auto &&uni_clss:_unique_cls_order){
        valid_clss.push_back(uni_clss.first);
    }
    return RET_CODE::SUCCESS;
}

static inline int get_unique_cls_num(std::vector<CLS_TYPE>& output_clss, std::map<CLS_TYPE,int> &unique_cls_order ){
    unique_cls_order.clear();
    std::set<CLS_TYPE> unique_cls;
    unique_cls.insert(output_clss.begin(), output_clss.end());
    int i = 0;
    for(auto &&unicls: unique_cls){
        unique_cls_order.insert(std::pair<CLS_TYPE,int>(unicls,i++));
    }
    return unique_cls.size();
}
RET_CODE YoloDetectionV4::set_output_cls_order(std::vector<CLS_TYPE>& output_clss){
    if(_output_cls_num == 0)
        _output_cls_num = output_clss.size();
    else{
        if(_output_cls_num!=output_clss.size()){
            printf("_output_cls_num [%d] != input class size [%d]\n", _output_cls_num, output_clss.size());
            return RET_CODE::FAILED;
        }
    }
    m_output_cls_order = output_clss;
    _unique_cls_num = get_unique_cls_num(output_clss, _unique_cls_order);
    return RET_CODE::SUCCESS;
}


static inline int get_unique_cls_num(std::vector<std::string>& output_clss, std::map<std::string,int> &unique_cls_order ){
    unique_cls_order.clear();
    std::set<std::string> unique_cls;
    unique_cls.insert(output_clss.begin(), output_clss.end());
    int i = 0;
    for(auto &&unicls: unique_cls){
        unique_cls_order.insert(std::pair<std::string,int>(unicls,i++));
    }
    return unique_cls.size();
}
RET_CODE YoloDetectionV4::set_output_cls_order(std::vector<std::string> &output_clss){
    if(_output_cls_num == 0)
        _output_cls_num = output_clss.size();
    else{
        if(_output_cls_num!=output_clss.size()){
            printf("_output_cls_num [%d] != input class size [%d]\n", _output_cls_num, output_clss.size());
            return RET_CODE::FAILED;
        }
    }
    m_output_cls_order_str = output_clss;
    _unique_cls_num = get_unique_cls_num(output_clss, _unique_cls_order_str);
    return RET_CODE::SUCCESS;
}

/*******************************************************************************
YoloDetectionV4 + DeepSort
chaffee.chen@2022-09-30
*******************************************************************************/

/*******************************************************************************
YoloDetectionV4 + ByteTrack
use set_trackor to switch differenct version of ByteTrack
chaffee.chen@2022-09-30
*******************************************************************************/

