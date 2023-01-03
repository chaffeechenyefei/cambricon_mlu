#include "module_yoloface_detection.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include <fstream>
#include <iostream>

/**
 * jsoncpp https://github.com/open-source-parsers/jsoncpp/tree/jsoncpp_version
 * tag: 1.9.5
*/
#include "json/json.h"
#include "json_encoder/json_encoder.hpp"

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

#define NMS_UNION 0
#define NMS_MIN 1

#define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

// #include <future>
using namespace ucloud;
using namespace cv;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// YoloFace
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
RET_CODE YoloFaceDetection::init(const std::string &modelpath){
    LOGI << "-> YoloFaceDetection::init, modelpath: " << modelpath;
    bool pad_both_side = false;
    bool keep_aspect_ratio = true;
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    if (ret!=RET_CODE::SUCCESS) return ret;

    if(_output_cls_num == 0)
        _output_cls_num = m_net->m_outputShape[0].H() - m_dimOffset; //xywh+objectness+5x(xy) = 15
    else{
        if(_output_cls_num!=m_net->m_outputShape[0].H() - m_dimOffset){
            printf("_output_cls_num [%d] != models output shape [%d]\n", _output_cls_num, m_net->m_outputShape[0].H() - 5);
            ret = RET_CODE::FAILED;
        }
    }
    return ret;
}

RET_CODE YoloFaceDetection::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> YoloFaceDetection::init";
    LOGI << "use_auto_model: " << use_auto_model;
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }

    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end())
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    bool pad_both_side = false;
    bool keep_aspect_ratio = true;
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init( modelpath[InitParam::BASE_MODEL], config);
    if (ret!=RET_CODE::SUCCESS) {
        printf("ERR:: YoloFaceDetection::init(), m_net->base_init return [%d]\n", ret);
        return ret;
    }
    if(_output_cls_num == 0)
        _output_cls_num = m_net->m_outputShape[0].H() - m_dimOffset; //xywh+objectness+5x(xy) = 15
    else{
        if(_output_cls_num!=m_net->m_outputShape[0].H() - m_dimOffset){
            printf("_output_cls_num [%d] != models output shape [%d]\n", _output_cls_num, m_net->m_outputShape[0].H() - 5);
            ret = RET_CODE::FAILED;
        }
    }

    // if(modelpath.find(InitParam::SUB_MODEL)!=modelpath.end()){
    //     LicplateRecognition* _ptr_ = new LicplateRecognition();
    //     ret = _ptr_->init(modelpath[InitParam::SUB_MODEL]);
    //     if(ret!=RET_CODE::SUCCESS){
    //         printf("ERR::licplate recognition init return [%d]\n",ret);
    //         return ret;
    //     }
    //     m_licplateRecognizer.reset(_ptr_);
    // }

    LOGI << "<- YoloFaceDetection::init";
    return ret;
}


YoloFaceDetection::~YoloFaceDetection(){
    LOGI << "~YoloFaceDetection()";
}

RET_CODE YoloFaceDetection::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> YoloFaceDetection::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    float aspect_ratio = 1.0;
    float aX,aY;
    float** model_output = nullptr;
    {
        ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage,TvaiRect{0,0,tvimage.width, tvimage.height}, aspect_ratio, aX, aY);
        if(ret!=RET_CODE::SUCCESS) return ret;
        model_output = m_net->general_mlu_infer();
    }

    ret = postprocess(model_output[0], bboxes, threshold, nms_threshold ,1.0, aspect_ratio, tvimage.width, tvimage.height);
    m_net->cpu_free(model_output);
    if(ret!=RET_CODE::SUCCESS) return ret;
    // if(m_licplateRecognizer){
    //     ret = m_licplateRecognizer->run(tvimage, bboxes);
    //     if(ret!=RET_CODE::SUCCESS) {
    //         LOGI << "m_licplateRecognizer err...";
    //         return ret;
    //     }
    // }
    return ret;
}

RET_CODE YoloFaceDetection::run_yuv_on_mlu(TvaiImage &tvimage, TvaiRect tvrect ,VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> YoloFaceDetection::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    float aspect_ratio = 1.0;
    float aX,aY;
    float** model_output = nullptr;
    {
        ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, tvrect ,aspect_ratio, aX, aY);
        if(ret!=RET_CODE::SUCCESS) return ret;
        model_output = m_net->general_mlu_infer();
    }

    ret = postprocess(model_output[0], bboxes, tvrect, threshold, nms_threshold ,1.0, aspect_ratio, tvimage.width, tvimage.height);
    m_net->cpu_free(model_output);
    if(ret!=RET_CODE::SUCCESS) return ret;
    return ret;
}

RET_CODE YoloFaceDetection::run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> YoloFaceDetection::run_bgr_on_cpu";
    RET_CODE ret = RET_CODE::FAILED;
    float aspect_ratio = 1.0;
    float aX,aY;
    float** model_output = nullptr;
    {
        ret = m_net->general_preprocess_bgr_on_cpu(tvimage, aspect_ratio, aX , aY);
        if(ret!=RET_CODE::SUCCESS) return ret;
        model_output = m_net->general_mlu_infer();
    }
    ret = postprocess(model_output[0], bboxes, threshold, nms_threshold, 1.0, aspect_ratio, tvimage.width, tvimage.height);
    m_net->cpu_free(model_output);
    return ret;
}


RET_CODE YoloFaceDetection::run(TvaiImage &tvimage, VecObjBBox &bboxes,  float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);
    if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
        ret = run_bgr_on_cpu(tvimage, bboxes, threshold, nms_threshold);
    }
    else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu(tvimage,bboxes,threshold, nms_threshold);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;    

    return ret;
}

RET_CODE YoloFaceDetection::run(TvaiImage &tvimage, TvaiRect tvrect, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu(tvimage, tvrect, bboxes, threshold, nms_threshold);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

static inline int get_unique_cls_num(std::vector<CLS_TYPE>& output_clss, std::map<CLS_TYPE,int> &unique_cls_order ){
    unique_cls_order.clear();
    std::vector<CLS_TYPE> unique_cls;
    for(auto i=output_clss.begin(); i !=output_clss.end(); i++){
        bool conflict = false;
        for(auto iter=unique_cls.begin(); iter!=unique_cls.end(); iter++){
            if( *i == *iter ){
                conflict = true;
                break;
            }
        }
        if(!conflict) unique_cls.push_back(*i);
    }
    for(int i=0; i < unique_cls.size(); i++ ){
        unique_cls_order.insert(std::pair<CLS_TYPE,int>(unique_cls[i],i));
    }
    return unique_cls.size();
}
RET_CODE YoloFaceDetection::set_output_cls_order(std::vector<CLS_TYPE>& output_clss){
    if(_output_cls_num == 0)
        _output_cls_num = output_clss.size();
    else{
        if(_output_cls_num!=output_clss.size()){
            printf("_output_cls_num [%d] != input class size [%d]\n", _output_cls_num, output_clss.size());
            return RET_CODE::FAILED;
        }
    }
    m_output_cls_order = output_clss;
    _unique_cls_num = get_unique_cls_num(output_clss, m_unique_cls_order);
    return RET_CODE::SUCCESS;
}

RET_CODE YoloFaceDetection::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    LOGI << "-> get_class_type: inner_class_num = " << _output_cls_num;
    if(_output_cls_num<=0) return RET_CODE::ERR_MODEL_NOT_INIT;
    if(m_output_cls_order.empty()) return RET_CODE::ERR_MODEL_NOT_INIT;
    for(auto &&uni_clss:m_unique_cls_order){
        valid_clss.push_back(uni_clss.first);
    }
    return RET_CODE::SUCCESS;
}


RET_CODE YoloFaceDetection::postprocess(float* model_output, VecObjBBox &bboxes, float threshold, float nms_threshold,
    float expand_ratio, float aspect_ratio, int imgW, int imgH)
{
    LOGI << "-> YoloFaceDetection::postprocess";
    int nBBox = m_net->m_outputShape[0].C();
    int featLen = m_net->m_outputShape[0].H();
    std::vector<VecObjBBox> vecBox;
    VecObjBBox vecBox_after_nms;
    base_output2ObjBox_multiCls_yoloface(model_output, vecBox, m_output_cls_order, m_unique_cls_order , nBBox, featLen, threshold, m_dimOffset);
    LOGI << "vecBox.size(): " << vecBox[0].size() << ", " << vecBox[1].size() << ", " << vecBox[2].size() << ", " << vecBox[3].size();
    base_nmsBBox(vecBox, nms_threshold, NMS_MIN ,vecBox_after_nms );
    LOGI << "vecBox_after_nms.size(): " << vecBox_after_nms.size();

    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aspect_ratio);
    bboxes.insert(bboxes.end(), vecBox_after_nms.begin(), vecBox_after_nms.end());
    LOGI << "after filter " << bboxes.size() << std::endl;

    std::vector<VecObjBBox>().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    // free(cpu_chw);
    return RET_CODE::SUCCESS;     
}

static void shift_box_from_roi_to_org(VecObjBBox &bboxes, TvaiRect &roirect){
    for(auto &&bbox: bboxes){
        bbox.rect.x += roirect.x;
        bbox.rect.y += roirect.y;
    }
}
RET_CODE YoloFaceDetection::postprocess(float* model_output, VecObjBBox &bboxes, TvaiRect tvrect, float threshold, float nms_threshold,
    float expand_ratio, float aspect_ratio, int imgW, int imgH)
{
    LOGI << "-> YoloFaceDetection::postprocess";
    int nBBox = m_net->m_outputShape[0].C();
    int featLen = m_net->m_outputShape[0].H();
    std::vector<VecObjBBox> vecBox;
    VecObjBBox vecBox_after_nms;
    base_output2ObjBox_multiCls_yoloface(model_output, vecBox, m_output_cls_order, m_unique_cls_order , nBBox, featLen, threshold, m_dimOffset);
    base_nmsBBox(vecBox, nms_threshold, NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aspect_ratio);
    shift_box_from_roi_to_org(vecBox_after_nms, tvrect);

    bboxes.insert(bboxes.end(), vecBox_after_nms.begin(), vecBox_after_nms.end());

    LOGI << "after filter " << bboxes.size() << std::endl;
    std::vector<VecObjBBox>().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    // free(cpu_chw);
    return RET_CODE::SUCCESS;     
}

float YoloFaceDetection::clip_threshold(float x){
    if(x < 0) return m_default_threshold;
    if(x > 1) return m_default_threshold;
    return x;
}
float YoloFaceDetection::clip_nms_threshold(float x){
    if(x < 0) return m_default_nms_threshold;
    if(x > 1) return m_default_nms_threshold;
    return x;
}





////////////////////////////////////////////////////////////////////////////////////////////////////////
//车牌字符识别
////////////////////////////////////////////////////////////////////////////////////////////////////////
/*******************************************************************************
LicplateRecognition 车牌识别, 取消BaseModel的继承
shawn.qian@2022-09-20
chaffee.chen@2022-10-09
*******************************************************************************/

const std::vector<std::string> LicplateRecognition::LICPLATE_CHARS = {"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                        "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                        "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
                        "新",
                        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                        "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                        "W", "X", "Y", "Z", "I", "O", "-"};


RET_CODE LicplateRecognition::init(const std::string &modelpath){
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    //Self param
    return ret;    
}

RET_CODE LicplateRecognition::init(std::map<InitParam, std::string> &modelpath){
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }
    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end()) {
        printf("ERR::LicplateRecognition::init, BASE_MODEL is not given\n");
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath[InitParam::BASE_MODEL], config);
    //Self param
    return ret;
}

//clear self param
LicplateRecognition::~LicplateRecognition(){LOGI << "-> LicplateRecognition::~LicplateRecognition";}

RET_CODE LicplateRecognition::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(_cls_);
    return RET_CODE::SUCCESS;
};


RET_CODE LicplateRecognition::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
        ret = run_bgr_on_cpu(tvimage, bboxes);
    }
    else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}


static string decode_license(float *output){
    int maxid = -1;
    int pos;
    float max_val;
    vector<int> preb_label, no_repeat_blank_label;
    for (int i = 0; i < 18; i++){
        pos = 0;
        max_val = output[i*68];
        for (int j = 0; j < 68; j++){
            if (output[i*68 + j] > max_val){
                pos = j;
                max_val = output[i*68 + j];
            }
        }
        preb_label.push_back(pos);
    }

    int pre_c = preb_label[0];
    if (pre_c != 67){
        no_repeat_blank_label.push_back(pre_c);
    }
    int c;
    for (int i = 0; i < preb_label.size(); i++){
        c = preb_label[i];
        if (pre_c == c || c == 67){
            if (c == 67){
                pre_c = c;
            }
            continue;
        }
        no_repeat_blank_label.push_back(c);
        pre_c = c;
    }
    
    string licplate_str = "";
    for (int i = 0; i < no_repeat_blank_label.size(); i++){
        licplate_str +=  LicplateRecognition::LICPLATE_CHARS[no_repeat_blank_label[i]];
    }
    return licplate_str;
}

RET_CODE LicplateRecognition::postprocess(float* model_output, BBox &bbox){
    int featDim = m_net->m_outputShape[0].DataCount();
    LOGI << "featDim: " << featDim;
    if(featDim != 68*18 ) return RET_CODE::ERR_MODEL_NOT_MATCH;
    string license_str = decode_license(model_output);
    // std::cout << "licplate_str: " << license_str << std::endl;

    UcloudJsonEncoder jsonWriter;
    jsonWriter.initial_context_with_string(bbox.desc);
    jsonWriter.add_context(tagJSON_ROOT::VEHICLE, tagJSON_ATTR::LICPLATE , license_str);
    std::string json_file = jsonWriter.output_to_string();
    bbox.desc = json_file;

    LOGI << "====JSON====";
    LOGI << json_file;
    //TODO
    return RET_CODE::SUCCESS;
}

RET_CODE LicplateRecognition::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes){
    LOGI << "-> LicplateRecognition::run_yuv_on_mlu_phyAddr";
    RET_CODE ret = RET_CODE::FAILED;
    LOGI << "bboxes.size(): " << bboxes.size();
    for(int i = 0; i < bboxes.size(); i++){
        TvaiRect roiRect = bboxes[i].rect;
        roiRect = globalscaleTvaiRect(roiRect, _expand_ratio, tvimage.width, tvimage.height);
        float aspect_ratio = 1.0;
        float aX, aY;
        float** model_output = nullptr;
        {
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roiRect, aspect_ratio, aX, aY);
            if(ret!=RET_CODE::SUCCESS) {
                printf("[%s][%d]ERR::LicplateRecognition::run_yuv_on_mlu_phyAddr m_net return [%d]\n", __FILE__, __LINE__, ret);
                return ret;
            }
            model_output = m_net->general_mlu_infer();
        }
        //TODO post process 
        ret = postprocess(model_output[0],bboxes[i]);
        if(ret!=RET_CODE::SUCCESS) return ret;
        m_net->cpu_free(model_output);
    }
    return RET_CODE::SUCCESS;
}


RET_CODE LicplateRecognition::run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes){
    /**
     * ATT. 存在问题, 没有进行 rect expand!!!!
     */
    LOGI << "-> LicplateRecognition::run_bgr_on_cpu";
    RET_CODE ret = RET_CODE::FAILED;
    std::vector<CLS_TYPE> valid_cls = {CLS_TYPE::LICPLATE_YELLOW,CLS_TYPE::LICPLATE_SGREEN, CLS_TYPE::LICPLATE_BLUE, CLS_TYPE::LICPLATE_BGREEN,CLS_TYPE::LICPLATE};
    std::vector<float*> model_outputs;
    std::vector<float> aspect_ratios;
    VecObjBBox bboxes_tmp = bboxes;
    for(auto &&box: bboxes_tmp){
        box.rect = globalscaleTvaiRect(box.rect, _expand_ratio, tvimage.width, tvimage.height);
    }
    {
        ret = m_net->general_preprocess_infer_bgr_on_cpu(tvimage, bboxes_tmp,model_outputs, aspect_ratios, valid_cls);
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    //TODO post processing
    for(int i=0; i< model_outputs.size(); i++){
        if(model_outputs[i]==nullptr) continue;
        ret = postprocess(model_outputs[i], bboxes[i]);
        free(model_outputs[i]);
        // if(ret!=RET_CODE::SUCCESS)默认后处理没有问题, 否则存在内存泄漏问题
        //     break;        
    }
    model_outputs.clear();
    return ret;
}
