#include "module_binary_classification.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
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

// #include <future>
using namespace ucloud;
using namespace cv;
using std::vector;

/*******************************************************************************
inner function
*******************************************************************************/
static TvaiRect scaleRect(TvaiRect &rect, float scale){
    TvaiRect output;
    float cx = rect.x + rect.width/2;
    float cy = rect.y + rect.height/2;
    output.width = rect.width*scale;
    output.height = rect.height*scale;
    output.x = cx - output.width/2;
    output.y = cy - output.height/2;
    return output;
}

static bool is_bbox_valid(BBox &bbox, vector<CLS_TYPE> &valid_clss ){
    if(valid_clss.empty()) return true;//空则返回有效
    for(auto iter=valid_clss.begin(); iter!=valid_clss.end(); iter++){
        if(bbox.objtype == *iter) return true;
    }
    return false;
}


/*******************************************************************************
ClassificationV4 use BaseModelV2
chaffee.chen@2022-10-17
 * ClassificationV4
 * 解决什么问题: 输入图像, 以及候选区域, 返回候选区域属于某个类别的概率, 同时修改候选框
 * 注意:
 *  FUNC: set_expand_ratio, 对候选框进行扩大, 扩大后的结果输入到模型中进行分类
 *  set_output_cls_order 设定模型输出维度对应的类别 OTHERS表示占位符
*******************************************************************************/
RET_CODE ClassificationV4::init(const std::string &modelpath){
    LOGI << "-> ClassificationV4::init";
    bool pad_both_side = true;//双边留黑
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    //Self param
    return ret;
}

RET_CODE ClassificationV4::init(std::map<InitParam, std::string> &modelpath){
    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    return init(modelpath[InitParam::BASE_MODEL]);
}

/*******************************************************************************
 * get_class_type 返回剔除占位类型OTHERS后的有效分类类别
*******************************************************************************/  
ucloud::RET_CODE ClassificationV4::get_class_type(std::vector<ucloud::CLS_TYPE> &valid_clss){
    LOGI << "-> get_class_type: inner_class_num = " << m_clss.size();
    if(m_clss.empty()) return ucloud::RET_CODE::ERR_MODEL_NOT_INIT;
    std::set<CLS_TYPE> unique_vec;
    unique_vec.insert(m_clss.begin(), m_clss.end());
    unique_vec.erase(CLS_TYPE::OTHERS);
    valid_clss.insert(valid_clss.end(), unique_vec.begin(), unique_vec.end());
    return ucloud::RET_CODE::SUCCESS;
}

/*******************************************************************************
 * set_output_cls_order
 * 例如 {OTHERS, FIRE, OTHERS} 长度必须与模型输出一致 否则后续运行会FAILED
 * 表示输出的 dim0 = OTHERS, dim1 = FIRE, dim2 = OTHERS
 * OTHERS仅表示占位, 仅dim1会被输出
 *******************************************************************************/ 
ucloud::RET_CODE ClassificationV4::set_output_cls_order(std::vector<ucloud::CLS_TYPE> &output_clss){
    m_clss = output_clss;
    return RET_CODE::SUCCESS;
}
ucloud::RET_CODE ClassificationV4::set_output_cls_order(std::vector<std::string> &output_clss){
    m_clss_str = output_clss;
    return RET_CODE::SUCCESS;
}

RET_CODE ClassificationV4::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
        ret = run_bgr_on_cpu(tvimage, bboxes, threshold);
    }
    else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu(tvimage,bboxes, threshold);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE ClassificationV4::run_yuv_on_mlu(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold){
    LOGI << "-> ClassificationV4::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    for(int i = 0; i < bboxes.size(); i++){
        //根据设定的类型过滤bbox, 即仅对某些类别的bbox进一步进行分类
        TvaiRect roiRect = bboxes[i].rect;
        //EXPAND
        TvaiRect roiRectXL = scaleRect(roiRect, m_expand_ratio);
        float aspect_ratio = 1.0;
        float aX,aY;
        float** model_output = nullptr;
        {
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roiRectXL, aspect_ratio, aX , aY);
            if(ret!=RET_CODE::SUCCESS) return ret;
            model_output = m_net->general_mlu_infer();
        }
        //TODO post process 
        ret = postprocess(model_output[0], bboxes[i], threshold);
        if(ret!=RET_CODE::SUCCESS){ m_net->cpu_free(model_output); return ret; }
        m_net->cpu_free(model_output);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE ClassificationV4::run_bgr_on_cpu(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold){
    LOGI << "-> ClassificationV4::run_bgr_on_cpu";
    RET_CODE ret = RET_CODE::FAILED;
    std::vector<float*> model_outputs;
    std::vector<float> aspect_ratios;
    //临时BBox, 用于存放扩大后的bbox
    vector<BBox> bboxesXL;
    for(int i = 0; i < bboxes.size(); i++ ){
        BBox bbox;
        bbox.rect = scaleRect(bboxes[i].rect, m_expand_ratio);
        bbox.objtype = bboxes[i].objtype;
        bboxesXL.push_back(bbox);
    }
    {
        std::vector<CLS_TYPE> valid_clss;
        ret = m_net->general_preprocess_infer_bgr_on_cpu(tvimage, bboxesXL ,model_outputs, aspect_ratios, valid_clss);
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    //TODO post processing
    for(int i=0; i< model_outputs.size(); i++){
        if(model_outputs[i]==nullptr) continue;
        ret = postprocess(model_outputs[i], bboxes[i], threshold);
        if(ret!=RET_CODE::SUCCESS)
            break;        
    }
    for(int i=0; i< model_outputs.size(); i++){
        if(model_outputs[i]!=nullptr)
            free(model_outputs[i]);
    }
    model_outputs.clear();
    return ret;
}

RET_CODE ClassificationV4::postprocess(float *model_output, BBox &bbox, float threshold){
    if(m_clss.size() != m_net->m_outputShape[0].DataCount()){
        printf("**[%s][%d] m_clss.size()[%d] != m_net->m_outputShape[0].DataCount()[%d]\n", __FILE__, __LINE__, m_clss.size(), m_net->m_outputShape[0].DataCount() );
        return RET_CODE::FAILED;
    }
    float max_score = -1;
    CLS_TYPE max_score_type = CLS_TYPE::OTHERS;
    std::string max_score_type_str = "";
    float *ptr =model_output;
    if(!m_clss.empty()){
        for(int i = 0; i < m_clss.size(); i++){
            if(m_clss[i]==CLS_TYPE::OTHERS) continue;
            if(ptr[i] > max_score){
                max_score = ptr[i];
                max_score_type = m_clss[i];
            }
        }
        if(max_score > threshold){
            bbox.objtype = max_score_type;
            bbox.confidence = max_score;
            bbox.objectness = max_score;
        }
    } else if(!m_clss_str.empty()){
        for(int i = 0; i < m_clss_str.size(); i++){
            if(m_clss_str[i]=="") continue;
            if(ptr[i] > max_score){
                max_score = ptr[i];
                max_score_type_str = m_clss_str[i];
            }
        }
        if(max_score > threshold){
            bbox.objtype = CLS_TYPE::TARGET;
            bbox.objname = max_score_type_str;
            bbox.confidence = max_score;
            bbox.objectness = max_score;
        }        
    } else {
        printf("**ERR[%s][%d]: The meaning of model output's is not defined.\n", __FILE__, __LINE__);
        return RET_CODE::FAILED;
    }

    LOGI << "max_score: " << max_score << ", cls_type: " << max_score_type << "(" << max_score_type_str << ")";

    return RET_CODE::SUCCESS;
}





/*******************************************************************************
BinaryClassificationV4 use BaseModelV2
chaffee.chen@2022-10-08
*******************************************************************************/
RET_CODE BinaryClassificationV4::init(const std::string &modelpath){
    LOGI << "-> BinaryClassificationV4::init";
    bool pad_both_side = true;//双边留黑
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    //Self param
    return ret;
}

RET_CODE BinaryClassificationV4::init(std::map<InitParam, std::string> &modelpath){
    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    return init(modelpath[InitParam::BASE_MODEL]);
}

//clear self param
BinaryClassificationV4::~BinaryClassificationV4(){LOGI << "-> BinaryClassificationV4::~BinaryClassificationV4";}

RET_CODE BinaryClassificationV4::run_yuv_on_mlu(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold){
    LOGI << "-> BinaryClassificationV4::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    for(int i = 0; i < bboxes.size(); i++){
        //根据设定的类型过滤bbox, 即仅对某些类别的bbox进一步进行分类
        if( !is_bbox_valid(bboxes[i] ,m_in_valid_cls)  ) continue;
        TvaiRect roiRect = bboxes[i].rect;
        //EXPAND
        TvaiRect roiRectXL = scaleRect(roiRect, m_expand_ratio);
        float aspect_ratio = 1.0;
        float aX,aY;
        float** model_output = nullptr;
        {
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roiRectXL, aspect_ratio, aX , aY);
            if(ret!=RET_CODE::SUCCESS) return ret;
            model_output = m_net->general_mlu_infer();
        }
        //TODO post process 
        ret = postprocess(model_output[0], bboxes[i], threshold);
        if(ret!=RET_CODE::SUCCESS){ m_net->cpu_free(model_output); return ret; }
        m_net->cpu_free(model_output);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE BinaryClassificationV4::run_bgr_on_cpu(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold){
    LOGI << "-> BinaryClassificationV4::run_bgr_on_cpu";
    RET_CODE ret = RET_CODE::FAILED;
    std::vector<float*> model_outputs;
    std::vector<float> aspect_ratios;
    //临时BBox, 用于存放扩大后的bbox
    vector<BBox> bboxesXL;
    for(int i = 0; i < bboxes.size(); i++ ){
        BBox bbox;
        bbox.rect = scaleRect(bboxes[i].rect, m_expand_ratio);
        bbox.objtype = bboxes[i].objtype;
        bboxesXL.push_back(bbox);
    }
    {
        ret = m_net->general_preprocess_infer_bgr_on_cpu(tvimage, bboxesXL ,model_outputs, aspect_ratios, m_in_valid_cls);
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    //TODO post processing
    for(int i=0; i< model_outputs.size(); i++){
        if(model_outputs[i]==nullptr) continue;
        ret = postprocess(model_outputs[i], bboxes[i], threshold);
        if(ret!=RET_CODE::SUCCESS)
            break;        
    }
    for(int i=0; i< model_outputs.size(); i++){
        if(model_outputs[i]!=nullptr)
            free(model_outputs[i]);
    }
    model_outputs.clear();
    return ret;
}

RET_CODE BinaryClassificationV4::postprocess(float *model_output, BBox &bbox, float threshold){
    float score = model_output[m_primary_rank];
    bbox.confidence = score;
    bbox.objectness = score;
    if(score > threshold )
        bbox.objtype = m_cls;
    else
        bbox.objtype = CLS_TYPE::UNKNOWN;
    return RET_CODE::SUCCESS;
}

RET_CODE BinaryClassificationV4::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
        ret = run_bgr_on_cpu(tvimage, bboxes, threshold);
    }
    else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu(tvimage,bboxes, threshold);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE BinaryClassificationV4::get_class_type(std::vector<CLS_TYPE> &clss){
    clss.push_back(m_cls);
    return RET_CODE::SUCCESS;
}

RET_CODE BinaryClassificationV4::set_filter_cls(std::vector<CLS_TYPE> &cls_seqs){
    m_in_valid_cls = cls_seqs;
    return RET_CODE::SUCCESS;
}

RET_CODE BinaryClassificationV4::set_primary_output_cls(int rank, CLS_TYPE cls){
    if (rank > 1) rank = 1;
    if (rank < 0) rank = 0;
    m_primary_rank = rank;
    m_cls = cls;
    return RET_CODE::SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// Begin of SmokingClassification
////////////////////////////////////////////////////////////////////////////////////////////////
RET_CODE SmokingClassification::init(const std::string &modelpath){
    LOGI << "-> SmokingClassification::init";
    bool pad_both_side = true;//双边留黑
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    // if (ret!=RET_CODE::SUCCESS) return ret;
    //Self param
    return ret;
}

RET_CODE SmokingClassification::init(std::map<InitParam, std::string> &modelpath){
    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    return init(modelpath[InitParam::BASE_MODEL]);
}

//clear self param
SmokingClassification::~SmokingClassification(){LOGI << "-> SmokingClassification::~SmokingClassification";}

/**
 * Smoking
 */
RET_CODE SmokingClassification::run(TvaiImage &tvimage, VecSmokingBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes, threshold);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE SmokingClassification::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecSmokingBox &bboxes, float threshold){
    LOGI << "-> SmokingClassification::run_yuv_on_mlu_phyAddr";
    RET_CODE ret = RET_CODE::FAILED;

    for(int i = 0; i < bboxes.size(); i++){//存在问题，默认所有bbox都存在
        VecObjBBox batch_bbox;
        batch_bbox.push_back(bboxes[i].face);
        batch_bbox.push_back(bboxes[i].handl);
        batch_bbox.push_back(bboxes[i].handr);
        float** model_output = nullptr;
        vector<float> batch_aspect_ratio;
        {
            float aspect_ratio, aX, aY;
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, bboxes[i].face.rect,aspect_ratio, aX, aY );
            model_output = m_net->general_mlu_infer();
            // if(_N==1 && _MI > 1)//MIMO
            //     ret = BaseModel::general_preprocess_yuv_on_mlu(tvimage, batch_bbox, batch_aspect_ratio, 0);
            // else if(_N==1 && _MI == 1){ //SISO--FACE input only
            //     float aspect_ratio, aX, aY;
            //     ret = BaseModel::general_preprocess_yuv_on_mlu_union(tvimage, bboxes[i].face.rect,aspect_ratio, aX, aY );
            // } else//Batch
            //     ret = BaseModel::general_batch_preprocess_yuv_on_mlu(tvimage, batch_bbox, batch_aspect_ratio, 0);
        }
        //TODO postprocess
        postprocess(model_output[0], bboxes[i], threshold);
        m_net->cpu_free(model_output);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE SmokingClassification::postprocess(float *model_output, SMOKING_BOX &bbox, float threshold ){
    float score = model_output[m_primary_rank];
    bbox.body.confidence = score;
    bbox.body.objectness = score;
    if(score > threshold )
        bbox.body.objtype = m_cls;
    else
        bbox.body.objtype = CLS_TYPE::UNKNOWN;
    return RET_CODE::SUCCESS;
}

RET_CODE SmokingClassification::get_class_type(std::vector<CLS_TYPE> &clss){
    clss.push_back(m_cls);
    return RET_CODE::SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////
// End of SmokingClassification
////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////
// Begin of PhoningClassification
////////////////////////////////////////////////////////////////////////////////////////////////
RET_CODE PhoningClassification::init(const std::string &modelpath){
    LOGI << "-> PhoningClassification::init";
    bool pad_both_side = true;//双边留黑
    bool keep_aspect_ratio = false;//不保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    // if (ret!=RET_CODE::SUCCESS) return ret;
    //Self param
    return ret;
}

RET_CODE PhoningClassification::init(std::map<InitParam, std::string> &modelpath){
    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    return init(modelpath[InitParam::BASE_MODEL]);
}

//clear self param
PhoningClassification::~PhoningClassification(){LOGI << "-> PhoningClassification::~PhoningClassification";}

RET_CODE PhoningClassification::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes, threshold);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE PhoningClassification::run(TvaiImage &tvimage, VecPedBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes, threshold);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE PhoningClassification::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold){
    LOGI << "-> PhoningClassification::run_yuv_on_mlu_phyAddr";
    RET_CODE ret = RET_CODE::FAILED;
    for(int i = 0; i < bboxes.size(); i++){
        if(bboxes[i].objtype !=CLS_TYPE::PEDESTRIAN) continue;
        TvaiRect roiRect = bboxes[i].rect;
        float aspect_ratio = 1.0;
        float aX,aY;
        float** model_output = nullptr;
        {
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roiRect, aspect_ratio, aX , aY);
            if(ret!=RET_CODE::SUCCESS) return ret;
            model_output = m_net->general_mlu_infer();
        }
        //TODO post process 
        ret = postprocess(model_output[0],bboxes[i], threshold);
        m_net->cpu_free(model_output);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE PhoningClassification::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecPedBox &bboxes, float threshold){
    LOGI << "-> PhoningClassification::run_yuv_on_mlu_phyAddr";
    RET_CODE ret = RET_CODE::FAILED;
    for(int i = 0; i < bboxes.size(); i++){
        TvaiRect roiRect = bboxes[i].target.rect;
        float aspect_ratio = 1.0;
        float aX,aY;
        float** model_output = nullptr;
        {
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roiRect, aspect_ratio, aX , aY);
            if(ret!=RET_CODE::SUCCESS) return ret;
            model_output = m_net->general_mlu_infer();
        }
        //TODO post process 
        ret = postprocess(model_output[0],bboxes[i], threshold);
        m_net->cpu_free(model_output);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE PhoningClassification::postprocess(float *model_output, BBox &bbox, float threshold){
    float score = std::max( model_output[1], model_output[2] );
    bbox.confidence = score;
    bbox.objectness = score;
    if(score > threshold )
        bbox.objtype = m_cls;
    else
        bbox.objtype = CLS_TYPE::UNKNOWN;
    return RET_CODE::SUCCESS;
}

RET_CODE PhoningClassification::postprocess(float *model_output, PED_BOX &bbox, float threshold){
    float score = std::max( model_output[1], model_output[2] );
    // std::cout << model_output[0] << ", " << model_output[1] <<", " << model_output[2] << std::endl;
    bbox.body.confidence = score;
    bbox.body.objectness = score;
    if(score > threshold )
        bbox.body.objtype = m_cls;
    else
        bbox.body.objtype = CLS_TYPE::UNKNOWN;
    return RET_CODE::SUCCESS;
}

RET_CODE PhoningClassification::get_class_type(std::vector<CLS_TYPE> &clss){
    clss.push_back(m_cls);
    return RET_CODE::SUCCESS;
}
////////////////////////////////////////////////////////////////////////////////////////////////
// End of PhoningClassification
////////////////////////////////////////////////////////////////////////////////////////////////




// ////////////////////////////////////////////////////////////////////////////////////////////////
// // Begin of BinaryClassification
// ////////////////////////////////////////////////////////////////////////////////////////////////

// RET_CODE BinaryClassification::init(const std::string &modelpath){
//     LOGI << "-> BinaryClassification::init";
//     bool pad_both_side = true;//双边留黑
//     bool keep_aspect_ratio = true;//保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);
//     // if (ret!=RET_CODE::SUCCESS) return ret;
//     //Self param
//     return ret;
// }

// RET_CODE BinaryClassification::init(std::map<InitParam, std::string> &modelpath){
//     if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
//     return init(modelpath[InitParam::BASE_MODEL]);
// }

// //clear self param
// BinaryClassification::~BinaryClassification(){LOGI << "-> BinaryClassification::~BinaryClassification";}


// RET_CODE BinaryClassification::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> BinaryClassification::run_yuv_on_mlu_phyAddr";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         if( !is_bbox_valid(bboxes[i] ,m_in_valid_cls)  ) continue;
//         TvaiRect roiRect = bboxes[i].rect;
//         //EXPAND
//         TvaiRect roiRectXL = scaleRect(roiRect, m_expand_ratio);
//         float aspect_ratio = 1.0;
//         float aX, aY;
//         float* model_output = nullptr;
//         {
//             std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRectXL, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             model_output = BaseModel::general_mlu_infer();
//         }
        
//         //TODO post process 
//         ret = postprocess(model_output, bboxes[i]);
//         if(ret!=RET_CODE::SUCCESS){ free(model_output); return ret; }
//         free(model_output);
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE BinaryClassification::run_yuv_on_mlu(TvaiImage& tvimage, VecObjBBox &bboxes){
//     LOGI << "-> BinaryClassification::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         //根据设定的类型过滤bbox, 即仅对某些类别的bbox进一步进行分类
//         if( !is_bbox_valid(bboxes[i] ,m_in_valid_cls)  ) continue;
//         TvaiRect roiRect = bboxes[i].rect;
//         //EXPAND
//         TvaiRect roiRectXL = scaleRect(roiRect, m_expand_ratio);
//         float aspect_ratio = 1.0;
//         float aX,aY;
//         float* model_output = nullptr;
//         {
//             std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu(tvimage, roiRectXL, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             model_output = BaseModel::general_mlu_infer();
//         }
//         //TODO post process 
//         ret = postprocess(model_output, bboxes[i]);
//         if(ret!=RET_CODE::SUCCESS){ free(model_output); return ret; }
//         free(model_output);
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE BinaryClassification::run_bgr_on_cpu(TvaiImage& tvimage, VecObjBBox &bboxes){
//     LOGI << "-> BinaryClassification::run_bgr_on_cpu";
//     RET_CODE ret = RET_CODE::FAILED;
//     std::vector<float*> model_outputs;
//     std::vector<float> aspect_ratios;
//     //临时BBox, 用于存放扩大后的bbox
//     vector<BBox> bboxesXL;
//     for(int i = 0; i < bboxes.size(); i++ ){
//         BBox bbox;
//         bbox.rect = scaleRect(bboxes[i].rect, m_expand_ratio);
//         bbox.objtype = bboxes[i].objtype;
//         bboxesXL.push_back(bbox);
//     }
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_infer_bgr_on_cpu(tvimage, bboxesXL ,model_outputs, aspect_ratios, m_in_valid_cls);
//     }
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     //TODO post processing
//     for(int i=0; i< model_outputs.size(); i++){
//         if(model_outputs[i]==nullptr) continue;
//         ret = postprocess(model_outputs[i], bboxes[i]);
//         if(ret!=RET_CODE::SUCCESS)
//             break;        
//     }
//     for(int i=0; i< model_outputs.size(); i++){
//         if(model_outputs[i]!=nullptr)
//             free(model_outputs[i]);
//     }
//     model_outputs.clear();
//     return ret;
// }

// RET_CODE BinaryClassification::postprocess(float *model_output, BBox &bbox){
//     float score = model_output[m_primary_rank];
//     bbox.confidence = score;
//     bbox.objectness = score;
//     if(score > m_threshold )
//         bbox.objtype = m_cls;
//     else
//         bbox.objtype = CLS_TYPE::UNKNOWN;
//     return RET_CODE::SUCCESS;
// }

// RET_CODE BinaryClassification::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
//     if(batch_tvimages.empty()) return RET_CODE::SUCCESS;
//     RET_CODE ret = run(batch_tvimages[0], bboxes);
//     return ret;
// }

// RET_CODE BinaryClassification::run(TvaiImage &tvimage, VecObjBBox &bboxes){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
//         ret = run_bgr_on_cpu(tvimage, bboxes);
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

// RET_CODE BinaryClassification::get_class_type(std::vector<CLS_TYPE> &clss){
//     clss.push_back(m_cls);
//     return RET_CODE::SUCCESS;
// }

// RET_CODE BinaryClassification::set_filter_cls(std::vector<CLS_TYPE> &cls_seqs){
//     m_in_valid_cls = cls_seqs;
//     return RET_CODE::SUCCESS;
// }

// RET_CODE BinaryClassification::set_primary_output_cls(int rank, CLS_TYPE cls){
//     if (rank > 1) rank = 1;
//     if (rank < 0) rank = 0;
//     m_primary_rank = rank;
//     m_cls = cls;
//     return RET_CODE::SUCCESS;
// }
// ////////////////////////////////////////////////////////////////////////////////////////////////
// // End of BinaryClassification
// ////////////////////////////////////////////////////////////////////////////////////////////////