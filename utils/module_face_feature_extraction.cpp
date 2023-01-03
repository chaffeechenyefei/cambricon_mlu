#include "module_face_feature_extraction.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <cnrt.h>
#include <math.h>
#include "basic.hpp"
// #include "../inner_utils/inner_basic.hpp"
#include <fstream>
// #include <cn_api.h>
#include "module_retinaface_detection.hpp"

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

// #include <future>
using namespace ucloud;
using namespace cv;

static float decode_age(float *output){
    float real_age = 0;
    for(int i = 0; i < 101; i++){
        real_age += output[i]*(i+1);
    }
    return real_age;
}

/*******************************************************************************
FaceExtractionV4 使用BaseModelV2
*******************************************************************************/
FaceExtractionV4::FaceExtractionV4(){
    m_net = std::make_shared<BaseModelV2>();
}

RET_CODE FaceExtractionV4::init(const std::string &modelpath){
    LOGI << "-> FaceExtractionV4::init";
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    //Self param
    LOGI << "<- FaceExtractionV4::init";
    return ret;
}

RET_CODE FaceExtractionV4::init(std::map<InitParam, std::string> &modelpath){
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) {
            printf("auto_model_file_search failed, return %d\n",ret);
            return ret;
        }
    }

    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath[InitParam::BASE_MODEL], config);
    //Self param
    return ret;
}

//clear self param
FaceExtractionV4::~FaceExtractionV4(){LOGI << "-> FaceExtractionV4::~FaceExtractionV4";}

RET_CODE FaceExtractionV4::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(_cls_);
    return RET_CODE::SUCCESS;
};

RET_CODE FaceExtractionV4::postprocess(float* model_output, BBox &bbox){
    int featDim = m_net->m_outputShape[0].DataCount();
    normalize_l2_unit(model_output,featDim);
    if(bbox.feat.pFeature!=nullptr && bbox.feat.featureLen!=0){
        free(bbox.feat.pFeature);
    }
    bbox.feat.featureLen = featDim*sizeof(float);
    bbox.feat.pFeature = reinterpret_cast<unsigned char*>(model_output);
    return RET_CODE::SUCCESS;
}


RET_CODE FaceExtractionV4::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::SUCCESS;
    float preprocess_time{0}, npu_inference_time{0}, postprocess_time{0};
    {
        if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
            std::vector<CLS_TYPE> valid_cls = {CLS_TYPE::FACE};
            std::vector<float*> _model_outputs;
            std::vector<float> aspect_ratios;
            {
                ret = m_net->general_preprocess_infer_bgr_on_cpu(tvimage, bboxes,_model_outputs, aspect_ratios, valid_cls);
            }
            if(ret!=RET_CODE::SUCCESS) return ret;
            //TODO post processing
            for(int i=0; i< _model_outputs.size(); i++){
                if(_model_outputs[i]==nullptr) continue;
                ret = postprocess(_model_outputs[i], bboxes[i]);
                // if(ret!=RET_CODE::SUCCESS)默认后处理没有问题, 否则存在内存泄漏问题
                //     break;        
            }
            _model_outputs.clear();
        } else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
            for(int i = 0; i < bboxes.size(); i++){
                float** model_output = nullptr;
                if(bboxes[i].objtype !=CLS_TYPE::FACE) continue;
                TvaiRect roiRect = bboxes[i].rect;
                float aspect_ratio = 1.0;
                float aX, aY;
                {
                    m_Tk.start();
                    ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roiRect, aspect_ratio, aX , aY);
                    preprocess_time = m_Tk.end("preprocess",false);
                    if(ret!=RET_CODE::SUCCESS) return ret;
                    m_Tk.start();
                    model_output = m_net->general_mlu_infer();
                    npu_inference_time = m_Tk.end("npu inference", false);
                }
                
                //TODO post process
                m_Tk.start(); 
                ret = postprocess(model_output[0],bboxes[i]);
                postprocess_time = m_Tk.end("postprocess", false);
                if(model_output) free(model_output);
                bboxes[i].tmInfo = {preprocess_time, npu_inference_time, postprocess_time};
            }
        } else
            ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;    
        if(ret!=RET_CODE::SUCCESS) return ret;
    }
    return ret;
}



/*******************************************************************************
FaceAttributionV4 使用BaseModelV2
*******************************************************************************/

float FaceAttributionV4::get_box_expand_ratio(){
    return _expand_ratio;
}

RET_CODE FaceAttributionV4::init(std::string &modelpath){
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    //Self param
    return ret;    
}

RET_CODE FaceAttributionV4::init(std::map<InitParam, std::string> &modelpath){
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }

    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    bool pad_both_side = true;//两边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath[InitParam::BASE_MODEL], config);
    //Self param
    return ret;
}

//clear self param
FaceAttributionV4::~FaceAttributionV4(){LOGI << "-> FaceAttributionV4::~FaceAttributionV4";}

RET_CODE FaceAttributionV4::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(_cls_);
    return RET_CODE::SUCCESS;
};


RET_CODE FaceAttributionV4::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
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



RET_CODE FaceAttributionV4::postprocess(float* model_output, BBox &bbox){
    int featDim = m_net->m_outputShape[0].DataCount();
    if(featDim != 103 ) return RET_CODE::ERR_MODEL_NOT_MATCH;
    float probWomanVsMan = model_output[0];
    float *age_vec = &model_output[2];
    float age = decode_age(age_vec);

    UcloudJsonEncoder jsonWriter;
    jsonWriter.initial_context_with_string(bbox.desc);
    jsonWriter.add_context(tagJSON_ROOT::FACE_ATTRIBUTION, tagJSON_ATTR::AGE , std::to_string(int(age)) );
    jsonWriter.add_context(tagJSON_ROOT::FACE_ATTRIBUTION, tagJSON_ATTR::SEX, probWomanVsMan > 0.5 ? "Female":"Male" );
    std::string json_file = jsonWriter.output_to_string();
    bbox.desc = json_file;

    LOGI << "====JSON====";
    LOGI << json_file;
    //TODO
    return RET_CODE::SUCCESS;
}

RET_CODE FaceAttributionV4::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes){
    LOGI << "-> FaceAttributionV4::run_yuv_on_mlu_phyAddr";
    RET_CODE ret = RET_CODE::SUCCESS;
    for(int i = 0; i < bboxes.size(); i++){
        if(bboxes[i].objtype !=CLS_TYPE::FACE) continue;
        TvaiRect roiRect = bboxes[i].rect;
        roiRect = globalscaleTvaiRect(roiRect, 2.0/FaceDetectionV4::get_box_expand_ratio(), tvimage.width, tvimage.height);
        float aspect_ratio = 1.0;
        float aX, aY;
        float** model_output = nullptr;
        {
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roiRect, aspect_ratio, aX, aY);
            if(ret!=RET_CODE::SUCCESS) return ret;
            model_output = m_net->general_mlu_infer();
        }
        //TODO post process 
        ret = postprocess(model_output[0],bboxes[i]);
        m_net->cpu_free(model_output);
    }
    return RET_CODE::SUCCESS;
}


RET_CODE FaceAttributionV4::run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes){
    /**
     * ATT. 存在问题, 没有进行 rect expand!!!!
     */
    LOGI << "-> FaceAttributionV4::run_bgr_on_cpu";
    RET_CODE ret = RET_CODE::SUCCESS;
    std::vector<CLS_TYPE> valid_cls = {CLS_TYPE::FACE};
    std::vector<float*> model_outputs;
    std::vector<float> aspect_ratios;
    VecObjBBox bboxes_tmp = bboxes;
    for(auto &&box: bboxes_tmp){
        box.rect = globalscaleTvaiRect(box.rect, 2.0/FaceDetectionV4::get_box_expand_ratio(), tvimage.width, tvimage.height);
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

// //inner function
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// //人脸特征提取
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// RET_CODE FaceExtractionV2::init(const std::string &modelpath){
//     LOGI << "-> FaceExtractionV2::init";
//     bool pad_both_side = true;//两边预留
//     bool keep_aspect_ratio = true;//保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);
//     //Self param
//     return ret;
// }

// RET_CODE FaceExtractionV2::auto_model_file_search(std::map<InitParam, std::string> &modelpath){
//     std::string basemodelfile, trackmodelfile;
//     for(auto &&m_root: m_roots){
//         std::vector<std::string> modelfiles;
//         ls_files(m_root, modelfiles, ".cambricon");
//         for(auto &&modelfile: modelfiles){
//             if(hasBegining(modelfile, m_basemodel_startswith)) basemodelfile = m_root + modelfile;
//         }
//     }

//     std::cout << "auto loading model from: " << basemodelfile << std::endl;
//     if(basemodelfile.empty() || basemodelfile==""){
//         return RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
//     }

//     modelpath = { {InitParam::BASE_MODEL, basemodelfile} };
//     return RET_CODE::SUCCESS;
// }

// RET_CODE FaceExtractionV2::init(std::map<InitParam, std::string> &modelpath){
//     if(use_auto_model){
//         RET_CODE ret = auto_model_file_search(modelpath);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//     }

//     if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
//     bool pad_both_side = true;//两边预留
//     bool keep_aspect_ratio = true;//保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath[InitParam::BASE_MODEL], config);
//     //Self param
//     return ret;
// }

// //clear self param
// FaceExtractionV2::~FaceExtractionV2(){LOGI << "-> FaceExtractionV2::~FaceExtractionV2";}


// RET_CODE FaceExtractionV2::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> FaceExtractionV2::run_yuv_on_mlu_phyAddr";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         if(bboxes[i].objtype !=CLS_TYPE::FACE) continue;
//         TvaiRect roiRect = bboxes[i].rect;
//         float aspect_ratio = 1.0;
//         float aX, aY;
//         float* model_output = nullptr;
//         {
//             std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             model_output = BaseModel::general_mlu_infer();
//         }
        
//         //TODO post process 
//         ret = postprocess(model_output,bboxes[i]);
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE FaceExtractionV2::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> FaceExtractionV2::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         if(bboxes[i].objtype !=CLS_TYPE::FACE) continue;
//         TvaiRect roiRect = bboxes[i].rect;
//         float aspect_ratio = 1.0;
//         float aX,aY;
//         float* model_output = nullptr;
//         {
//             std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu(tvimage, roiRect, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             model_output = BaseModel::general_mlu_infer();
//         }
//         //TODO post process 
//         ret = postprocess(model_output,bboxes[i]);
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE FaceExtractionV2::run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> FaceExtractionV2::run_bgr_on_cpu";
//     RET_CODE ret = RET_CODE::FAILED;
//     // std::vector<TvaiRect> roiRects;
//     // for(int i=0; i < bboxes.size(); i++){
//     //     roiRects.push_back(bboxes[i].rect);
//     // }
//     std::vector<CLS_TYPE> valid_cls = {CLS_TYPE::FACE};
//     std::vector<float*> model_outputs;
//     std::vector<float> aspect_ratios;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_infer_bgr_on_cpu(tvimage, bboxes,model_outputs, aspect_ratios, valid_cls);
//     }
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     //TODO post processing
//     for(int i=0; i< model_outputs.size(); i++){
//         if(model_outputs[i]==nullptr) continue;
//         ret = postprocess(model_outputs[i], bboxes[i]);
//         // if(ret!=RET_CODE::SUCCESS)默认后处理没有问题, 否则存在内存泄漏问题
//         //     break;        
//     }
//     model_outputs.clear();
//     return ret;
// }

// RET_CODE FaceExtractionV2::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
//     if(batch_tvimages.empty()) return RET_CODE::SUCCESS;
//     RET_CODE ret = run(batch_tvimages[0], bboxes);
//     return ret;
// }

// RET_CODE FaceExtractionV2::run(TvaiImage& tvimage, VecObjBBox &bboxes){
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



// RET_CODE FaceExtractionV2::postprocess(float* model_output, BBox &bbox){
//     int featDim = _oH*_oW*_oC;
//     normalize_l2_unit(model_output,featDim);
//     if(bbox.feat.pFeature!=nullptr && bbox.feat.featureLen!=0){
//         free(bbox.feat.pFeature);
//     }
//     bbox.feat.featureLen = featDim*sizeof(float);
//     bbox.feat.pFeature = reinterpret_cast<unsigned char*>(model_output);
//     return RET_CODE::SUCCESS;
// }

// RET_CODE FaceExtractionV2::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss.push_back(_cls_);
//     return RET_CODE::SUCCESS;
// };







// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// //人脸属性分类(年龄、性别)
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// float FaceAttribution::get_box_expand_ratio(){
//     return _expand_ratio;
// }

// RET_CODE FaceAttribution::init(std::string &modelpath){
//     bool pad_both_side = true;//两边预留
//     bool keep_aspect_ratio = true;//保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);
//     //Self param
//     return ret;    
// }

// RET_CODE FaceAttribution::init(std::map<InitParam, std::string> &modelpath){
//     if(use_auto_model){
//         RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//     }

//     if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
//     bool pad_both_side = true;//两边预留
//     bool keep_aspect_ratio = true;//保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath[InitParam::BASE_MODEL], config);
//     //Self param
//     return ret;
// }

// //clear self param
// FaceAttribution::~FaceAttribution(){LOGI << "-> FaceAttribution::~FaceAttribution";}

// RET_CODE FaceAttribution::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss.push_back(_cls_);
//     return RET_CODE::SUCCESS;
// };


// RET_CODE FaceAttribution::run(TvaiImage& tvimage, VecObjBBox &bboxes){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
//         ret = run_bgr_on_cpu(tvimage, bboxes);
//     }
//     else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
//         ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes);
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     return ret;
// }

// RET_CODE FaceAttribution::postprocess(float* model_output, BBox &bbox){
//     int featDim = _oH*_oW*_oC;
//     if(featDim != 103 ) return RET_CODE::ERR_MODEL_NOT_MATCH;
//     float probWomanVsMan = model_output[0];
//     float *age_vec = &model_output[2];
//     float age = decode_age(age_vec);

//     UcloudJsonEncoder jsonWriter;
//     jsonWriter.initial_context_with_string(bbox.desc);
//     jsonWriter.add_context(tagJSON_ROOT::FACE_ATTRIBUTION, tagJSON_ATTR::AGE , std::to_string(int(age)) );
//     jsonWriter.add_context(tagJSON_ROOT::FACE_ATTRIBUTION, tagJSON_ATTR::SEX, probWomanVsMan > 0.5 ? "Female":"Male" );
//     std::string json_file = jsonWriter.output_to_string();
//     bbox.desc = json_file;

//     LOGI << "====JSON====";
//     LOGI << json_file;
//     //TODO
//     return RET_CODE::SUCCESS;
// }

// RET_CODE FaceAttribution::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> FaceAttribution::run_yuv_on_mlu_phyAddr";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         if(bboxes[i].objtype !=CLS_TYPE::FACE) continue;
//         TvaiRect roiRect = bboxes[i].rect;
//         roiRect = globalscaleTvaiRect(roiRect, 2.0/FaceDetectionV2::get_box_expand_ratio(), tvimage.width, tvimage.height);
//         float aspect_ratio = 1.0;
//         float aX, aY;
//         float* model_output = nullptr;
//         {
//             std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu_union(tvimage, roiRect, aspect_ratio, aX, aY);
//             // ret = BaseModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             model_output = BaseModel::general_mlu_infer();
//         }
//         //TODO post process 
//         ret = postprocess(model_output,bboxes[i]);
//         free(model_output);
//     }
//     return RET_CODE::SUCCESS;
// }


// RET_CODE FaceAttribution::run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes){
//     /**
//      * ATT. 存在问题, 没有进行 rect expand!!!!
//      */
//     LOGI << "-> FaceAttribution::run_bgr_on_cpu";
//     RET_CODE ret = RET_CODE::FAILED;
//     std::vector<CLS_TYPE> valid_cls = {CLS_TYPE::FACE};
//     std::vector<float*> model_outputs;
//     std::vector<float> aspect_ratios;
//     VecObjBBox bboxes_tmp = bboxes;
//     for(auto &&box: bboxes_tmp){
//         box.rect = globalscaleTvaiRect(box.rect, 2.0/FaceDetectionV2::get_box_expand_ratio(), tvimage.width, tvimage.height);
//     }
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_infer_bgr_on_cpu(tvimage, bboxes_tmp,model_outputs, aspect_ratios, valid_cls);
//     }
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     //TODO post processing
//     for(int i=0; i< model_outputs.size(); i++){
//         if(model_outputs[i]==nullptr) continue;
//         ret = postprocess(model_outputs[i], bboxes[i]);
//         free(model_outputs[i]);
//         // if(ret!=RET_CODE::SUCCESS)默认后处理没有问题, 否则存在内存泄漏问题
//         //     break;        
//     }
//     model_outputs.clear();
//     return ret;
// }




