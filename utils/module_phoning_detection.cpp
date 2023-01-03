#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include <fstream>

#include "module_phoning_detection.hpp"



#ifdef DEBUG
#include <chrono>
#include <sys/time.h>
#include "../inner_utils/module.hpp"
#endif


// #include <future>
using namespace ucloud;
using namespace cv;
using std::vector;
using std::cout;
using std::endl;

RET_CODE PhoningDetection::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> PhoningDetection::init";
    std::string ped_detect_modelpath ,classify_modelpath;
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }    

    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end() || \
        modelpath.find(InitParam::SUB_MODEL)==modelpath.end()) {
        std::cout << modelpath.size() << endl;
        for(auto param: modelpath){
            LOGI << param.first << "," << param.second;
        }
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    }
    ped_detect_modelpath = modelpath[InitParam::BASE_MODEL];
    classify_modelpath = modelpath[InitParam::SUB_MODEL];

    //ped detection
    RET_CODE ret = RET_CODE::SUCCESS;
    ret = m_ped_detectHandle->init(ped_detect_modelpath);
    if(ret!=RET_CODE::SUCCESS){
        printf("ERR::PhoningDetection m_ped_detectHandle init return [%d]\n", ret);
        return ret;
    }
    vector<CLS_TYPE> yolov5s_conv_9 = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
    ret = m_ped_detectHandle->set_output_cls_order(yolov5s_conv_9);
    if(ret!=RET_CODE::SUCCESS){
        printf("ERR::PhoningDetection m_ped_detectHandle set_output_cls_order return [%d]\n", ret);
        return ret;
    }
    //smoking classifier
    ret = m_classifyHandle->init(classify_modelpath);
    if(ret!=RET_CODE::SUCCESS){
        printf("ERR::PhoningDetection m_classifyHandle init return [%d]\n", ret);
        return ret;
    }
    return RET_CODE::SUCCESS;
}

RET_CODE PhoningDetection::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> PhoningDetection::run";
    if(tvimage.format!=TVAI_IMAGE_FORMAT_NV21 && tvimage.format!=TVAI_IMAGE_FORMAT_NV12 ) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    VecObjBBox det_bboxes;
    RET_CODE ret = m_ped_detectHandle->run(tvimage, det_bboxes, m_ped_threshold, 0.6);
    if(ret!=RET_CODE::SUCCESS) return ret;
    VecObjBBox ped_bboxes;
    for(auto &&box: det_bboxes){
        if(box.objtype == CLS_TYPE::PEDESTRIAN)
            ped_bboxes.push_back(box);
    }
    LOGI << "ped detected: " << ped_bboxes.size();
    // std::cout << "ped detected: " << ped_bboxes.size() << std::endl;
    //TODO: merge hand,face,body into a struct
    VecPedBox cand_bboxes;
    transform_box_to_ped_box(ped_bboxes, cand_bboxes);
    // for (int i = 0 ; i < cand_bboxes.size(); i++ ){
    //     cout << "#" << cand_bboxes[i].target.objtype << " = " <<cand_bboxes[i].target.objectness 
    //     << " [" 
    //     << cand_bboxes[i].target.rect.x << ", " << cand_bboxes[i].target.rect.y << ", " 
    //     << cand_bboxes[i].target.rect.width << ", " << cand_bboxes[i].target.rect.height
    //     << "], " ;
    // }
    // std::cout << std::endl;
    LOGI << "target detected: " << cand_bboxes.size();
    // std::cout << "target detected: " << cand_bboxes.size() << std::endl;

    //Classification
    ret = m_classifyHandle->run(tvimage, cand_bboxes, threshold, nms_threshold);
    if(ret!=RET_CODE::SUCCESS) return ret;
    for(auto &&box: cand_bboxes){
        if(box.body.objtype == m_cls){
            bboxes.push_back(box.body);
        }
        #ifndef MLU220
        else {bboxes.push_back(box.body);}
        #endif
            
    }
    return RET_CODE::SUCCESS;
}


RET_CODE PhoningDetection::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(m_cls);
    return RET_CODE::SUCCESS;
}

void PhoningDetection::transform_box_to_ped_box(VecObjBBox &in_boxes, VecPedBox &out_boxes){
    for( auto &&in_box: in_boxes){
        PED_BOX out_box;
        out_box.body = in_box;
        TvaiRect body_rect = in_box.rect;
        float hw_ratio = ((float)(1.0*body_rect.height))/ body_rect.width;
        if(hw_ratio >= 2){
            body_rect.height *= 0.5;
        }else if(hw_ratio >= 1.5){
            body_rect.height *= 0.8;
        }
        out_box.target = in_box;
        out_box.target.rect = body_rect;
        out_boxes.push_back(out_box);
    }
    return;
}