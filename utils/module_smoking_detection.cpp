#include "module_smoking_detection.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include <fstream>

#include "module_smoking_detection.hpp"
#include "module_yolo_detection.hpp"
#include "module_retinaface_detection.hpp"


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
using std::vector;
using std::cout;
using std::endl;

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SmokingDetection BEGIN
////////////////////////////////////////////////////////////////////////////////////////////////////////////
static bool objectnessGreaterSort(BBox &a, BBox &b){
    return a.objectness > b.objectness;
}

static void recalcxyxy(VecObjBBox &boxes){
    for(auto &&box: boxes){
        box.x0 = box.rect.x;
        box.y0 = box.rect.y;
        box.x1 = box.rect.x + box.rect.width;
        box.y1 = box.rect.y + box.rect.height;
    }
}

static void transform_face_to_mouth(VecObjBBox &bboxes){
    for(auto &&box: bboxes){
        int cx = box.rect.x + box.rect.width/2;
        int cy = box.rect.y + box.rect.height/2;
        int L = std::max(box.rect.width, box.rect.height) + 0.3*std::min(box.rect.width, box.rect.height);
        L /= 1.3;
        box.rect.x = cx - L / 2;
        box.rect.y = cy;
        box.rect.width = L;
        box.rect.height = L/2;
    }
}

static void expand_hand_region(VecObjBBox &bboxes, float pad_ratio=0.2){
    for(auto &&box: bboxes){
        int pad = std::max(box.rect.width,box.rect.height)*pad_ratio;
        box.rect.x -= pad;
        box.rect.y -=pad;
        box.rect.width += 2*pad;
        box.rect.height += 2*pad;
    }
}

/**
 * FUNC: func_is_matched
 * PARAM:
 *  a: ped
 *  b: hand/face
 */
static bool func_is_matched(BBox &a, BBox &b){
    float match_iou_threshold = 0.9;
    // float area0 = (a.y1 - a.y0 + 1)*(a.x1 - a.x0 + 1);
    float area1 = (b.y1 - b.y0 + 1)*(b.x1 - b.x0 + 1);
    float roiWidth = std::min(a.x1, b.x1) - std::max(a.x0, b.x0);
    float roiHeight = std::min(a.y1, b.y1) - std::max(a.y0, b.y0);
    if (roiWidth<=0 || roiHeight<=0) return false;
    float ratio = roiWidth*roiHeight / area1;
    if (ratio > match_iou_threshold) return true;
    else return false;
}

static void merge_bboxes(VecObjBBox &ped_bboxes, VecObjBBox &hand_bboxes, VecObjBBox &face_bboxes, 
                         VecSmokingBox &target_bboxes, VecObjBBox &remain_ped_bboxes){
    std::sort(ped_bboxes.begin(), ped_bboxes.end(), objectnessGreaterSort);//降序排列
    std::sort(hand_bboxes.begin(), hand_bboxes.end(), objectnessGreaterSort);//降序排列
    std::sort(face_bboxes.begin(), face_bboxes.end(), objectnessGreaterSort);//降序排列
    vector<int> hand_bboxes_used(hand_bboxes.size(),0);
    vector<int> face_bboxes_used(face_bboxes.size(),0);
    //Transform TvaiRect into x0,y0,x1,y1
    recalcxyxy(ped_bboxes);
    recalcxyxy(hand_bboxes);
    recalcxyxy(face_bboxes);

    for(auto &&ped_bbox: ped_bboxes){//Loop ped
        bool flag_matched = false;
        for(int f=0; f < face_bboxes.size(); f++){//Loop face
            if(face_bboxes_used[f]!=0) continue;
            if(func_is_matched(ped_bbox, face_bboxes[f])){
                face_bboxes_used[f] = 1;
                flag_matched = true;
                SMOKING_BOX temp_box;
                temp_box.body = ped_bbox;
                temp_box.face = face_bboxes[f];
                VecObjBBox temp_hand_boxes;
                for(int h=0; h < hand_bboxes.size(); h++){//Loop hand
                    if(hand_bboxes_used[h]!=0) continue;
                    if(func_is_matched(ped_bbox, hand_bboxes[h])){
                        temp_hand_boxes.push_back(hand_bboxes[h]);
                        hand_bboxes_used[h] = 1;
                        if(temp_hand_boxes.size()>=2) break;//Stop loop of hand, if find 2 hands.
                    }
                }
                if(temp_hand_boxes.size()==1){
                    temp_box.handl = temp_hand_boxes[0];
                    temp_box.handr = BBox();
                }
                else if(temp_hand_boxes.size()==2){
                    temp_box.handl = temp_hand_boxes[0];
                    temp_box.handr = temp_hand_boxes[1];
                } else{
                    temp_box.handl = BBox();
                    temp_box.handr = BBox();
                }
                // temp_box.handl = BBox();
                // temp_box.handr = BBox();
                target_bboxes.push_back(temp_box);
                break;//Stop loop of face, if find.
            }
        }
        if(!flag_matched){//put unmatched box into remain_ped_bboxes
            remain_ped_bboxes.push_back(ped_bbox);
        }
    }//End of loop ped
}


// RET_CODE SmokingDetection::init(std::map<InitParam, std::string> &modelpath){
//     LOGI << "-> SmokingDetection::init";
//     std::string ped_detect_modelpath, hand_detect_modelpath, face_detect_modelpath ,classify_modelpath;
//     if(modelpath.find(InitParam::PED_MODEL)==modelpath.end() || \
//         modelpath.find(InitParam::FACE_MODEL)==modelpath.end() || \
//         modelpath.find(InitParam::SUB_MODEL)==modelpath.end()) {
//             std::cout << modelpath.size() << endl;
//             for(auto param: modelpath){
//                 LOGI << param.first << "," << param.second;
//             }
//             return RET_CODE::ERR_INIT_PARAM_FAILED;
//         }

//     ped_detect_modelpath = modelpath[InitParam::PED_MODEL];
//     face_detect_modelpath = modelpath[InitParam::FACE_MODEL];
//     classify_modelpath = modelpath[InitParam::SUB_MODEL];

//     ucloud::TvaiResolution maxTarget={0,0};
//     ucloud::TvaiResolution minTarget={0,0};
//     std::vector<ucloud::TvaiRect> pRoi;

//     //ped detection
//     YoloDetectionV4* ped_detector = new YoloDetectionV4();
//     ped_detector->init(ped_detect_modelpath);
//     vector<CLS_TYPE> yolov5s_conv_9 = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
//     ped_detector->set_output_cls_order(yolov5s_conv_9);
//     ped_detector->set_param(m_ped_threshold,0.2,maxTarget, minTarget, pRoi);
//     m_ped_detectHandle.reset(ped_detector);
//     //face detection
//     m_face_detectHandle = std::make_shared<FaceDetectionV4>();
//     m_face_detectHandle->init(face_detect_modelpath);
//     m_face_detectHandle->set_param(m_face_threshold,0.2, maxTarget, minTarget, pRoi);
//     //hand detection
//     if(modelpath.find(InitParam::HAND_MODEL)!=modelpath.end()){
//         hand_detect_modelpath = modelpath[InitParam::HAND_MODEL];
//         YoloDetectionV4* hand_detector = new YoloDetectionV4();
//         hand_detector->init(hand_detect_modelpath);
//         vector<CLS_TYPE> yolov5s_hand = {CLS_TYPE::HAND};
//         hand_detector->set_output_cls_order(yolov5s_hand);
//         hand_detector->set_param(m_hand_threshold,0.2,maxTarget, minTarget, pRoi);
//         m_hand_detectHandle.reset(hand_detector);
//     } else {
//         m_hand_detectHandle = nullptr;
//     }
    
//     //smoking classifier
//     SmokingClassification* classifier = new SmokingClassification();
//     classifier->init(classify_modelpath);
//     m_classifyHandle.reset(classifier);
    
//     return RET_CODE::SUCCESS;
// }

// RET_CODE SmokingDetection::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
//     if(batch_tvimages.empty()) return RET_CODE::SUCCESS;
//     return run(batch_tvimages[0], bboxes);
// }

// RET_CODE SmokingDetection::run(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> SmokingDetection::run";
//     if(tvimage.format!=TVAI_IMAGE_FORMAT_NV21 && tvimage.format!=TVAI_IMAGE_FORMAT_NV12 ) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     VecObjBBox det_bboxes;
//     m_ped_detectHandle->run(tvimage, det_bboxes);
//     VecObjBBox ped_bboxes;
//     for(auto &&box: det_bboxes){
//         if(box.objtype == CLS_TYPE::PEDESTRIAN)
//             ped_bboxes.push_back(box);
//     }
//     LOGI << "ped detected: " << ped_bboxes.size();

//     VecObjBBox face_bboxes;
//     m_face_detectHandle->run(tvimage, face_bboxes);
//     LOGI << "face detected: " << face_bboxes.size();
//     transform_face_to_mouth(face_bboxes);
    
//     VecObjBBox hand_bboxes;
//     if(m_hand_detectHandle!=nullptr){
//         m_hand_detectHandle->run(tvimage, hand_bboxes);
//     }
//     LOGI << "hand detected: " << hand_bboxes.size();
//     expand_hand_region(hand_bboxes);

//     //TODO: merge hand,face,body into a struct
//     VecSmokingBox cand_bboxes;
//     VecObjBBox ped_bboxes_remain;
//     merge_bboxes(ped_bboxes, hand_bboxes, face_bboxes, cand_bboxes, ped_bboxes_remain);
//     LOGI << "target detected: " << cand_bboxes.size();

//     //Classification
//     m_classifyHandle->run(tvimage, cand_bboxes);
//     for(auto &&box: cand_bboxes){
//         if(box.body.objtype == m_cls){
//             bboxes.push_back(box.body);
//             bboxes.push_back(box.face);
//             if(box.handl.objtype!=CLS_TYPE::UNKNOWN)
//                 bboxes.push_back(box.handl);
//             if(box.handr.objtype!=CLS_TYPE::UNKNOWN)
//                 bboxes.push_back(box.handr);
//         }
            
//     }
//     return RET_CODE::SUCCESS;
// }



// RET_CODE SmokingDetection::set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect){
//     if(float_in_range(threshold,1,0)){
//         m_threshold = threshold;
//         m_classifyHandle->set_threshold(m_threshold);
//     }
//     else
//         return RET_CODE::ERR_INIT_PARAM_FAILED;
//     m_maxTargeSize = base_get_valid_maxSize(maxTargetSize);
//     m_minTargeSize = minTargetSize;
//     std::vector<TvaiRect>().swap(m_pAoiRect);
//     m_pAoiRect.clear();
//     for(int i=0; i<pAoiRect.size(); i++ ){
//         m_pAoiRect.push_back(pAoiRect[i]);
//     }
//     return RET_CODE::SUCCESS;    
// }


// RET_CODE SmokingDetection::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss.push_back(m_cls);
//     return RET_CODE::SUCCESS;
// }
////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SmokingDetection END
////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SmokingDetectionV2 BEGIN
////////////////////////////////////////////////////////////////////////////////////////////////////////////
RET_CODE SmokingDetectionV2::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> SmokingDetectionV2::init";
    std::string face_detect_modelpath ,cig_detect_modelpath;
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }

    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end() || \
        modelpath.find(InitParam::SUB_MODEL)==modelpath.end()) {
            std::cout << modelpath.size() << endl;
            for(auto param: modelpath){
                printf( "[%d]:[%s], ", param.first, param.second);
            }
            printf("ERR:: SmokingDetectionV2->init() still missing models\n");
            return RET_CODE::ERR_INIT_PARAM_FAILED;
        }
    RET_CODE ret = RET_CODE::FAILED;
    face_detect_modelpath = modelpath[InitParam::BASE_MODEL];
    cig_detect_modelpath = modelpath[InitParam::SUB_MODEL];

    //face detection
    ret = m_face_detectHandle->init(face_detect_modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;

    //cig detection
    ret = m_cig_detectHandle->init(cig_detect_modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;
    vector<CLS_TYPE> cls_types = {CLS_TYPE::SMOKING};
    m_cig_detectHandle->set_output_cls_order(cls_types);
    
    return RET_CODE::SUCCESS;
}

RET_CODE SmokingDetectionV2::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> SmokingDetectionV2::run";
    if(tvimage.format!=TVAI_IMAGE_FORMAT_NV21 && tvimage.format!=TVAI_IMAGE_FORMAT_NV12 ) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    RET_CODE ret = RET_CODE::FAILED;
    float expand_scale = 1.5;
    VecObjBBox face_bboxes;
    ret = m_face_detectHandle->run(tvimage, face_bboxes, m_face_threshold, 0.6);
    for(auto &&face_bbox: face_bboxes){
        VecObjBBox target_bboxes;
        TvaiRect scaled_face_rect = globalscaleTvaiRect(face_bbox.rect, expand_scale, tvimage.width, tvimage.height);
        ret = m_cig_detectHandle->run(tvimage, face_bbox.rect, target_bboxes, threshold, nms_threshold);
        if(!target_bboxes.empty()){
            face_bbox.confidence = target_bboxes[0].confidence;
            face_bbox.objectness = target_bboxes[0].objectness;
            face_bbox.rect = scaled_face_rect;
            face_bbox.objtype = target_bboxes[0].objtype;
            bboxes.push_back(face_bbox);
        } 
#ifndef MLU220 //270情况下, 同时返回香烟的检测结果以及人脸框
        // else {
        //     bboxes.push_back(face_bbox);
        // }
        for(auto &&target_bbox: target_bboxes)
        {
            bboxes.push_back(target_bbox);
        }
#endif
    }
    return RET_CODE::SUCCESS;
}

RET_CODE SmokingDetectionV2::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(m_cls);
    return RET_CODE::SUCCESS;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SmokingDetectionV2 END
////////////////////////////////////////////////////////////////////////////////////////////////////////////