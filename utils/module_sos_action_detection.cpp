#include "module_smoking_detection.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include <fstream>

#include "module_sos_action_detection.hpp"



#ifdef DEBUG
#include <chrono>
#include <sys/time.h>
#include "../inner_utils/module.hpp"
#endif

#define PI 3.1415

// #include <future>
using namespace ucloud;
using namespace cv;
using std::vector;
using std::cout;
using std::endl;

static float calc_angle(uPoint &a, uPoint &b, uPoint &c);
//angle between ac and bc. [0-180]
float calc_angle(uPoint &a, uPoint &b, uPoint &c){
    uPoint ac{0,0};
    ac.x = a.x - c.x;
    ac.y = a.y - c.y;
    uPoint bc{0,0};
    bc.x = b.x - c.x;
    bc.y = b.y - c.y;

    float m = ac.x*bc.x + ac.y*bc.y;
    float n = std::sqrt( ac.x * ac.x + ac.y * ac.y )* std::sqrt( bc.x * bc.x + bc.y * bc.y ) + 1e-3;
    float cos_abc = std::acos(m/n)*180/PI;
    // cout << cos_abc << endl;
    //[-180,180]
    if( cos_abc > 180) cos_abc = cos_abc - 360;
    if( cos_abc < -180) cos_abc = cos_abc + 360;
    //[0,180]
    return std::abs(cos_abc);
}
/**Reference
 * "keypoints": {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
},
*/


////////////////////////////////////////////////////////////////////////////////
// SOSDetectionV1 BEGIN
////////////////////////////////////////////////////////////////////////////////
// RET_CODE SOSDetectionV1::init(std::map<InitParam, std::string> &modelpath){
//     LOGI << "-> SOSDetectionV1::init";
//     std::string ped_detect_modelpath ,sk_detect_modelpath;
//     if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end() || \
//         modelpath.find(InitParam::SUB_MODEL)==modelpath.end()) {
//             std::cout << modelpath.size() << endl;
//             for(auto param: modelpath){
//                 LOGI << param.first << "," << param.second;
//             }
//             return RET_CODE::ERR_INIT_PARAM_FAILED;
//         }
    
//     RET_CODE ret = RET_CODE::FAILED;
//     ped_detect_modelpath = modelpath[InitParam::BASE_MODEL];
//     sk_detect_modelpath = modelpath[InitParam::SUB_MODEL];

//     ucloud::TvaiResolution maxTarget={0,0};
//     ucloud::TvaiResolution minTarget={0,0};
//     std::vector<ucloud::TvaiRect> pRoi;

//     //Initial ped detector
//     YoloDetectionV4 *_ptr = new YoloDetectionV4();
//     std::vector<CLS_TYPE> yolov5s_conv_9 = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
//     _ptr->set_output_cls_order(yolov5s_conv_9);
//     m_ped_detector.reset(_ptr);
//     ret = m_ped_detector->init(ped_detect_modelpath);
//     // m_ped_detector->set_param(m_ped_threshold, 0.2, maxTarget, minTarget, pRoi);
//     if(ret!=RET_CODE::SUCCESS) return ret;

//     //Initial sk detector
//     m_sk_detector = std::make_shared<SkeletonDetector>();
//     ret = m_sk_detector->init(sk_detect_modelpath);
//     if(ret!=RET_CODE::SUCCESS) return ret;

//     return ret;
// }

// RET_CODE SOSDetectionV1::set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect){
//     if(float_in_range(threshold,1,0)){
//         m_ped_threshold = threshold;
//         RET_CODE ret = m_ped_detector->set_param(m_ped_threshold, nms_threshold, maxTargetSize, minTargetSize, pAoiRect );
//         return ret;
//     }
//     else return RET_CODE::ERR_INIT_PARAM_FAILED;
//     return RET_CODE::SUCCESS;    
// }


// RET_CODE SOSDetectionV1::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss = {m_cls};
//     return RET_CODE::SUCCESS;
// }

// RET_CODE SOSDetectionV1::run(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> SOSDetectionV1::run";
//     RET_CODE ret = RET_CODE::FAILED;
//     VecObjBBox ped_bboxes;
//     ret = m_ped_detector->run(tvimage, ped_bboxes);
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     ret = m_sk_detector->run(tvimage, ped_bboxes);
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     for(auto &&ped_box: ped_bboxes){
//         if(is_pose_sos(ped_box)){
//             ped_box.objtype = m_cls;
//             bboxes.push_back(ped_box);
//         }
//     #ifndef MLU220
//         else{
//             bboxes.push_back(ped_box);
//         }
//     #endif
//     }
//     return RET_CODE::SUCCESS;
// }



// /**
//  * Rule:
//  * 1. 手肘垂直
//  * 2. 手腕过头顶
//  */
// bool SOSDetectionV1::is_pose_sos(BBox &box){
//     float threshold_wrist_elbow_angle = 30;
//     uPoint head_center = box.Pts.pts[0];
//     uPoint left_wrist = box.Pts.pts[9];
//     uPoint left_elbow = box.Pts.pts[7];
//     uPoint left_axis = left_elbow;
//     left_axis.y -= 10;
//     uPoint right_wrist = box.Pts.pts[10];
//     uPoint right_elbow = box.Pts.pts[8];
//     uPoint right_axis = right_elbow;
//     right_axis.y -= 10;

//     box.Pts.pts = {head_center, left_wrist, left_elbow, right_wrist, right_elbow};

//     //手肘与手腕的中心是否高于鼻子
//     bool b_left_wrist_overhead = (head_center.y > (left_elbow.y + left_wrist.y)/2) ? true: false;
//     bool b_right_wrist_overhead = (head_center.y > (right_elbow.y + right_wrist.y)/2) ? true: false;
//     //如果都不高于鼻子, 则直接返回失败
//     if(!b_left_wrist_overhead && !b_right_wrist_overhead ) return false;

//     //对高于鼻子的情况进行判断, 手肘与手腕的矢量是否垂直画面
//     if(b_left_wrist_overhead){
//         float left_angle = calc_angle(left_wrist, left_axis, left_elbow);
//         if(left_angle <= threshold_wrist_elbow_angle) return true;
//     }

//     if(b_right_wrist_overhead){
//         float right_angle = calc_angle(right_wrist, right_axis, right_elbow);
//         if(right_angle <= threshold_wrist_elbow_angle) return true;
//     }

//     return false;
// }
////////////////////////////////////////////////////////////////////////////////
// SOSDetectionV1 END
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// SOSDetectionV2 BEGIN
////////////////////////////////////////////////////////////////////////////////

RET_CODE SOSDetectionV2::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> SOSDetectionV2::init";
    std::string ped_detect_modelpath ,hand_detect_modelpath;
    if( use_auto_model ){
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
    
    RET_CODE ret = RET_CODE::FAILED;
    ped_detect_modelpath = modelpath[InitParam::BASE_MODEL];
    hand_detect_modelpath = modelpath[InitParam::SUB_MODEL];

    //Initial ped detector
    std::vector<CLS_TYPE> yolov5s_conv_9 = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
    m_ped_detector->set_output_cls_order(yolov5s_conv_9);
    // _ptr->set_param(m_ped_threshold, 0.6, maxTarget, minTarget, pRoi);
    std::map<InitParam, std::string> modelconfig = {{InitParam::BASE_MODEL, ped_detect_modelpath}};
    ret = m_ped_detector->init(modelconfig);
    if(ret!=RET_CODE::SUCCESS) return ret;

    //Initial hand detector
    std::vector<CLS_TYPE> hand_cls = {CLS_TYPE::HAND};
    m_hand_detector->set_output_cls_order(hand_cls);
    // _ptr2->set_param(m_hand_threshold, 0.2, maxTarget, minTarget, pRoi);
    ret = m_hand_detector->init(hand_detect_modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;

    return ret;
}

RET_CODE SOSDetectionV2::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss = {m_cls};
    return RET_CODE::SUCCESS;
}

RET_CODE SOSDetectionV2::set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect){
    if(float_in_range(threshold,1,0)){
        m_ped_threshold = threshold;
        RET_CODE ret = RET_CODE::SUCCESS;
        if(m_ped_detector!=nullptr)
            ret = m_ped_detector->set_param(m_ped_threshold, nms_threshold, maxTargetSize, minTargetSize, pAoiRect );
        return ret;
    }
    else return RET_CODE::ERR_INIT_PARAM_FAILED;
    return RET_CODE::SUCCESS;    
}


RET_CODE SOSDetectionV2::run(TvaiImage &tvimage, VecObjBBox &bboxes){
    LOGI << "-> SOSDetectionV1::run";
    RET_CODE ret = RET_CODE::FAILED;
    VecObjBBox ped_bboxes, hand_bboxes;
    ret = m_ped_detector->run(tvimage, ped_bboxes, m_ped_threshold, 0.6);
    if(ret!=RET_CODE::SUCCESS) return ret;
    ret = m_hand_detector->run(tvimage, hand_bboxes, m_hand_threshold, 0.6);
    if(ret!=RET_CODE::SUCCESS) return ret;

    VecSOSBox candboxes;
    VecObjBBox others;
    merge(ped_bboxes, hand_bboxes, candboxes, others);
    // cout << "candboxes: " << candboxes.size() << endl;
    //判断candboxes中是否存在举手动作
    for(auto &&sosbox: candboxes){
        BBox matched_hand_box;
        if(is_sos_trigger(sosbox, matched_hand_box)){
            sosbox.body.objtype = m_cls;
            bboxes.push_back(sosbox.body);
        #ifndef MLU220
            if(matched_hand_box.objtype == CLS_TYPE::HAND)
                bboxes.push_back(matched_hand_box);
        #endif
        }
    }
// #ifndef MLU220
//     bboxes.insert(bboxes.end(), hand_bboxes.begin(), hand_bboxes.end());
// #endif
    return RET_CODE::SUCCESS;
}

/**
 * sosbox: 输入的组合框
 * handbox: 输出被触发的手
 * 触发条件:
 * 框内手, 手在身体框的top10%区域
 * 或
 * 框外手, 手在身体框上部, 且手与身体中心连线30度内
 */
bool SOSDetectionV2::is_sos_trigger(SOSBox &sosbox, BBox &handbox){
    float upper_range = 0.1;
    float y_threshold = sosbox.body.rect.y + sosbox.body.rect.height*upper_range;
    handbox = BBox();
    for(auto &&hand: sosbox.hands_in){
        if(hand.rect.y < y_threshold ){//框内手, 手在身体框的top10%区域
            handbox = hand;
            return true;
        }
    }
    float angle_threshold = 30;
    for(auto &&hand: sosbox.hands_out){
        if(hand.rect.y < sosbox.body.rect.y){//框外手, 手在身体框上部, 且手与身体中心连线30度内
            uPoint hand_pt = {0,0}; uPoint body_pt = {0,0};
            hand_pt.x = hand.rect.x + hand.rect.width/2;
            hand_pt.y = hand.rect.y + hand.rect.height/2;
            body_pt.x= sosbox.body.rect.x + sosbox.body.rect.width/2;
            body_pt.y = sosbox.body.rect.y + sosbox.body.rect.height/2;
            uPoint axis = body_pt;
            axis.y = body_pt.y - 10;
            float angle = calc_angle(hand_pt, axis, body_pt);
            // cout << hand_pt.x << ", " << hand_pt.y << endl;
            // cout << body_pt.x << ", " << body_pt.y << endl;
            // cout << axis.x << ", " << axis.y << endl;
            // cout << angle << endl;
            if(angle < angle_threshold){
                handbox = hand;
                return true;
            }
        }
    }

    return false;
}

static bool isABJoint(BBox& a, BBox& b){
    float roiWidth = std::min(a.x1, b.x1) - std::max(a.x0, b.x0);
    float roiHeight = std::min(a.y1, b.y1) - std::max(a.y0, b.y0);
    if(roiHeight<=0||roiWidth<=0) return false;
    else return true;
}
/**
 * isABNear
 * a: body
 * b: hand
 * 将b扩大N倍, 即默认手掌可以离开身体框的距离
 */
static bool isABNear(BBox& a, BBox& b, float expand_b_ratio = 3){
    float cx = (b.x0 + b.x1)/2;
    float cy = (b.y0 + b.y1)/2;
    float w = expand_b_ratio*(b.x1 - b.x0);
    float h = expand_b_ratio*(b.y1 - b.y0);
    float x0 = cx - w/2;
    float x1 = cx + w/2;
    float y0 = cy - h/2;
    float y1 = cy + h/2;
    BBox expb;
    expb.x0 = x0; expb.y0 = y0; expb.x1 = x1; expb.y1 = y1;
    return isABJoint(a, expb);
} 
static bool sortGreater(BBox& a, BBox& b){
    return a.confidence > b.confidence;
}
static void transferxywh_to_x0y0x1y1(VecObjBBox &bboxes){
    for(auto &&box: bboxes){
        box.x0 = box.rect.x;
        box.y0 = box.rect.y;
        box.x1 = box.rect.x + box.rect.width;
        box.y1 = box.rect.y + box.rect.height;
    }
}
/**
 * 身体和手的合并逻辑:
 * 对每个身体遍历匹配每个手, 手在框内, 则匹配成功. 
 * 手在框外, 则满足一定空间规则时, 匹配成功. 
 * 手可以重复被多个身体使用, 但手已经被匹配为框内的, 不能用于框外.
 */
void SOSDetectionV2::merge(VecObjBBox& bodyboxIN, VecObjBBox& handboxIN, \
                        VecSOSBox &sosboxOUT, VecObjBBox &othersOUT){
    sort(bodyboxIN.begin(), bodyboxIN.end(), sortGreater );
    sort(handboxIN.begin(), handboxIN.end(), sortGreater );

    transferxywh_to_x0y0x1y1(bodyboxIN);
    transferxywh_to_x0y0x1y1(handboxIN);

    VecSOSBox candboxes;
    //TODO: 手在框内
    for(auto &&bodybox: bodyboxIN){
        SOSBox tmp;
        tmp.body = bodybox;
        for(auto &&handbox: handboxIN){
            if(isABJoint(bodybox, handbox)){
                handbox.track_id = 1;
                tmp.hands_in.push_back(handbox);
            }
        }
        candboxes.push_back(tmp);
    }
    //手在框外
    for(auto &&candbox: candboxes){
        for(auto &&handbox: handboxIN){
            if(handbox.track_id>0) continue;//复用track_id, 大于零, 说明已经匹配框内人体, 则不进行匹配
            //手在框外但足够近
            if(isABNear( candbox.body, handbox, 5)){
                candbox.hands_out.push_back(handbox);
            }
        }
    }
    sosboxOUT = candboxes;
}

////////////////////////////////////////////////////////////////////////////////
// SOSDetectionV2 END
////////////////////////////////////////////////////////////////////////////////