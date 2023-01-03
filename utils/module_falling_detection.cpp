#include "module_falling_detection.hpp"
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

// #include <future>
using namespace ucloud;
using namespace cv;
using std::vector;
using std::cout;
using std::endl;

#define PI 3.1415

/*******************************************************************************
inner function
*******************************************************************************/
static uPoint get_head_center(BBox &box);
static uPoint get_upper_body_center(BBox &box);
static float calc_angle(uPoint &a, uPoint &b, uPoint &c);
static float calc_angle2(uPoint &a, uPoint &b, uPoint &c);

uPoint get_head_center(BBox &box){
    //HEAD
    uPoint head_center{0,0};
    for(int i=0; i < 5; i++){
        head_center.x += box.Pts.pts[i].x;
        head_center.y += box.Pts.pts[i].y;
    }
    head_center.x /= 5;
    head_center.y /= 5;
    return head_center;
}

uPoint get_upper_body_center(BBox &box){
    uPoint left_shoulder = box.Pts.pts[5];
    uPoint right_shoulder = box.Pts.pts[6];
    uPoint left_hip = box.Pts.pts[11];
    uPoint right_hip = box.Pts.pts[12];
    //UPPER_BODY_CENTER
    uPoint upper_body_center{0,0};
    upper_body_center.x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x)/4;
    upper_body_center.y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y)/4;
    return upper_body_center;
}

//angle between ac and bc. [0-180]
float calc_angle(uPoint &a, uPoint &b, uPoint &c){
    uPoint ac{0,0};
    ac.x = a.x - c.x;
    ac.y = a.y - c.y;
    uPoint bc{0,0};
    bc.x = b.x - c.x;
    bc.y = b.y - c.y;
    /**
     * https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
     * The orientation of this angle matches that of the coordinate system. 
     * In a left-handed coordinate system, i.e. x pointing right and y down as is common for computer graphics, 
     * this will mean you get a positive sign for clockwise angles. 
     * If the orientation of the coordinate system is mathematical with y up, 
     * you get counter-clockwise angles as is the convention in mathematics. 
     * Changing the order of the inputs will change the sign, so if you are unhappy with the signs just swap the inputs.
     * 
     * https://en.cppreference.com/w/cpp/numeric/math/atan2
     * If no errors occur, the arc tangent of y/x (arctan(y/x)) in the range [-π , +π] radians, is returned.
     */
    float dot = ac.x*bc.x + ac.y*bc.y;
    float det = ac.x*bc.y - ac.y*bc.x;
    float angle = std::atan2(det,dot)*180/PI;
    // std::cout << "angle = " << angle << std::endl;
    return std::abs(angle);
    // float m = ac.x*bc.x + ac.y*bc.y;
    // float n = std::sqrt( ac.x * ac.x + ac.y * ac.y )* std::sqrt( bc.x * bc.x + bc.y * bc.y ) + 1e-3;
    // float cos_abc = std::acos(m/n)*180/PI;
    // //[-180,180]
    // if( cos_abc > 180) cos_abc = cos_abc - 360;
    // if( cos_abc < -180) cos_abc = cos_abc + 360;
    // //[0,180]
    // return std::abs(cos_abc);
}


float calc_angle2(uPoint &a, uPoint &b, uPoint &c){
    uPoint ac{0,0};
    ac.x = a.x - c.x;
    ac.y = a.y - c.y;
    uPoint bc{0,0};
    bc.x = b.x - c.x;
    bc.y = b.y - c.y;

    float m = ac.x*bc.x + ac.y*bc.y;
    float n = std::sqrt( ac.x * ac.x + ac.y * ac.y )* std::sqrt( bc.x * bc.x + bc.y * bc.y ) + 1e-3;
    float cos_abc = std::acos(m/n)*180/PI;
    //[-180,180]
    if( cos_abc > 180) cos_abc = cos_abc - 360;
    if( cos_abc < -180) cos_abc = cos_abc + 360;
    //[0,180]
    // std::cout << "angle = " << cos_abc << std::endl;
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


/*******************************************************************************
行人摔倒级联式检测
*******************************************************************************/

RET_CODE PedFallingDetection::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> PedFallingDetection::init";
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }

    std::string detect_modelpath ,skeleton_modelpath;
    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end()) {
            // std::cout << modelpath.size() << endl;
            for(auto param: modelpath){
                LOGI << param.first << "," << param.second;
            }
            return RET_CODE::ERR_INIT_PARAM_FAILED;
        }

    detect_modelpath = modelpath[InitParam::BASE_MODEL];

    //ped detection
    std::map<ucloud::InitParam, std::string> modelconfig = {
        {ucloud::InitParam::BASE_MODEL, detect_modelpath}
    };
    RET_CODE ret = m_detectHandle->init(modelconfig);
    if(ret!=RET_CODE::SUCCESS) return ret;
    vector<CLS_TYPE> detector_cls = {m_cls};
    ret = m_detectHandle->set_output_cls_order(detector_cls);
    //skeleton model
    if(modelpath.find(InitParam::SUB_MODEL)!=modelpath.end()){
        skeleton_modelpath = modelpath[InitParam::SUB_MODEL];
        ret = m_skeletonHandle->init(skeleton_modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
        std::vector<CLS_TYPE> sk_valid_cls = {m_cls};
        m_skeletonHandle->set_output_cls_order(sk_valid_cls);
    } else m_skeletonHandle = nullptr;
    
    return RET_CODE::SUCCESS;
}


RET_CODE PedFallingDetection::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> PedFallingDetection::run";
    if(tvimage.format!=TVAI_IMAGE_FORMAT_NV21 && tvimage.format!=TVAI_IMAGE_FORMAT_NV12 ) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    VecObjBBox det_bboxes;
    RET_CODE ret = m_detectHandle->run(tvimage, det_bboxes, threshold, nms_threshold);
    if(ret!=RET_CODE::SUCCESS) return ret;
    LOGI << "ped fall detected: " << det_bboxes.size();
    // std::cout << "ped fall detected: " << det_bboxes.size(); << std::endl;
    // Filter
    VecObjBBox cand_bboxes;
    for(auto &&box: det_bboxes){
        if(box.objtype == m_cls){
            cand_bboxes.push_back(box);
        }
    }
    
    if( m_skeletonHandle!=nullptr){
        //SK detector 自带过滤
        ret = m_skeletonHandle->run(tvimage, cand_bboxes);
        //判断是否符合摔倒规则
        filter_valid_pose(cand_bboxes, bboxes);
    } else {
        bboxes = cand_bboxes;
    }
    return RET_CODE::SUCCESS;
}


RET_CODE PedFallingDetection::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(m_cls);
    return RET_CODE::SUCCESS;
}


void PedFallingDetection::filter_valid_pose(VecObjBBox &bboxes_in, VecObjBBox &bboxes_out){
    for(auto &&box: bboxes_in){
        uPoint head = get_head_center(box);
        uPoint body = get_upper_body_center(box);
        uPoint cam = body;
        cam.y = body.y - 20;
        float angle = calc_angle(head, cam, body);
        if( angle > m_threshold_angle_of_body ){
            //摔倒规则通过
            box.objtype = m_cls;
            box.Pts.pts = {head, box.Pts.pts[5], box.Pts.pts[6]};
            bboxes_out.push_back(box);
        }
    #ifndef MLU220 //只有MLU270的情况下,才返回行人数据供分析
        else {
            box.objtype = CLS_TYPE::PEDESTRIAN;
            bboxes_out.push_back(box);
        }
    #endif        
    }

}


/*******************************************************************************
行人弯腰级联式检测
*******************************************************************************/

RET_CODE PedSkeletonDetection::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> PedSkeletonDetection::init";
    if(use_auto_model){
        modelpath.clear();
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }

    std::string detect_modelpath ,skeleton_modelpath;
    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end()) {
            std::cout << modelpath.size() << endl;
            for(auto param: modelpath){
                LOGI << param.first << "," << param.second;
            }
            return RET_CODE::ERR_INIT_PARAM_FAILED;
    }

    detect_modelpath = modelpath[InitParam::BASE_MODEL];

    algoTRule.init(30,4);

    //ped detection
    std::map<ucloud::InitParam, std::string> modelconfig = {
        {ucloud::InitParam::BASE_MODEL, detect_modelpath}
    };
    RET_CODE ret = m_detectHandle->init(modelconfig);
    if(ret!=RET_CODE::SUCCESS) return ret;
    vector<CLS_TYPE> detector_cls = {CLS_TYPE::PEDESTRIAN};
    ret = m_detectHandle->set_output_cls_order(detector_cls);
    //skeleton model
    if(modelpath.find(InitParam::SUB_MODEL)!=modelpath.end()){
        skeleton_modelpath = modelpath[InitParam::SUB_MODEL];
        ret = m_skeletonHandle->init(skeleton_modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
        std::vector<CLS_TYPE> sk_valid_cls = {CLS_TYPE::PEDESTRIAN};
        m_skeletonHandle->set_output_cls_order(sk_valid_cls);
    } else m_skeletonHandle = nullptr;
    
    return RET_CODE::SUCCESS;
}


RET_CODE PedSkeletonDetection::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> PedSkeletonDetection::run";
    if(tvimage.format!=TVAI_IMAGE_FORMAT_NV21 && tvimage.format!=TVAI_IMAGE_FORMAT_NV12 ) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    VecObjBBox det_bboxes;
    RET_CODE ret = m_detectHandle->run(tvimage, det_bboxes, threshold, nms_threshold);
    if(ret!=RET_CODE::SUCCESS) return ret;
    LOGI << "ped detected: " << det_bboxes.size();
    // std::cout << "ped fall detected: " << det_bboxes.size(); << std::endl;
    // Filter
    VecObjBBox cand_bboxes;
    for(auto &&box: det_bboxes){
        if(box.objtype == CLS_TYPE::PEDESTRIAN){
            //规则:不能在画面边缘,画面边缘容易产生问题
            if(is_valid_position(tvimage,box))
                cand_bboxes.push_back(box);
        }
    }
    
    if( m_skeletonHandle!=nullptr){
        //SK detector 自带过滤
        ret = m_skeletonHandle->run(tvimage, cand_bboxes);
        //判断是否符合弯腰规则
        filter_valid_pose(cand_bboxes, bboxes);
        //增加一次判断
    } else {
        bboxes = cand_bboxes;
    }

    if(use_post_rule){
        VecObjBBox tmpBBoxes;
        bboxes.swap(tmpBBoxes);
        algoTRule.push_back(tmpBBoxes, bboxes);
    }

    return RET_CODE::SUCCESS;
}


RET_CODE PedSkeletonDetection::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss = m_cls;
    return RET_CODE::SUCCESS;
}

void PedSkeletonDetection::filter_valid_pose(VecObjBBox &bboxes_in, VecObjBBox &bboxes_out){
    for(auto &&box: bboxes_in){
        float hw_ratio = ((float)box.rect.height) / box.rect.width;
        if( hw_ratio > 2) continue;
        uPoint head = get_head_center(box);
        uPoint body = get_upper_body_center(box);
        uPoint cam = body;
        cam.y = body.y - 20;
        float angle = calc_angle(head, cam, body);
        if( angle > m_threshold_angle_of_body ){
            //规则通过
            box.objtype = CLS_TYPE::PEDESTRIAN_BEND;
            box.Pts.pts = {head, body, box.Pts.pts[5], box.Pts.pts[6] };
            bboxes_out.push_back(box);
        }
    // #ifndef MLU220 //只有MLU270的情况下,才返回行人数据供分析
    //     else if(!use_post_rule) {
    //         box.objtype = CLS_TYPE::PEDESTRIAN;
    //         bboxes_out.push_back(box);
    //     }
    // #endif
    }
}

bool PedSkeletonDetection::is_valid_position(TvaiImage &tvimage, BBox &boxIn){
    int H = tvimage.height;
    int W = tvimage.width;
    float ratio = 0.03;
    int minW = ratio*W;
    int maxW = W - minW;
    int minH = ratio*H;
    int maxH = H - minH;

    int x0 = boxIn.rect.x;
    int y0 = boxIn.rect.y;
    int x1 = boxIn.rect.x + boxIn.rect.width;
    int y1 = boxIn.rect.y + boxIn.rect.height;

    if( x0 < minW || x0 > maxW) return false;
    if( x1 < minW || x1 > maxW) return false;
    if( y0 < minH || y0 > maxH) return false;
    if( y1 < minH || y1 > maxH) return false;
    return true;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
// 算法内后处理单元, 设置rule based trigger条件
////////////////////////////////////////////////////////////////////////////////////////////////////////
TargetBBox::TargetBBox(BBox &box, int life_time){
    rect.x = box.rect.x;
    rect.y = box.rect.y;
    rect.width = box.rect.width;
    rect.height = box.rect.height;
    score = box.confidence;
    trackid = box.track_id;
    current_life_time = life_time;
}

void AlgoTriggerRule::push_back(VecObjBBox &bboxesIN, VecObjBBox &bboxesOUT){
    for(auto &&boxIN: bboxesIN){
        if(boxIN.track_id < 0) continue;
        TargetBBox tbox(boxIN, def_life_time);
        buckets[tbox.trackid].push_back(tbox);
        //启动规则校验 reason:只有新增的会触发校验, 因为旧的在上一次处理过了
        if(rule(buckets[tbox.trackid], def_thresh_hits)) bboxesOUT.push_back(boxIN);
    }
    this->decrease_time();
}

void AlgoTriggerRule::decrease_time(){
    std::vector<int> keys_to_del;
    for(auto &&bucket: buckets){
        for(auto &&box: bucket.second){
            box.decrease_time();
        }
        if(bucket.second[0].current_life_time < 0){
            bucket.second.erase(bucket.second.begin());
        }
        if(bucket.second.empty()) keys_to_del.push_back(bucket.first);
    }
    for(auto ky: keys_to_del){
        buckets.erase(ky);
    }
}

bool AlgoTriggerRule::rule(std::vector<TargetBBox> &tboxes, int thresh_hits){
    if(tboxes.size()>=thresh_hits) return true;
    else return false;
}

