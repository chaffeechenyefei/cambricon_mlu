#ifndef _MODULE_FALLING_DETECTION_HPP_
#define _MODULE_FALLING_DETECTION_HPP_
#include "module_base.hpp"
#include "module_yolo_detection_v2.hpp"
#include "module_skeleton_detection.hpp"
#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 行人摔倒级联式检测
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

class PedFallingDetection: public AlgoAPI{
public:
    PedFallingDetection(){//直接在头文件中显示使用的方法, 清晰一点
        m_detectHandle = std::make_shared<YoloDetectionV4ByteTrack>();
        m_skeletonHandle = std::make_shared<SkeletonDetectorV4>();
    }
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~PedFallingDetection(){};
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);

private:
    //过滤出符合的体态
    void filter_valid_pose(VecObjBBox &bboxes_in, VecObjBBox &bboxes_out);

    const float m_threshold_angle_of_body = 30;

    AlgoAPISPtr m_detectHandle = nullptr;
    AlgoAPISPtr m_skeletonHandle = nullptr;

    CLS_TYPE m_cls = CLS_TYPE::PEDESTRIAN_FALL;

public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "yolov5s-conv-fall-ped"},
        {InitParam::SUB_MODEL, "posenet"},
    };    
    bool use_auto_model = false;   
};

////////////////////////////////////////////////////////////////////////////////////////////////////////
// 行人弯腰级联式检测
////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 算法内后处理单元, 设置rule based trigger条件
////////////////////////////////////////////////////////////////////////////////////////////////////////
class TargetBBox{
public:
    TargetBBox(){}
    ~TargetBBox(){}
    TargetBBox(BBox &box, int life_time=5);

    void decrease_time(){current_life_time--;}

    int current_life_time = 0;
    int num_iou_overlap = 0;
    cv::Rect rect = cv::Rect(0,0,0,0);
    float score = 0;
    int trackid = -1;
};

/**
 * 触发机制:
 * 一段时间内(def_life_time),触发超过def_thresh_hits次
 */
class AlgoTriggerRule{
public:
    AlgoTriggerRule(){}    
    AlgoTriggerRule(int life_time, int thresh_hits):\
        def_life_time(life_time), \
        def_thresh_hits(thresh_hits){}
    void init(int life_time, int thresh_hits){
        def_life_time = life_time;
        def_thresh_hits = thresh_hits;
    }
    ~AlgoTriggerRule(){}

    void push_back(VecObjBBox &bboxesIN, VecObjBBox &bboxesOUT);
    void decrease_time();

    bool rule(std::vector<TargetBBox> &tboxes, int thresh_hits = 2);

    std::map<int, std::vector<TargetBBox>> buckets;

    int def_life_time = 5;
    int def_thresh_hits = 4;

};

class PedSkeletonDetection: public AlgoAPI{
public:
     PedSkeletonDetection(){
         m_detectHandle = std::make_shared<YoloDetectionV4ByteTrack>();
         m_skeletonHandle = std::make_shared<SkeletonDetectorV4>();
     }
    //兼容老接口, 不进行骨架检测
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~PedSkeletonDetection(){};
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);

private:
    //过滤出符合的体态
    void filter_valid_pose(VecObjBBox &bboxes_in, VecObjBBox &bboxes_out);
    //判断检测框是否在边缘,边缘的直接过滤
    bool is_valid_position(TvaiImage &tvimage, BBox &boxIn);
    //后处理规则, 暂时不使用, 根据业务情况考虑是否采用
    bool use_post_rule = false;
    AlgoTriggerRule algoTRule;

    const float m_threshold_angle_of_body = 50;

    std::shared_ptr<AlgoAPI> m_detectHandle = nullptr;
    std::shared_ptr<AlgoAPI> m_skeletonHandle = nullptr;

    std::vector<CLS_TYPE> m_cls = { CLS_TYPE::PEDESTRIAN_BEND, CLS_TYPE::PEDESTRIAN };

public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "yolov5s-conv-people"},
        {InitParam::SUB_MODEL, "posenet"},
    };
    bool use_auto_model = false;
};





#endif