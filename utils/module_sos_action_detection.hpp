#ifndef _MODULE_SOS_ACTION_DETECTION_HPP_
#define _MODULE_SOS_ACTION_DETECTION_HPP_
#include "module_base.hpp"
#include "module_yolo_detection_v2.hpp"
#include "module_skeleton_detection.hpp"
// #include "module_yolo_detection.hpp"
// #include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 举手求救检测的几种方式
// 1. 手在头上 (举手)
// 2. 手掌在头上, 且运动轨迹匹配
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

// /** 
//  * SOSDetectionV1
//  * @DESC: 人体检测+骨架检测, 检测到手在头上, 即认为求救
//  */
// class SOSDetectionV1: public AlgoAPI{
// public:
//     SOSDetectionV1(){}
//     ~SOSDetectionV1(){}
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
//     RET_CODE set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect);

// private:
//     float m_ped_threshold = 0.6;
//     bool is_pose_sos(BBox &box);

//     // TvaiResolution m_minTargeSize{0,0};
//     // TvaiResolution m_maxTargeSize{0,0};
//     // std::vector<TvaiRect> m_pAoiRect;

//     std::shared_ptr<AlgoAPI> m_ped_detector = nullptr;
//     std::shared_ptr<AlgoAPI> m_sk_detector = nullptr;

//     CLS_TYPE m_cls = CLS_TYPE::OTHERS_A;//求救
// };

typedef struct _SOSBox{
    BBox body;
    VecObjBBox hands_in;//在body框中的手
    VecObjBBox hands_out;//在body框外的手, 但是满足距离判定
}SOSBox;
typedef std::vector<SOSBox> VecSOSBox;

/** 
 * SOSDetectionV2
 * @DESC: 人体检测+手的检测, 判断手和身体位置关系
 */
class SOSDetectionV2: public AlgoAPI{
public:
    SOSDetectionV2(){
        m_ped_detector = std::make_shared<YoloDetectionV4ByteTrack>();
        m_hand_detector = std::make_shared<YoloDetectionV4>();
    }
    ~SOSDetectionV2(){}
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
    RET_CODE set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect);

protected:
    bool is_sos_trigger(SOSBox &sosbox, BBox &handbox);
    void merge(VecObjBBox& bodyboxIN, VecObjBBox& handboxIN, VecSOSBox &sosboxOUT, VecObjBBox &othersOUT);

private:
    float m_ped_threshold = 0.6;
    float m_hand_threshold = 0.78;

    std::shared_ptr<AlgoAPI> m_ped_detector = nullptr;
    std::shared_ptr<AlgoAPI> m_hand_detector = nullptr;

    CLS_TYPE m_cls = CLS_TYPE::OTHERS_A;//求救

public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "yolov5s-conv-9"},
        {InitParam::SUB_MODEL, "yolov5s-conv-hand"},
    };      
    bool use_auto_model = false;  
};



#endif