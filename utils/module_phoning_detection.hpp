#ifndef _MODULE_PHONING_DETECTION_HPP_
#define _MODULE_PHONING_DETECTION_HPP_
#include "module_base.hpp"
#include "module_binary_classification.hpp"
#include "module_yolo_detection_v2.hpp"
#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 抽烟级联式检测
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

class PhoningDetection: public AlgoAPI{
public:
    PhoningDetection(){
        m_ped_detectHandle = std::make_shared<YoloDetectionV4ByteTrack>();
        m_classifyHandle = std::make_shared<PhoningClassification>();
    };
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~PhoningDetection(){};
    /**
     * PARAM:
     *  threshold: 打电话分类模型的概率
     */
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);

private:
    void transform_box_to_ped_box(VecObjBBox &in_boxes, VecPedBox &out_boxes);

    float m_ped_threshold = 0.6;

    std::shared_ptr<AlgoAPI> m_ped_detectHandle = nullptr;
    std::shared_ptr<PhoningClassification> m_classifyHandle = nullptr;

    CLS_TYPE m_cls = CLS_TYPE::PHONING;

public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "yolov5s-conv-9"},
        {InitParam::SUB_MODEL, "phoning"},
    };
    bool use_auto_model = false;   
};



#endif