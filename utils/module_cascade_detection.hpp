#ifndef _MODULE_CASCADE_DETECTION_HPP_
#define _MODULE_CASCADE_DETECTION_HPP_
#include "module_base.hpp"
#include "module_yolo_detection_v2.hpp"
#include "module_binary_classification.hpp"

#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 级联式检测组件: yolo + binary
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

class CascadeDetection: public AlgoAPI{
public:
    CascadeDetection(){
        m_detectHandle = std::make_shared<YoloDetectionV4ByteTrack>();
        m_classifyHandle = std::make_shared<BinaryClassificationV4>();//BinaryClassificationV4
    };
    RET_CODE init(  const std::string &detect_modelpath, 
                    const std::string &classify_modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~CascadeDetection(){};
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);

    RET_CODE set_trust_threshold(float val){m_trust_det_threshold = val;}

private:

    std::shared_ptr<AlgoAPI> m_detectHandle = nullptr;
    std::shared_ptr<BinaryClassificationV4> m_classifyHandle = nullptr; 

    CLS_TYPE m_cls = CLS_TYPE::FIRE;
    float m_cls_threshold = 0.5;
    float m_trust_det_threshold = 0.7;//高于这个阈值的检测结果直接通过

public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "yolov5s-conv-fire"},
        {InitParam::SUB_MODEL, "resnet34fire"},
    };
    bool use_auto_model = false;        
};


#endif