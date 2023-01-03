#include "module_cascade_detection.hpp"
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

#define NMS_UNION 0
#define NMS_MIN 1

#define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

// #include <future>
using namespace ucloud;
using namespace cv;
using std::vector;
using std::cout;
using std::endl;

RET_CODE CascadeDetection::init(std::map<InitParam, std::string> &modelpath){
    std::string detect_modelpath, classify_modelpath;
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }

    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end() || \
        modelpath.find(InitParam::SUB_MODEL)==modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    detect_modelpath = modelpath[InitParam::BASE_MODEL];
    classify_modelpath = modelpath[InitParam::SUB_MODEL];

    RET_CODE ret = this->init(detect_modelpath, classify_modelpath);
    return ret;
}



RET_CODE CascadeDetection::init(  const std::string &detect_modelpath,
                    const std::string &classify_modelpath){
    LOGI << "-> CascadeDetection::init_detect_and_binary_classification";
    //Detection model
    std::vector<CLS_TYPE> yolov5s_conv_cls = {m_cls};
    m_detectHandle->set_output_cls_order(yolov5s_conv_cls);
    RET_CODE ret = RET_CODE::FAILED;

    std::map<InitParam, std::string> modelconfig = {
        {InitParam::BASE_MODEL, detect_modelpath}
    };
    ret = m_detectHandle->init(modelconfig);

    if(ret!=RET_CODE::SUCCESS)
        return ret;

    vector<CLS_TYPE> filter_cls{m_cls};
    m_classifyHandle->set_filter_cls(filter_cls);
    m_classifyHandle->set_primary_output_cls(1, m_cls);
    m_classifyHandle->set_expand_ratio(2);
    ret = m_classifyHandle->init(classify_modelpath);
    return ret;
}

RET_CODE CascadeDetection::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    VecObjBBox _bboxes;
    ret = m_detectHandle->run(tvimage, _bboxes, threshold, nms_threshold);
    if(ret!=RET_CODE::SUCCESS) return ret;
#ifdef VERBOSE
    cout << "confidence from yolo detector" << endl;
    for(auto iter=_bboxes.begin(); iter != _bboxes.end(); iter++){
        if(iter->objtype!=m_cls) continue;
        else cout << iter->confidence << ", ";
    }
    cout << endl;
#endif
    VecObjBBox box4cls;
    for(auto &&box: _bboxes){
        float _trust_det_threshold = std::max(m_trust_det_threshold, threshold);
        if(box.confidence > _trust_det_threshold) bboxes.push_back(box);
        else box4cls.push_back(box);
    }
    ret = m_classifyHandle->run(tvimage, box4cls, m_cls_threshold);
    LOGI << "-> CascadeDetection::run [" << _bboxes.size() << "] executed";
    if(ret!=RET_CODE::SUCCESS) return ret;
    for(auto &&box: box4cls){
        if(box.objtype!=m_cls) continue;
        else bboxes.push_back(box);
    }
    return ret;
}

RET_CODE CascadeDetection::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(m_cls);
    return RET_CODE::SUCCESS;
}

// RET_CODE CascadeDetection::set_param(float threshold, float nms_threshold, 
//     TvaiResolution maxTargetSize, TvaiResolution minTargetSize, 
//     std::vector<TvaiRect> &pAoiRect){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(m_detectHandle!=nullptr)
//         ret = m_detectHandle->set_param(threshold, nms_threshold, maxTargetSize, minTargetSize, pAoiRect );
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     BinaryClassificationV4* ptrTmp = reinterpret_cast<BinaryClassificationV4*>(m_classifyHandle.get());
//     // ptrTmp->set_threshold(m_cls_threshold);
//     return ret;
// }