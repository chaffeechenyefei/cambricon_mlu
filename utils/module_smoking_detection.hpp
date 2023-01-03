#ifndef _MODULE_SMOKING_DETECTION_HPP_
#define _MODULE_SMOKING_DETECTION_HPP_
#include "module_base.hpp"
#include "module_binary_classification.hpp"
#include "module_yolo_detection_v2.hpp"
#include "module_retinaface_detection.hpp"
#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 抽烟级联式检测
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

class SmokingDetectionV2: public AlgoAPI{
public:
    SmokingDetectionV2(){
        m_face_detectHandle = std::make_shared<FaceDetectionV4ByteTrack>();
        m_cig_detectHandle = std::make_shared<YoloDetectionV4>();
    }
    ~SmokingDetectionV2(){}
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);

private:
    float m_face_threshold = 0.7;

    AlgoAPISPtr m_face_detectHandle = nullptr;
    std::shared_ptr<YoloDetectionV4> m_cig_detectHandle = nullptr;

    CLS_TYPE m_cls = CLS_TYPE::SMOKING;

public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "retinaface"},
        {InitParam::SUB_MODEL, "yolov5s-conv-cig"},
    };
    bool use_auto_model = false;   
};





// /**
//  * SmokingDetection
//  * 人体检测+人脸检测+手的检测(非必要)构成人脸+双手的图像输入, 通过分类器判定是否抽烟
//  */
// class SmokingDetection: public AlgoAPI{
// public:
//     SmokingDetection(){};
//     /**
//      * 20211117
//      * 新接口形式
//      */
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     ~SmokingDetection(){};
//     /**
//      * @IN:
//      *  tvimage: YUV_NV21 format data ONLY
//      * @OUT:
//      *  bboxes: bounding box
//      * @DESC:
//      *  Support model: firstconv(input channel=4, uint8) only.
//      *  When NV21 is input, resize and crop ops are done on mlu.
//      *  Postprocess id done on cpu. Will be moved to mlu.
//      **/
//     RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
//     RET_CODE set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);

// private:
//     float m_threshold = 0.4;
//     float m_ped_threshold = 0.5;
//     float m_face_threshold = 0.7;
//     float m_hand_threshold = 0.6;
//     TvaiResolution m_minTargeSize{0,0};
//     TvaiResolution m_maxTargeSize{0,0};
//     std::vector<TvaiRect> m_pAoiRect;

//     std::shared_ptr<AlgoAPI> m_ped_detectHandle;
//     std::shared_ptr<AlgoAPI> m_hand_detectHandle;
//     std::shared_ptr<AlgoAPI> m_face_detectHandle;
//     std::shared_ptr<SmokingClassification> m_classifyHandle;

//     CLS_TYPE m_cls = CLS_TYPE::SMOKING;
// };



#endif