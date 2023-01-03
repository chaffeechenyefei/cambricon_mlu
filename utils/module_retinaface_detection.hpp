#ifndef _MODULE_RETINAFACE_DETECTION_HPP_
#define _MODULE_RETINAFACE_DETECTION_HPP_
#include "module_base.hpp"
#include "module_obj_feature_extraction.hpp"
#include "module_face_feature_extraction.hpp"
#include "module_iqa.hpp"
#include <easytrack/easy_track.h>

#include "framework_detection.hpp"
#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
//人脸检测
////////////////////////////////////////////////////////////////////////////////////////////////////////
// class FaceDetectionV2;
class FaceDetectionV4;//对应YoloDetectionV4, 使用BaseModelV2 不含跟踪
class FaceDetectionV4DeepSort;
class FaceDetectionV4ByteTrack;

using namespace ucloud;

/*******************************************************************************
FaceDetectionV4 使用BaseModelV2 不含跟踪
*******************************************************************************/
class FaceDetectionV4: public AlgoAPI{
public:
    FaceDetectionV4();
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~FaceDetectionV4(){};
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.85, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
    /***non AlgoAPI function***/
    static float get_box_expand_ratio();
private:
    RET_CODE postprocess(float* model_output, VecObjBBox &bboxes, float threshold, float nms_threshold ,float expand_ratio, float aspect_ratio, int imgW, int imgH);
    float clip_threshold(float x);
    float clip_nms_threshold(float x);
    //Quality相关
    /**
     * 输出的quality结果在bboxes中进行修改
     */
    RET_CODE iqa_quality(TvaiImage &tvimage, VecObjBBox &bboxes);

    //通用变量
    float m_default_threshold = 0.55;
    float m_default_nms_threshold = 0.6;

    static constexpr float _expand_ratio = 1.3;//返回的人脸检测框扩大比例

    CLS_TYPE _cls_ = CLS_TYPE::FACE;

    AlgoAPISPtr m_faceAttrExtractor = nullptr;//人脸属性分类网络
    MLUNet_Ptr m_net = nullptr;//人脸检测主干网络
    IQA_Face_Evaluator iqa_evaluator;//人脸质量评估

    Timer m_Tk;

public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "retinaface"},
        {InitParam::SUB_MODEL, "attribution"},
    };  
    bool use_auto_model = false;   
};

/*******************************************************************************
FaceDetectionV4DeepSort
*******************************************************************************/
class FaceDetectionV4DeepSort: public AnyDetectionV4DeepSort{
public:
    FaceDetectionV4DeepSort(){
        m_detector = std::make_shared<FaceDetectionV4>();
    }
    virtual ~FaceDetectionV4DeepSort(){};

};

class FaceDetectionV4ByteTrack: public AnyDetectionV4ByteTrack{
public:
    FaceDetectionV4ByteTrack(){
        m_detector = std::make_shared<FaceDetectionV4>();
    }
    virtual ~FaceDetectionV4ByteTrack(){}
};

// /*******************************************************************************
// FaceDetectionV2原始人脸检测, 继承BaseModel
// *******************************************************************************/
// class FaceDetectionV2: public BaseModel{
// public:
//     FaceDetectionV2(){};
//     RET_CODE init(const std::string &modelpath);
//     RET_CODE init(const std::string &modelpath, const std::string &trackmodelpath);
//     /**
//      * 20211117
//      * 新接口形式
//      */
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     ~FaceDetectionV2();
//     /**
//      * @IN:
//      *  tvimage: BGR/YUV_NV21 format data
//      * @OUT:
//      *  bboxes: bounding box
//      * @DESC:
//      *  Support model: firstconv(input channel=4, uint8) only.
//      *  When NV21 is input, resize and crop ops are done on mlu.
//      *  Postprocess id done on cpu. Will be moved to mlu.
//      **/
//     RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes);
//     // RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
//     RET_CODE set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
//     static float get_box_expand_ratio();
// private:
//     /**
//      * @param:
//      * input_bboxes: run函数返回的检测框
//      * output_bboxes: 过滤后的检测框(框的大小过滤)
//      **/
//     void object_filter(VecObjBBox &input_bboxes, VecObjBBox &output_bboxes, int imgW, int imgH);
//     RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     // RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE postprocess(float* model_output, VecObjBBox &bboxes, float expand_ratio, float aspect_ratio, int imgW, int imgH);
//     //Quality相关
//     /**
//      * 输出的quality结果在bboxes中进行修改
//      */
//     RET_CODE iqa_quality(TvaiImage &tvimage, VecObjBBox &bboxes);
//     //Tracking相关
//     /**
//      * trackprocess
//      * default 版本的跟踪, 基于featureMatch. cpu模式下的匹配.
//      */
//     RET_CODE trackprocess(TvaiImage &tvimage, VecObjBBox &bboxes_in);
//     /**
//      * init_trackor
//      * 初始化所有跟踪相关的模块和变量
//      */
//     RET_CODE init_trackor(const std::string &trackmodelpath);
//     RET_CODE create_trackor(int uuid_cam=-1);
    
//     std::shared_ptr<ObjFeatureExtraction> m_trackFeatExtractor = nullptr;
//     AlgoAPISPtr m_faceAttrExtractor = nullptr;
//     // std::unique_ptr<edk::EasyTrack> m_Trackor = nullptr;
//     std::map<int,std::shared_ptr<edk::EasyTrack>> m_Trackors;

//     //通用变量
//     float _threshold = 0.8;
//     float _nms_threshold = 0.2;
//     static constexpr float _expand_ratio = 1.3;//返回的人脸检测框扩大比例
//     TvaiResolution _minTargeSize{0,0};
//     TvaiResolution _maxTargeSize{0,0};
//     std::vector<TvaiRect> _pAoiRect;

//     CLS_TYPE _cls_ = CLS_TYPE::FACE;
//     IQA_Face_Evaluator iqa_evaluator;

// private://tracking param
// #ifdef MLU220
//     int m_fps = 4;
//     float m_max_cosine_distance = 0.5;//0.2 for fps=25
// #else
// #ifdef SIM_MLU220
//     int m_fps = 4;
//     float m_max_cosine_distance = 0.5;//0.2 for fps=25
// #else
//     int m_fps = 25;
//     float m_max_cosine_distance = 0.2;
// #endif
// #endif
//     int m_nn_budget = 25;
//     float m_max_iou_distance = 0.5;
//     int m_n_init = 2;

// public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
//     std::vector<std::string> m_roots = {"/cambricon/model"};
//     std::map<ucloud::InitParam, std::string> m_models_startswith = {
//         {InitParam::BASE_MODEL, "retinaface"},
//         {InitParam::TRACK_MODEL, "feature_extract_4c4b"},
//     };  
//     bool use_auto_model = false;   
// };




#endif

/**
 * Problem:
 *  const static float _expand_ratio = 1.3
 *  error: ‘constexpr’ needed for in-class initialization of static data member ‘tolerance’ of non-integral type
 *  https://stackoverflow.com/questions/9141950/initializing-const-member-within-class-declaration-in-c 
 * 
 * only int memeber can be assigned as const static, float should use constexpr:
 * static constexpr float _expand_ratio = 1.3
 */

