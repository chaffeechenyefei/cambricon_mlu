/**
 * 套个壳, 方便各种检测网络叠加不同跟踪器
*/
#ifndef _FRAMEWORK_DETECTION_HPP_
#define _FRAMEWORK_DETECTION_HPP_

#include "module_track.hpp"
#include <mutex>
#include "basic.hpp"
#include "module_obj_feature_extraction.hpp"

/*******************************************************************************
目录
*******************************************************************************/
class AnyDetectionV4DeepSort;//任意检测模型+DeepSort
class AnyDetectionV4ByteTrack;//任意检测模型+ByteTrack
class PipelineNaive;//任意模型管道式组合

// typedef struct tagAlgoNode AlgoNode;


/*******************************************************************************
AnyDetection + DeepSort
chaffee.chen@2022-09-30
*******************************************************************************/
class AnyDetectionV4DeepSort:public AlgoAPI{
public:
    AnyDetectionV4DeepSort();
    virtual RET_CODE init(std::map<InitParam, std::string> &modelpath);
    virtual RET_CODE init(std::map<InitParam, WeightData> &modelpath);
    virtual RET_CODE init(const std::string &modelpath);
    ~AnyDetectionV4DeepSort(){};
    virtual RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    virtual RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
    virtual RET_CODE set_output_cls_order(std::vector<CLS_TYPE> &output_clss);

    /** -----------------non AlgoAPI-------------------**/
    virtual RET_CODE set_detector(AlgoAPI* ptr);

protected:
    float clip_threshold(float x);
    float clip_nms_threshold(float x);
    AlgoAPISPtr m_detector = nullptr;
    std::shared_ptr<TrackPoolAPI<DEEPSORTPARM>> m_trackor = nullptr;
    std::shared_ptr<ObjFeatureExtractionV2> m_trackFeatExtractor = nullptr;

    float m_default_threshold = 0.55;
    float m_default_nms_threshold = 0.6;

#ifdef MLU220
    int m_fps = 5;
    int m_nn_buf = 20;
    float m_max_cosine_dist = 0.5;//0.2 for fps=25
#else
#ifdef SIM_MLU220
    int m_fps = 5;
    int m_nn_buf = 20;
    float m_max_cosine_dist = 0.5;//0.2 for fps=25
#else
    int m_fps = 25;//25
    int m_nn_buf = 30;
    float m_max_cosine_dist = 0.2;//0.2 for fps=25
#endif
#endif

public://yolo检测系列, 开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "yolov5s-conv-9"},
        {InitParam::TRACK_MODEL, "feature_extract_4c4b"},
    };        
    bool use_auto_model = false;
};

/*******************************************************************************
AnyDetection + ByteTrack
use set_trackor to switch differenct version of ByteTrack
chaffee.chen@2022-09-30
*******************************************************************************/
class AnyDetectionV4ByteTrack:public AlgoAPI{
public:
    AnyDetectionV4ByteTrack();
    virtual RET_CODE init(std::map<InitParam, std::string> &modelpath);
    virtual RET_CODE init(std::map<InitParam, WeightData> &modelpath);
    virtual RET_CODE init(const std::string &modelpath);
    virtual ~AnyDetectionV4ByteTrack(){};
    virtual RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    virtual RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
    virtual RET_CODE set_output_cls_order(std::vector<CLS_TYPE> &output_clss);

    /** -----------------non AlgoAPI-------------------**/
    virtual RET_CODE set_trackor(TRACKMETHOD trackmethod);
    virtual RET_CODE set_detector(AlgoAPI* ptr);

protected:
    float clip_threshold(float x);
    float clip_nms_threshold(float x);
    AlgoAPISPtr m_detector = nullptr;
    std::shared_ptr<TrackPoolAPI<BYTETRACKPARM>> m_trackor = nullptr;

    float m_default_threshold = 0.55;
    float m_default_nms_threshold = 0.6;

#ifdef MLU220
    int m_fps = 5;
    int m_nn_buf = 20;
#else
#ifdef SIM_MLU220
    int m_fps = 5;
    int m_nn_buf = 20;
#else
    int m_fps = 25;//25
    int m_nn_buf = 30;
#endif
#endif

public://yolo检测系列, 开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "yolov5s-conv-9"},
    };        
    bool use_auto_model = false;
};


/*******************************************************************************
PipelineNaive
chaffee.chen@2022-10-09
*******************************************************************************/
class PipelineNaive: public AlgoAPI{
public:
    PipelineNaive(){}
    virtual ~PipelineNaive(){}

    virtual RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    virtual RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);

    /**push_back 
     * handles, threshold, nms_threshold and 是否使用run传入的阈值参数 fixed_threshold=true表示不使用
     * 需要放入成熟的AlgoAPISPtr
     * **/
    virtual RET_CODE push_back(AlgoAPISPtr apihandle, bool fixed_threshold=true, float threshold=0.55, float nms_threshold=0.6){
        m_handles.push_back(apihandle);
        m_thresholds.push_back(threshold);
        m_nms_thresholds.push_back(nms_threshold);
        if(!fixed_threshold) unfixed_thresholds_index = m_handles.size()-1;
    }

protected:
    std::vector<AlgoAPISPtr> m_handles;
    std::vector<float> m_thresholds;
    std::vector<float> m_nms_thresholds;
    int unfixed_thresholds_index = -1;

};

// /*******************************************************************************
// GraphNodeNaive
// chaffee.chen@2022-10-25
// *******************************************************************************/

// typedef struct tagAlgoNode{
//     //preprocess null
//     //main body
//     AlgoAPISPtr algo{nullptr};
//     //post process null
//     tagAlgoNode** next_nodes{nullptr};
//     int num_of_next_nodes{0};
// };

// class GraphNodeNaive: public AlgoAPI{
// public:
//     GraphNodeNaive(){}
//     ~GraphNodeNaive(){}


// protected:

// };

#endif