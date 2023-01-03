//chaffee.chen@2022-09-30
#ifndef _MODULE_YOLO_DETECTION_V2_HPP_
#define _MODULE_YOLO_DETECTION_V2_HPP_

#include <mutex>
#include "basic.hpp"
#include "module_obj_feature_extraction.hpp"
#include "framework_detection.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////
//通用检测
////////////////////////////////////////////////////////////////////////////////////////////////////////
class YoloDetectionV4;
class YoloDetectionV4DeepSort;
class YoloDetectionV4ByteTrack;

using namespace ucloud;

/*******************************************************************************
YoloDetectionV4
只有检测, 没有跟踪, 使用BaseModelV2
chaffee.chen@2022-09-30
*******************************************************************************/
class YoloDetectionV4: public AlgoAPI{
public:
    YoloDetectionV4();
    /**
     * 20211117
     * 新接口形式
     */
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    RET_CODE init(std::map<InitParam, WeightData> &weightConfig);
    
    RET_CODE init(const std::string &modelpath);
    ~YoloDetectionV4();
    /**
     * @IN:
     *  tvimage: BGR/YUV_NV21 format data
     * @OUT:
     *  bboxes: bounding box
     * @DESC:
     *  Support model: firstconv(input channel=4, uint8) only.
     *  When NV21 is input, resize and crop ops are done on mlu.
     *  Postprocess id done on cpu. Will be moved to mlu.
     **/
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE run(TvaiImage &tvimage, TvaiRect tvrect, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);

    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
    //@overload
    RET_CODE set_output_cls_order(std::vector<CLS_TYPE> &output_clss);
    RET_CODE set_output_cls_order(std::vector<std::string> &output_clss);

protected:
    float clip_threshold(float x);
    float clip_nms_threshold(float x);

    MLUNet_Ptr m_net = nullptr;
    const int m_dimOffset = 5;//xywh+objectness (5 elements), without nc



private:
    RET_CODE postprocess(float* model_output, VecObjBBox &bboxes, float threshold, float nms_threshold,
        float expand_ratio, float aspect_ratio, int imgW, int imgH);

    RET_CODE postprocess(float* model_output, VecObjBBox &bboxes, TvaiRect tvrect, float threshold, float nms_threshold, 
        float expand_ratio, float aspect_ratio, int imgW, int imgH);

    /**
     * init_trackor
     * 初始化所有跟踪相关的模块和变量
     */
    //当传入的参数超过边界时,采用默认数值
    float m_default_threshold = 0.55;
    float m_default_nms_threshold = 0.6;

    //二选一 A
    std::map<CLS_TYPE,int> _unique_cls_order;
    std::vector<CLS_TYPE> m_output_cls_order;
    //二选一 B
    std::map<std::string,int> _unique_cls_order_str;
    std::vector<std::string> m_output_cls_order_str;
    int _unique_cls_num = 0;
    int _output_cls_num = 0;

    Timer m_Tk;


public://yolo检测系列, 开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "yolov5s-conv-9"},
    };        
    bool use_auto_model = false;
};


/*******************************************************************************
YoloDetectionV4 + DeepSort
chaffee.chen@2022-09-30
*******************************************************************************/
class YoloDetectionV4DeepSort:public AnyDetectionV4DeepSort{
public:
    YoloDetectionV4DeepSort(){
        m_detector = std::make_shared<YoloDetectionV4>();
    }
    virtual ~YoloDetectionV4DeepSort(){}
};



/*******************************************************************************
YoloDetectionV4 + ByteTrack
use set_trackor to switch differenct version of ByteTrack
chaffee.chen@2022-09-30
*******************************************************************************/
class YoloDetectionV4ByteTrack:public AnyDetectionV4ByteTrack{
public:
    YoloDetectionV4ByteTrack(){
        m_detector = std::make_shared<YoloDetectionV4>();
    }
    virtual ~YoloDetectionV4ByteTrack(){}
};



/*******************************************************************************
YoloDetectionV4 + ByteTrack + POST_RULE_HOVER
use set_trackor to switch differenct version of ByteTrack
chaffee.chen@2022-09-30
*******************************************************************************/
#include "post_rule/post_rule_hover.hpp"
class YoloDetectionV4ByteTrack_POST_RULE_HOVER:public AnyDetectionV4ByteTrack{
public:
    YoloDetectionV4ByteTrack_POST_RULE_HOVER(){
        m_detector = std::make_shared<YoloDetectionV4>();
    }
    virtual RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6){
        RET_CODE ret = AnyDetectionV4ByteTrack::run(tvimage, bboxes, threshold, nms_threshold);
        if(ret!=RET_CODE::SUCCESS) return ret;
        ret = post_handle.run(tvimage, bboxes);
        return ret;
    }
    virtual ~YoloDetectionV4ByteTrack_POST_RULE_HOVER(){}

protected:
    POST_RULE_HOVER post_handle;

};


/*******************************************************************************
YoloDetectionV4DeepSort_POST_RULE_HOVER
chaffee.chen@2022-09-30
*******************************************************************************/
class YoloDetectionV4DeepSort_POST_RULE_HOVER:public AnyDetectionV4DeepSort{
public:
    YoloDetectionV4DeepSort_POST_RULE_HOVER(){
        m_detector = std::make_shared<YoloDetectionV4>();
    }
    virtual RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6){
        RET_CODE ret = AnyDetectionV4DeepSort::run(tvimage, bboxes, threshold, nms_threshold);
        if(ret!=RET_CODE::SUCCESS) return ret;
        ret = post_handle.run(tvimage, bboxes);
        return ret;
    }
    virtual ~YoloDetectionV4DeepSort_POST_RULE_HOVER(){}

protected:
    POST_RULE_HOVER post_handle;    
};



#endif