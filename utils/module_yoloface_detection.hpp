#ifndef _MODULE_YOLOFACE_DETECTION_HPP_
#define _MODULE_YOLOFACE_DETECTION_HPP_
#include "module_base.hpp"

#include "framework_detection.hpp"
#include <mutex>
#include <string>
////////////////////////////////////////////////////////////////////////////////////////////////////////
//通用检测
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

class YoloFaceDetection;//标准yoloface检测
class YoloFaceDetectionV4ByteTrack;//标准yoloface检测带跟踪
class LicplateRecognition;//车牌识别

/*******************************************************************************
YoloFaceDetectionV4 纯检测, 无跟踪, 输出关键点坐标
shawn.qian@2022-09-20
chaffee.chen@2022-10-08
*******************************************************************************/
class YoloFaceDetection: public AlgoAPI{
public:
    YoloFaceDetection():m_dimOffset(15){
        m_net = std::make_shared<BaseModelV2>();
    }
    YoloFaceDetection(int dimOffset):m_dimOffset(dimOffset){
        m_net = std::make_shared<BaseModelV2>();
    }
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~YoloFaceDetection();
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
    RET_CODE set_output_cls_order(std::vector<CLS_TYPE> &output_clss);
    /*-------non AlgoAPI---------*/
    RET_CODE run(TvaiImage &tvimage, TvaiRect tvrect, VecObjBBox &bboxes, float threshold, float nms_threshold);

private:
    RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE postprocess(float* model_output, VecObjBBox &bboxes, float threshold, float nms_threshold, float expand_ratio, float aspect_ratio, int imgW, int imgH);
    /*-------support non AlgoAPI---------*/
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, TvaiRect tvrect ,VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE postprocess(float* model_output, VecObjBBox &bboxes, TvaiRect tvrect, float threshold, float nms_threshold, float expand_ratio, float aspect_ratio, int imgW, int imgH);

    float clip_threshold(float x);
    float clip_nms_threshold(float x);

protected:
    MLUNet_Ptr m_net = nullptr;

    std::vector<CLS_TYPE> m_output_cls_order;
    std::map<CLS_TYPE,int> m_unique_cls_order;
    int _unique_cls_num = 0;
    int _output_cls_num = 0;

    //当传入的参数超过边界时,采用默认数值
    float m_default_threshold = 0.55;
    float m_default_nms_threshold = 0.6;

    const int m_dimOffset;


public://yolo检测系列, 开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "yolov5s-face-licplate"},
        // {InitParam::SUB_MODEL, "licplate-recog"},
    };        
    bool use_auto_model = false;
};



////////////////////////////////////////////////////////////////////////////////////////////////////////
//车牌识别
////////////////////////////////////////////////////////////////////////////////////////////////////////
/*******************************************************************************
LicplateRecognition 车牌识别, 取消BaseModel的继承
shawn.qian@2022-09-20
chaffee.chen@2022-10-09
*******************************************************************************/
class LicplateRecognition: public AlgoAPI{
public:
    LicplateRecognition(){
        m_net = std::make_shared<BaseModelV2>();
    }
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~LicplateRecognition();
    RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
protected:
    RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes);
    RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
    RET_CODE postprocess(float* model_output, BBox &bbox);

    CLS_TYPE _cls_ = CLS_TYPE::LICPLATE;

    static constexpr float _expand_ratio = 1.0f;//AAAATTTT

private:
    MLUNet_Ptr m_net = nullptr;

public:
    static const std::vector<std::string> LICPLATE_CHARS;

public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
        std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "licplate-recog"},
    };
    bool use_auto_model = false;    

}; 

/*******************************************************************************
YoloFaceDetectionV4ByteTrack 带跟踪
shawn.qian@2022-09-20
chaffee.chen@2022-10-08
*******************************************************************************/
class YoloFaceDetectionV4ByteTrack:public AnyDetectionV4ByteTrack{
public:
    YoloFaceDetectionV4ByteTrack(){
        m_detector = std::make_shared<YoloFaceDetection>();
    }
    virtual ~YoloFaceDetectionV4ByteTrack(){}
};

/*******************************************************************************
LicplateDetRec 纯检测, 无跟踪, 输出关键点坐标
chaffee.chen@2022-10-09
*******************************************************************************/
class LicplateDetRec:public PipelineNaive{
public:
    LicplateDetRec(){
        PipelineNaive::push_back(std::make_shared<YoloFaceDetectionV4ByteTrack>(), false);//采用动态阈值
        PipelineNaive::push_back(std::make_shared<LicplateRecognition>());//不需要阈值
        std::vector<CLS_TYPE> det_cls = {CLS_TYPE::LICPLATE_BLUE, CLS_TYPE::LICPLATE_SGREEN, CLS_TYPE::LICPLATE_BGREEN, CLS_TYPE::LICPLATE_YELLOW};
        m_handles[0]->set_output_cls_order(det_cls);
    }
    ~LicplateDetRec(){}
    RET_CODE init(std::map<InitParam, std::string> &modelpath){
        RET_CODE ret = RET_CODE::SUCCESS;
        ret = m_handles[0]->init(modelpath[InitParam::BASE_MODEL]);
        if(ret!=RET_CODE::SUCCESS) return ret;
        ret = m_handles[1]->init(modelpath[InitParam::SUB_MODEL]);
        if(ret!=RET_CODE::SUCCESS) return ret;
        return ret;
    }
};

#endif
