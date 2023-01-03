#ifndef _MODULE_MOD_TRADITIONAL_HPP_
#define _MODULE_MOD_TRADITIONAL_HPP_
#include "module_base.hpp"
#include "module_nn_match.hpp"
#include "framework_detection.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <mutex>
#include <map>

using namespace ucloud;
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 传统方案的高空抛物: 仅支持yuv
////////////////////////////////////////////////////////////////////////////////////////////////////////
class BackgroundSegment;
class BackgroundSegmentV4;//只有前后景减除+外接矩形
class BackgroundSegmentV4ByteTrack;//测试使用ByteTrack做后处理的效果

/*******************************************************************************
BackgroundSegmentV4
chaffee.chen@2022-10-21
*******************************************************************************/
class BackgroundSegmentV4: public AlgoAPI {
public:
    BackgroundSegmentV4(){
        m_net = std::make_shared<YuvCropResizeModel>();
    }
    virtual ~BackgroundSegmentV4(){}
    RET_CODE init(std::map<InitParam, std::string> &modelpath){return init();}
    RET_CODE init(const std::string &modelpath){return init();}
    RET_CODE init();
    RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);

protected://member function
    RET_CODE create_model(int uuid_cam);
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE postprocess( cv::Mat &cropped_img, VecObjBBox &bboxes, float aspect_ratio ,int uuid_cam=-1);
    RET_CODE postfilter(TvaiImage &tvimage, VecObjBBox &ins, VecObjBBox &outs, float threshold);

protected://member value
    MLUResize_Ptr m_net = nullptr;
    int m_dstH = 416;
    int m_dstW = 736;
    std::vector<CLS_TYPE> _cls_ = {CLS_TYPE::FALLING_OBJ, CLS_TYPE::FALLING_OBJ_UNCERTAIN};
#ifdef OPENCV3
    std::map<int,cv::Ptr<cv::BackgroundSubtractor>> m_Models;
#else
    std::map<int,cv::shared_ptr<cv::BackgroundSubtractor>> m_Models;
#endif    
};

/*******************************************************************************
BackgroundSegmentV4ByteTrack
chaffee.chen@2022-10-21
*******************************************************************************/
class BackgroundSegmentV4ByteTrack: public AnyDetectionV4ByteTrack{
public:
    BackgroundSegmentV4ByteTrack(){
        m_detector = std::make_shared<BackgroundSegmentV4>();
    }
    virtual ~BackgroundSegmentV4ByteTrack(){}
};


/*******************************************************************************
BackgroundSegmentV
chaffee.chen@2021-xx-xx
*******************************************************************************/
class BackgroundSegment: public YuvCropResizeModel{
public:
    BackgroundSegment(){};
    RET_CODE init(const std::string &modelpath);
    RET_CODE init();
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~BackgroundSegment();
    /**
     * @IN:
     *  tvimage: YUV_NV21 format data
     * @OUT:
     *  bboxes: bounding box
     * @DESC:
     *  Support model: firstconv(input channel=4, uint8) only.
     *  When NV21 is input, resize and crop ops are done on mlu.
     *  Postprocess id done on cpu. Will be moved to mlu.
     **/
    RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
private:
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    //后处理单元, 不移交指针
    RET_CODE postprocess( cv::Mat &cropped_img, VecObjBBox &bboxes, float aspect_ratio ,int uuid_cam=-1);
    RET_CODE postfilter(TvaiImage &tvimage, VecObjBBox &ins, VecObjBBox &outs, float threshold);

    RET_CODE init_trackor();
    RET_CODE create_trackor(int uuid_cam=-1);
    RET_CODE trackprocess(TvaiImage &tvimage, VecObjBBox &ins);

    RET_CODE init_model();
    RET_CODE create_model(int uuid_cam=-1);


    // void visual(TvaiImage& tvimage, float* model_output);

    std::vector<CLS_TYPE> _cls_ = {CLS_TYPE::FALLING_OBJ, CLS_TYPE::FALLING_OBJ_UNCERTAIN};

    std::map<int,std::shared_ptr<BoxTraceSet>> m_Trackors;
#ifdef OPENCV3
    std::map<int,cv::Ptr<cv::BackgroundSubtractor>> m_Models;
#else
    std::map<int,cv::shared_ptr<cv::BackgroundSubtractor>> m_Models;
#endif
    int m_dstH = 416;
    int m_dstW = 736;
};



/*******************************************************************************
IMP_OBJECT_REMAIN
chaffee.chen@2022-11-08
*******************************************************************************/
class IMP_OBJECT_REMAIN: public AlgoAPI {
public:
    IMP_OBJECT_REMAIN(){
        m_net = std::make_shared<YuvCropResizeModel>();
        m_ped_net = ucloud::AICoreFactory::getAlgoAPI(ucloud::AlgoAPIName::PED_DETECTOR);
    }
    virtual ~IMP_OBJECT_REMAIN(){}
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);

protected://member function
    RET_CODE create_model(int uuid_cam);
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE postprocess( cv::Mat &cropped_img, VecObjBBox &bboxes_ped, VecObjBBox &bboxes_obj,float aspect_ratio ,int uuid_cam=-1);
    RET_CODE postfilter(TvaiImage &tvimage, VecObjBBox &ins, VecObjBBox &outs, float threshold);

protected://member value
    MLUResize_Ptr m_net = nullptr;
    AlgoAPISPtr m_ped_net = nullptr;
    int m_dstH = 416;
    int m_dstW = 736;
    std::vector<CLS_TYPE> _cls_ = {CLS_TYPE::TARGET};
    float m_ped_threshold = 0.5;

    cv::Mat backgroud_data;
    float m_background_rate = 0.9;
    uint64_t m_background_history_max = 25;
    uint64_t m_background_history = 0; 
    bool m_background_init = false;

#ifdef OPENCV3
    std::map<int,cv::Ptr<cv::BackgroundSubtractor>> m_Models;
#else
    std::map<int,cv::shared_ptr<cv::BackgroundSubtractor>> m_Models;
#endif    
};


#endif