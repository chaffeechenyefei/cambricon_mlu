#ifndef _MODULE_PSPNET_WATER_SEGMENTATION_HPP_
#define _MODULE_PSPNET_WATER_SEGMENTATION_HPP_
#include "module_base.hpp"

#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 积水检测(实际使用分割算法)
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

/*******************************************************************************
PSPNetWaterSegmentV4 基于分割算法的积水检测
chaffee.chen@2022-10-
*******************************************************************************/
class PSPNetWaterSegmentV4: public AlgoAPI{
public:
    PSPNetWaterSegmentV4(){
        m_net = std::make_shared<BaseModelV2>();
    }
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~PSPNetWaterSegmentV4();
    RET_CODE run(TvaiImage &tvimage,VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
private:
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage,VecObjBBox &bboxes);
    RET_CODE run_bgr_on_cpu(TvaiImage& tvimage, VecObjBBox &bboxes);
    RET_CODE postprocess(TvaiImage &tvimage, float* model_output, VecObjBBox &bboxes, float aspect_ratio);

    void visual(TvaiImage& tvimage, float* model_output, float aspect_ratio);

    CLS_TYPE _cls_ = CLS_TYPE::WATER_PUDDLE;

    float m_predict_threshold = 0.5;
    MLUNet_Ptr m_net = nullptr;

    Timer m_Tk;

public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.  
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "pspwater"},
    };        
    bool use_auto_model = false;    
};

/*******************************************************************************
PSPNetWaterSegment 基于分割算法的积水检测
chaffee.chen@2022-10-
*******************************************************************************/
// class PSPNetWaterSegment: public BaseModel{
// public:
//     PSPNetWaterSegment(){};
//     RET_CODE init(const std::string &modelpath);
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     ~PSPNetWaterSegment();
//     /**
//      * @IN:
//      *  tvimage: YUV_NV21 format data
//      * @OUT:
//      *  bboxes: bounding box
//      * @DESC:
//      *  Support model: firstconv(input channel=4, uint8) only.
//      *  When NV21 is input, resize and crop ops are done on mlu.
//      *  Postprocess id done on cpu. Will be moved to mlu.
//      **/
//     RET_CODE run(TvaiImage &tvimage,VecObjBBox &bboxes);
//     RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
//     RET_CODE set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
// private:
//     RET_CODE run_yuv_on_mlu(TvaiImage &tvimage,VecObjBBox &bboxes);
//     RET_CODE run_bgr_on_cpu(TvaiImage& tvimage, VecObjBBox &bboxes);
//     //后处理单元, 不移交指针
//     RET_CODE postprocess(TvaiImage &tvimage, float* model_output, VecObjBBox &bboxes, float aspect_ratio);

//     void visual(TvaiImage& tvimage, float* model_output, float aspect_ratio);

//     CLS_TYPE _cls_ = CLS_TYPE::WATER_PUDDLE;

//     float m_predict_threshold = 0.5;
//     TvaiResolution m_maxTargetSize{0,0};
//     TvaiResolution m_minTargetSize{0,0};

//     VecRect m_pAoiRect;

// public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
//     std::vector<std::string> m_roots = {"/cambricon/model/"};
//     std::string m_basemodel_startswith = "pspwater";
//     bool use_auto_model = false;     
// protected:
//     RET_CODE auto_model_file_search(std::map<InitParam, std::string> &modelpath);
// };


#endif