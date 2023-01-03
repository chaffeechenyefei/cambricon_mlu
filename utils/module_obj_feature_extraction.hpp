#ifndef _MODULE_OBJ_FEATURE_EXTRACTION_HPP_
#define _MODULE_OBJ_FEATURE_EXTRACTION_HPP_
#include "module_base.hpp"

#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
//通用物体特征提取, 用于tracking算法
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

/*******************************************************************************
ObjFeatureExtractionV2 使用BaseModelV2
*******************************************************************************/
class ObjFeatureExtractionV2: public AlgoAPI{
public:
    ObjFeatureExtractionV2();
    RET_CODE init(WeightData wData);
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(const std::string &modelpath, MODEL_INPUT_FORMAT inpFMT, bool keep_aspect_ratio, bool pad_both_side);
    ~ObjFeatureExtractionV2(){};
    /**
     * @IN:
     *  tvimage: YUV_NV21 format data 不支持RGB/BGR数据
     * @OUT:
     *  bboxes: bounding box
     * @DESC:
     *  Support model: firstconv(input channel=4, uint8) only.
     *  When NV21 is input, resize and crop ops are done on mlu.
     *  Postprocess id done on cpu. Will be moved to mlu.
     **/
    RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes);
    // RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
private:
    //后处理单元, 这里移交了指针, 所以不要释放model_output
    // RET_CODE postprocess(std::shared_ptr<float> &model_output, BBox &bbox);
    RET_CODE postprocess(float* model_output, BBox &bbox);

    RET_CODE run_yuv_on_mlu_batch(TvaiImage &tvaiImage, VecObjBBox &bboxes);

    CLS_TYPE _cls_ = CLS_TYPE::UNKNOWN;

    MLUNet_Ptr m_net = nullptr;
};

/*******************************************************************************
ObjFeatureExtraction 继承 BaseModel
*******************************************************************************/
// class ObjFeatureExtraction: public BaseModel{
// public:
//     ObjFeatureExtraction(){};
//     RET_CODE init(const std::string &modelpath);
//     RET_CODE init(const std::string &modelpath, MODEL_INPUT_FORMAT inpFMT, bool keep_aspect_ratio, bool pad_both_side);
//     ~ObjFeatureExtraction();
//     /**
//      * @IN:
//      *  tvimage: YUV_NV21 format data 不支持RGB/BGR数据
//      * @OUT:
//      *  bboxes: bounding box
//      * @DESC:
//      *  Support model: firstconv(input channel=4, uint8) only.
//      *  When NV21 is input, resize and crop ops are done on mlu.
//      *  Postprocess id done on cpu. Will be moved to mlu.
//      **/
//     RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes);
//     RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
// private:
//     RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes);
//     // RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     //后处理单元, 这里移交了指针, 所以不要释放model_output
//     // RET_CODE postprocess(std::shared_ptr<float> &model_output, BBox &bbox);
//     RET_CODE postprocess(float* model_output, BBox &bbox);

//     RET_CODE run_yuv_on_mlu_batch(TvaiImage &tvaiImage, VecObjBBox &bboxes);

//     CLS_TYPE _cls_ = CLS_TYPE::UNKNOWN;
// };




#endif