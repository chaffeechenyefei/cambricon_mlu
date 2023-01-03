#ifndef _MODULE_SKELETON_DETECTION_HPP_
#define _MODULE_SKELETON_DETECTION_HPP_
#include "module_base.hpp"

#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
//骨架检测(开发中)
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

/*******************************************************************************
SkeletonDetectorV4 remove BaseModel
chaffee.chen@2022-10-08
*******************************************************************************/
class SkeletonDetectorV4: public AlgoAPI{
public:
    SkeletonDetectorV4(){m_net=std::make_shared<BaseModelV2>();}
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~SkeletonDetectorV4();
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
    RET_CODE set_output_cls_order(std::vector<CLS_TYPE> &output_clss);
private:
    RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes);
    RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
    //后处理单元
    //new version
    RET_CODE postprocess(float* model_output, TvaiRect pedRect, LandMark &kypts, float aX, float aY);

    std::vector<CLS_TYPE> _cls_ = {CLS_TYPE::PEDESTRIAN};

    MLUNet_Ptr m_net = nullptr;
};


/*******************************************************************************
SkeletonDetector based on BaseModel
*******************************************************************************/
// class SkeletonDetector: public BaseModel{
// public:
//     SkeletonDetector(){};
//     RET_CODE init(const std::string &modelpath);
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     ~SkeletonDetector();
//     RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
//     RET_CODE set_output_cls_order(std::vector<CLS_TYPE> &output_clss);
// private:
//     RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     //后处理单元
//     //not used
//     RET_CODE postprocess(float* model_output, TvaiRect pedRect ,SkLandmark &kypts, float aX, float aY);
//     //new version
//     RET_CODE postprocess(float* model_output, TvaiRect pedRect, LandMark &kypts, float aX, float aY);

//     std::vector<CLS_TYPE> _cls_ = {CLS_TYPE::PEDESTRIAN};
// };


#endif