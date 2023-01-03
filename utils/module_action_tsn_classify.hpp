#ifndef _MODULE_ACTION_TSN_CLASSIFY_HPP_
#define _MODULE_ACTION_TSN_CLASSIFY_HPP_
#include "module_base.hpp"

#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 行为识别: 打斗仅支持yuv
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

/*******************************************************************************
TSNActionClassifyV4 动作分类
chaffee.chen@2022-10-10
*******************************************************************************/
class TSNActionClassifyV4: public AlgoAPI{
public:
    TSNActionClassifyV4(){
        m_net = std::make_shared<BaseModelV2>();
    }
    /*---内部使用, 设定模型输入的图像数量---*/
    void set_batchsize(int m){m_batchsize=m;}
    int get_batchsize(){return m_batchsize;}
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~TSNActionClassifyV4();
    RET_CODE run(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes ,VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE run(BatchImageIN &batch_tvimages,VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
private:
    RET_CODE run_yuv_on_mlu(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes, VecObjBBox &bboxes, float threshold, float nms_threshold);
    RET_CODE run_yuv_on_mlu(BatchImageIN &batch_tvimages, VecObjBBox &bboxes, float threshold, float nms_threshold);
    //后处理单元, 不移交指针
    RET_CODE postprocess(float* model_output, BBox &bbox);
    RET_CODE postfilter(VecObjBBox &ins, VecObjBBox &outs, float threshold);

    float clip_threshold(float x);

    void merge_batch_bboxes_to_rect(BatchBBoxIN &batch_bboxes, VecRect &rects);

    CLS_TYPE _cls_ = CLS_TYPE::FIGHT;

    MLUNet_Ptr m_net = nullptr;

    float m_threshold_cluster = 0.8;
    int m_max_cluster_buffer = 50;
    float m_default_threshold_fight = 0.8;

    int m_batchsize = 8;

    VecRect m_pAoiRect;
};

/*******************************************************************************
TSNActionClassify 动作分类
chaffee.chen@2021
*******************************************************************************/
// class TSNActionClassify: public BaseModel{
// public:
//     TSNActionClassify(){};
//     void set_batchsize(int m){m_batchsize=m;}
//     int get_batchsize(){return m_batchsize;}
//     RET_CODE init(const std::string &modelpath);
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     ~TSNActionClassify();
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
//     RET_CODE run(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes ,VecObjBBox &bboxes);
//     RET_CODE run(BatchImageIN &batch_tvimages,VecObjBBox &bboxes, float threshold, float nms_threshold);
//     RET_CODE set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
// private:
//     RET_CODE run_yuv_on_mlu(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes, VecObjBBox &bboxes);
//     RET_CODE run_yuv_on_mlu(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
//     //后处理单元, 不移交指针
//     RET_CODE postprocess(float* model_output, BBox &bbox);
//     RET_CODE postfilter(VecObjBBox &ins, VecObjBBox &outs);

//     void merge_batch_bboxes_to_rect(BatchBBoxIN &batch_bboxes, VecRect &rects);

//     CLS_TYPE _cls_ = CLS_TYPE::FIGHT;

//     float m_threshold_cluster = 0.8;
//     int m_max_cluster_buffer = 50;
//     float m_threshold_fight = 0.8;

//     int m_batchsize = 8;

//     VecRect m_pAoiRect;
// };


#endif