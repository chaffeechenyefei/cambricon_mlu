#ifndef _MODULE_GENERAL_INFERENCE_HPP_
#define _MODULE_GENERAL_INFERENCE_HPP_
#include "module_base.hpp"

#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
//通用推理器
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;
/*******************************************************************************
GeneralInferenceSIMO
chaffee.chen@2022-10-27
*******************************************************************************/
class GeneralInferenceSIMO: public AlgoAPI{
public: 
    GeneralInferenceSIMO(){ m_net = std::make_shared<BaseModelV2>(); }
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~GeneralInferenceSIMO(){}
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
private:
    // RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold);
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold);

private:
    MLUNet_Ptr m_net = nullptr;
};
#endif