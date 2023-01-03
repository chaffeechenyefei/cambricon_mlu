#ifndef _MODULE_IQA_HPP_
#define _MODULE_IQA_HPP_

#include "module_base.hpp"
#include "../inner_utils/ip_iqa_blur.hpp"
#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
//通用检测
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

class IQA_Face_Evaluator: public YuvCropResizeModel{
public:
    IQA_Face_Evaluator(){};
    RET_CODE init(int dstW, int dstH);
    ~IQA_Face_Evaluator();
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
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
private:
    /**
     * @param:
     * input_bboxes: run函数返回的检测框
     * output_bboxes: 过滤后的检测框(框的大小过滤)
     **/
    RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes);
    RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes);
    RET_CODE postprocess(cv::Mat cropped_img, BBox &bbox);

    CLS_TYPE m_output_clss{CLS_TYPE::FACE};
    int m_H;
    int m_W;

    IQA_BLUR m_iqa_blur;
};

#endif