#ifndef _INNER_BASIC_HPP_
#define _INNER_BASIC_HPP_

#include <opencv2/opencv.hpp>
#include <cnrt.h>
#include <device/mlu_context.h>
#include <easybang/resize_and_colorcvt.h>
#include <easyinfer/easy_infer.h>
#include <easyinfer/mlu_memory_op.h>
#include <easyinfer/model_loader.h>
#include <vector>
#include "../utils/basic.hpp"

namespace ucloud {
typedef struct PtrHandle_S{
    edk::MluMemoryOp mem_op_;
    edk::EasyInfer infer_;
    edk::ShapeEx inputShape_;
    edk::ShapeEx outputShape_;
    edk::MluResizeConvertOp rc_op_mlu_;
    TransformOp rc_op_cpu_;
    edk::MluResizeConvertOp::Attr rc_attr_;
    std::shared_ptr<edk::ModelLoader> model_{nullptr};
} PtrHandle;

typedef struct PtrHandleV2_S{
    edk::MluMemoryOp mem_op_;
    edk::EasyInfer infer_;
    std::vector<edk::ShapeEx> inputShape_;
    std::vector<edk::ShapeEx> outputShape_;
    edk::MluResizeConvertOp rc_op_mlu_;
    TransformOp rc_op_cpu_;
    edk::MluResizeConvertOp::Attr rc_attr_;
    std::shared_ptr<edk::ModelLoader> model_{nullptr};
} PtrHandleV2;

typedef edk::MluResizeConvertOp::ColorMode mluColorMode;

void create_mlu_resize_func(PtrHandle* ptrHandle, edk::MluContext* env, mluColorMode colorMode ,bool pad_both_side=false, bool keep_aspect_ratio=true);
void create_mlu_resize_func(PtrHandle* ptrHandle, edk::MluContext* env, mluColorMode colorMode, int H, int W ,bool pad_both_side=false, bool keep_aspect_ratio=true);
void create_mlu_resize_func_light(PtrHandle* ptrHandle, edk::MluContext* env, mluColorMode colorMode, int H, int W ,bool pad_both_side=false, bool keep_aspect_ratio=true);

void create_mlu_resize_func(PtrHandleV2* ptrHandle, edk::MluContext* env, mluColorMode colorMode ,bool pad_both_side=false, bool keep_aspect_ratio=true);
void create_mlu_resize_func(PtrHandleV2* ptrHandle, edk::MluContext* env, mluColorMode colorMode, int H, int W ,bool pad_both_side=false, bool keep_aspect_ratio=true);

}

cv::Mat resize(cv::Mat &Input, cv::Size OupSz, bool inpRGB, bool oupRGB,bool pad_both_side, float &aspect_ratio);
cv::Mat resize_no_aspect(cv::Mat &Input, cv::Size OupSz, bool inpRGB, bool oupRGB, float &sX, float &sY);
void resize_to_rect(cv::Mat &Input, cv::Mat &Output, int OutHW = 112 );
void transform_data(uchar* input_data, int width, int height, cv::Mat& out_im);
float* transform_data_yolo(uchar* input_data, int width, int height, int outc, int outw, int outh, float& aspect_ratio,bool use_rgb=false,float std = 1., bool input_rgb=false);
float* transform_data(uchar* input_data, int width, int height, int outc, int outw, int outh, float& aspect_ratio,bool use_rgb=false,float std = 1., bool input_rgb=false, bool pad_both_side=false);
cv::Mat transform_data(uchar* input_data, int width, int height ,float* output_data, int outc, int outw, int outh, float& aspect_ratio,bool use_rgb=false, bool input_bgr=true);
void transform_transpose(float *src, float *dst, int srcH, int srcW);
void normalize_l2_unit(float *data, int dims);

/////////////////////////////////////////////////////////////////////
// inline function
/////////////////////////////////////////////////////////////////////
template<typename T>
inline void argmax(T* data, int L ,int &pos, T &val){
    assert(L>0);
    pos = 0;
    val = data[0];
    for(int i = 1; i < L; i++){
        if(data[i] > val){
            pos = i;
            val = data[i];
        }
    }
}

/////////////////////////////////////////////////////////////////////
// ASSERT function
/////////////////////////////////////////////////////////////////////
#include <sys/stat.h>
inline bool exists_file(const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

inline bool float_in_range(float val, float maxval, float minval){
    if (val<=maxval && val>=minval){
        return true;
    } else
        return false;
}

/////////////////////////////////////////////////////////////////////
// 避免TvaiRect数据越界
/////////////////////////////////////////////////////////////////////
template<typename T>
inline T get_valid_rect(T rect, int W, int H){
    if(rect.x<0)
        rect.x = 0;
    if(rect.y<0)
        rect.y = 0;
    if(rect.width<0)
        rect.width = 0;
    if(rect.height<0)
        rect.height = 0;
    if(rect.x+rect.width>=W)
        rect.width = W - rect.x - 1;
    if(rect.y+rect.height>=H)
        rect.height = H - rect.y - 1;
    return rect;
};



#endif