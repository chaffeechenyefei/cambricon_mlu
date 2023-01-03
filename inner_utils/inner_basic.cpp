#include "inner_basic.hpp"

using namespace ucloud;
using namespace cv;

#define NMS_UNION 0
#define NMS_MIN 1

void ucloud::create_mlu_resize_func(PtrHandle* ptrHandle, edk::MluContext* env, edk::MluResizeConvertOp::ColorMode colorMode ,bool pad_both_side, bool keep_aspect_ratio){
    create_mlu_resize_func(ptrHandle, env, colorMode, ptrHandle->inputShape_.H(), ptrHandle->inputShape_.W(), pad_both_side, keep_aspect_ratio);
}

void ucloud::create_mlu_resize_func(PtrHandleV2* ptrHandle, edk::MluContext* env, edk::MluResizeConvertOp::ColorMode colorMode ,bool pad_both_side, bool keep_aspect_ratio){
    create_mlu_resize_func(ptrHandle, env, colorMode, ptrHandle->inputShape_[0].H(), ptrHandle->inputShape_[0].W(), pad_both_side, keep_aspect_ratio);
}

void ucloud::create_mlu_resize_func(PtrHandle* ptrHandle, edk::MluContext* env, mluColorMode colorMode, int H, int W ,bool pad_both_side, bool keep_aspect_ratio){
    ptrHandle->rc_op_mlu_.Destroy();
    ptrHandle->rc_attr_.dst_h = H;
    ptrHandle->rc_attr_.dst_w = W;
    ptrHandle->rc_attr_.color_mode = colorMode;
    ptrHandle->rc_attr_.keep_aspect_ratio = keep_aspect_ratio; // keep width height ratio
    ptrHandle->rc_attr_.padMethod = (pad_both_side)?0:1;//padding on right or bottom side
    ptrHandle->rc_attr_.batch_size = ptrHandle->inputShape_.N();//batchsize suitable
    ptrHandle->rc_attr_.core_version = env->GetCoreVersion();
    ptrHandle->rc_op_mlu_.SetMluQueue(ptrHandle->infer_.GetMluQueue());//attach to current queue
    if (!ptrHandle->rc_op_mlu_.Init(ptrHandle->rc_attr_)) {
        THROW_EXCEPTION(edk::Exception::INTERNAL, ptrHandle->rc_op_mlu_.GetLastError());
    }
}

void ucloud::create_mlu_resize_func(PtrHandleV2* ptrHandle, edk::MluContext* env, mluColorMode colorMode, int H, int W ,bool pad_both_side, bool keep_aspect_ratio){
    ptrHandle->rc_op_mlu_.Destroy();
    ptrHandle->rc_attr_.dst_h = H;
    ptrHandle->rc_attr_.dst_w = W;
    ptrHandle->rc_attr_.color_mode = colorMode;
    ptrHandle->rc_attr_.keep_aspect_ratio = keep_aspect_ratio; // keep width height ratio
    ptrHandle->rc_attr_.padMethod = (pad_both_side)?0:1;//padding on right or bottom side
    ptrHandle->rc_attr_.batch_size = ptrHandle->inputShape_[0].N();//batchsize suitable
    ptrHandle->rc_attr_.core_version = env->GetCoreVersion();
    ptrHandle->rc_op_mlu_.SetMluQueue(ptrHandle->infer_.GetMluQueue());//attach to current queue
    if (!ptrHandle->rc_op_mlu_.Init(ptrHandle->rc_attr_)) {
        THROW_EXCEPTION(edk::Exception::INTERNAL, ptrHandle->rc_op_mlu_.GetLastError());
    }
}

void ucloud::create_mlu_resize_func_light(PtrHandle* ptrHandle, edk::MluContext* env, mluColorMode colorMode, int H, int W ,bool pad_both_side, bool keep_aspect_ratio){
    ptrHandle->rc_op_mlu_.Destroy();
    ptrHandle->rc_attr_.dst_h = H;
    ptrHandle->rc_attr_.dst_w = W;
    ptrHandle->rc_attr_.color_mode = colorMode;
    ptrHandle->rc_attr_.keep_aspect_ratio = keep_aspect_ratio; // keep width height ratio
    ptrHandle->rc_attr_.padMethod = (pad_both_side)?0:1;//padding on right or bottom side
    ptrHandle->rc_attr_.batch_size = 1;//batchsize suitable
    ptrHandle->rc_attr_.core_version = env->GetCoreVersion();
    // ptrHandle->rc_op_mlu_.SetMluQueue(ptrHandle->infer_.GetMluQueue());//attach to current queue
    if (!ptrHandle->rc_op_mlu_.Init(ptrHandle->rc_attr_)) {
        THROW_EXCEPTION(edk::Exception::INTERNAL, ptrHandle->rc_op_mlu_.GetLastError());
    }
}



/////////////////////////////////////////////////////////////////////
// Basic function
/////////////////////////////////////////////////////////////////////
Mat resize_no_aspect(cv::Mat &Input, cv::Size OupSz, bool inpRGB, bool oupRGB, float &sX, float &sY){
    int OupW = OupSz.width; int OupH = OupSz.height;
    int inpW = Input.cols; int inpH = Input.rows;
    sY = (1.0*OupH)/inpH;
    sX = (1.0*OupW)/inpW;
    Mat Output;
    resize(Input, Output, Size(OupW, OupH));
    if(inpRGB){
        if(oupRGB)
            cvtColor(Output,Output, COLOR_RGB2RGBA);
        else
            cvtColor(Output,Output, COLOR_RGB2BGRA);
    } else{
        if(oupRGB)
            cvtColor(Output,Output, COLOR_BGR2RGBA);
        else
            cvtColor(Output,Output, COLOR_BGR2BGRA);
    }
    return Output;
}

Mat resize(cv::Mat &Input, Size OupSz, bool inpRGB, bool oupRGB,bool pad_both_side, float &aspect_ratio){
    int OupW = OupSz.width; int OupH = OupSz.height;
    int inpW = Input.cols; int inpH = Input.rows;
    float aspect_ratio_H = (1.0*OupH)/inpH;
    float aspect_ratio_W = (1.0*OupW)/inpW;
    aspect_ratio = MIN(aspect_ratio_H, aspect_ratio_W);

    int _h = MIN(int(aspect_ratio*inpH), OupH);
    int _w = MIN(int(aspect_ratio*inpW), OupW);

    Mat Output = Mat::zeros(OupSz, CV_8UC3);
    Mat resizedInput;
    resize(Input, resizedInput, Size(_w, _h));
    int offset_x=0, offset_y=0;
    if(pad_both_side){
        offset_x = (OupW - _w)/2;
        offset_y = (OupH - _h)/2;
    }
    resizedInput.copyTo(Output(Rect(offset_x,offset_y, _w,_h)));
    if(inpRGB){
        if(oupRGB)
            cvtColor(Output,Output, COLOR_RGB2RGBA);
        else
            cvtColor(Output,Output, COLOR_RGB2BGRA);
    } else{
        if(oupRGB)
            cvtColor(Output,Output, COLOR_BGR2RGBA);
        else
            cvtColor(Output,Output, COLOR_BGR2BGRA);
    }
    return Output;
}


void resize_to_rect(Mat &Input, Mat &Output, int OutHW )
{
    Output = Mat::zeros(Size(OutHW,OutHW),Input.type());
    int InH = Input.rows;
    int InW = Input.cols;
    if ( InH > InW ){
        float aspect_ratio = (1.0*OutHW)/InH;
        int _W = aspect_ratio*InW;
        int _H = OutHW;
        Mat temp_resized;
        resize(Input, temp_resized, Size(_W,_H));
        int padding = OutHW - _W;
        int xoffset = padding/2;
        Mat temp_roi = Output(Rect(xoffset,0,_W,_H));
        temp_resized.copyTo(temp_roi);
    } else if ( InH < InW ){
        float aspect_ratio = (1.0*OutHW)/InW;
        int _W = OutHW;
        int _H = aspect_ratio*InH;
        Mat temp_resized;
        resize(Input, temp_resized, Size(_W,_H));
        int padding = OutHW - _H;
        int yoffset = padding/2;
        Mat temp_roi = Output(Rect(0,yoffset,_W,_H));
        temp_resized.copyTo(temp_roi);
    } else{
        resize(Input, Output, Size(OutHW,OutHW));
    }
}

void transform_data(uchar* input_data, int width, int height, Mat& out_im){
    Mat input_im(height,width,CV_8UC3, input_data);
    input_im.copyTo(out_im);
}

float* transform_data(uchar* input_data, int width, int height, int outc, int outw, int outh, float& aspect_ratio,bool output_rgb,float std, bool input_rgb, bool pad_both_side){
    assert(outc==3 || outc == 4);
    bool firstconv = (outc==3)? false:true;
    Mat input_im(height,width,CV_8UC3, input_data);
    float aspect_ratio_w = (1.0*outw)/width;
    float aspect_ratio_h = (1.0*outh)/height;
    aspect_ratio = MIN(aspect_ratio_h, aspect_ratio_w);
    float _w = MIN(aspect_ratio*width, outw);
    float _h = MIN(aspect_ratio*height, outh);
    Mat resized_bgr;
    resize(input_im, resized_bgr, Size(_w,_h));
    Mat dst = Mat::zeros(Size(outw,outh), CV_8UC3);
    int offset_x=0;
    int offset_y=0;
    if(pad_both_side){
        offset_x = (outw - _w)/2;
        offset_y = (outh - _h)/2;
    }
    Mat roi_dst(dst, Rect(offset_x,offset_y,_w,_h));
    resized_bgr.copyTo(roi_dst);
    // BGR->RGBA uchar*
    float* databuf = (float*)malloc(outh*outw*outc*sizeof(float));
    Mat dst_fmt;
    if(firstconv){
        if(output_rgb){
            if(input_rgb)
                cvtColor(dst,dst_fmt, COLOR_RGB2RGBA);
            else
                cvtColor(dst,dst_fmt, COLOR_BGR2RGBA);
        } else {
            if(input_rgb)
                cvtColor(dst,dst_fmt, COLOR_RGB2BGRA);
            else
                cvtColor(dst,dst_fmt, COLOR_BGR2BGRA);
        }
    } else {
        if(output_rgb){
            if(input_rgb)
                dst_fmt = dst;//rgb2rgb
            else
                cvtColor(dst,dst_fmt, COLOR_BGR2RGB);
        } else{
            if(input_rgb)
                cvtColor(dst,dst_fmt, COLOR_RGB2BGR);
            else
                dst_fmt = dst;//bgr2bgr
        }
            
    }

    Mat dst_fp32;
    if(firstconv) //std will be ignored
        dst_fmt.convertTo(dst_fp32,CV_32F);
    else
        dst_fmt.convertTo(dst_fp32,CV_32F,1/std);
    if(!dst_fp32.isContinuous())
        dst_fp32 = dst_fp32.clone();
    
    assert(dst_fp32.total() == outh*outw);
    memcpy(databuf, dst_fp32.data, sizeof(float)*dst_fp32.total()*outc);
    return databuf;
}

float* transform_data_yolo(uchar* input_data, int width, int height, int outc, int outw, int outh, float& aspect_ratio,bool use_rgb,float std, bool input_rgb){
    assert(outc==3 || outc == 4);
    bool firstconv = (outc==3)? false:true;
    Mat input_im(height,width,CV_8UC3, input_data);
    float aspect_ratio_w = (1.0*outw)/width;
    float aspect_ratio_h = (1.0*outh)/height;
    aspect_ratio = MIN(aspect_ratio_h, aspect_ratio_w);
    float _w = MIN(aspect_ratio*width, outw);
    float _h = MIN(aspect_ratio*height, outh);
    Mat resized_bgr;
    resize(input_im, resized_bgr, Size(_w,_h));
    Mat dst = Mat::zeros(Size(outw,outh), CV_8UC3);
    Mat roi_dst(dst, Rect(0,0,_w,_h));
    resized_bgr.copyTo(roi_dst);
    // BGR->RGBA uchar*
    float* databuf = (float*)malloc(outh*outw*outc*sizeof(float));
    Mat dst_fmt;
    if(firstconv){
        if(use_rgb){
            if(input_rgb)
                cvtColor(dst,dst_fmt, COLOR_RGB2RGBA);
            else
                cvtColor(dst,dst_fmt, COLOR_BGR2RGBA);
        } else {
            if(input_rgb)
                cvtColor(dst,dst_fmt, COLOR_RGB2BGRA);
            else
                cvtColor(dst,dst_fmt, COLOR_BGR2BGRA);
        }
    } else {
        if(use_rgb){
            if(input_rgb)
                dst_fmt = dst;//rgb2rgb
            else
                cvtColor(dst,dst_fmt, COLOR_BGR2RGB);
        } else{
            if(input_rgb)
                cvtColor(dst,dst_fmt, COLOR_RGB2BGR);
            else
                dst_fmt = dst;//bgr2bgr
        }
            
    }

    Mat dst_fp32;
    if(firstconv) //std will be ignored
        dst_fmt.convertTo(dst_fp32,CV_32F);
    else
        dst_fmt.convertTo(dst_fp32,CV_32F,1/std);
    if(!dst_fp32.isContinuous())
        dst_fp32 = dst_fp32.clone();
    
    assert(dst_fp32.total() == outh*outw);
    memcpy(databuf, dst_fp32.data, sizeof(float)*dst_fp32.total()*outc);
    return databuf;
}

Mat transform_data(uchar* input_data, int width, int height ,float* output_data, int outc, int outw, int outh, 
float& aspect_ratio,bool use_rgb, bool input_bgr){
    Mat input_im(height,width,CV_8UC3, input_data);
    float aspect_ratio_w = (1.0*outw)/width;
    float aspect_ratio_h = (1.0*outh)/height;
    aspect_ratio = MIN(aspect_ratio_h, aspect_ratio_w);
    float _w = MIN(aspect_ratio*width, outw);
    float _h = MIN(aspect_ratio*height, outh);
    Mat resized_bgr;
    resize(input_im, resized_bgr, Size(_w,_h));
    Mat dst = Mat::zeros(Size(outw,outh), CV_8UC3);
    Mat roi_dst(dst, Rect(0,0,_w,_h));
    resized_bgr.copyTo(roi_dst);
    // BGR->RGBA uchar*
    uchar* rawdata = new uchar[dst.total()*4];
    if(use_rgb){
        Mat rgba(dst.size(), CV_8UC4 , rawdata);
        if(input_bgr)
            cvtColor(dst,rgba, COLOR_BGR2RGBA);
        else
            cvtColor(dst,rgba, COLOR_RGB2RGBA);
    } else {
        Mat bgra(dst.size(), CV_8UC4, rawdata);
        if(input_bgr)
            cvtColor(dst,bgra, COLOR_BGR2BGRA);
        else
            cvtColor(dst,bgra, COLOR_RGB2BGRA);
    }
    for(int i=0; i < dst.total()*4; i++){
        output_data[i] = (float)(rawdata[i]);
    }
    delete[] rawdata;
    return dst;
}

void transform_transpose(float *src, float *dst, int srcH, int srcW){
    int dstH = srcW;
    int dstW = srcH;
    for ( int r = 0 ; r < dstH; r++){
        for ( int c = 0; c < dstW; c++ ){
            *dst++ = src[c*dstH+r];
        }
    }
}

void normalize_l2_unit(float *data, int dims){
    float sqrtSum = 0;
    float* ptr = data;
    for( int i = 0; i < dims; i++ ){
        float t = *ptr++;
        sqrtSum += t*t;
    }
    sqrtSum = sqrtf(sqrtSum) + 1e-3;
    for( int i = 0; i < dims; i++ ){
        *data = (*data)/sqrtSum;
        data++;
    }
}





