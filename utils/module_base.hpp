#ifndef _MODULE_BASE_HPP_
#define _MODULE_BASE_HPP_
#include "../libai_core.hpp"
#include <opencv2/opencv.hpp>
#include <device/mlu_context.h>
#include <easytrack/easy_track.h>
#include "../inner_utils/inner_basic.hpp"
#include <glog/logging.h>
#include <mutex>


#if CV_VERSION_EPOCH == 2 //mlu270: opencv2.4.9-dev
#define OPENCV2
#elif CV_VERSION_MAJOR == 3 //mlu220: opencv3.4.6
#define  OPENCV3
#else
#error Not support this OpenCV version  
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////
//基础工具类(开发中) public PrivateContext(Singleton环境设定), public AlgoAPI(外部接口)
//基础工具类中的函数, 均应不与AlgoAPI发生冲突
////////////////////////////////////////////////////////////////////////////////////////////////////////
#define NMS_UNION 0
#define NMS_MIN 1

#ifdef VERBOSE
#define LOGI LOG(INFO)
#else
#define LOGI 0 && LOG(INFO)
#endif

#define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

using namespace ucloud;

/*******************************************************************************
目录
*******************************************************************************/
class PrivateContextV2; //MLU环境变量
class BaseModelV2;  //基于mlu的核心推理模块
class YuvCropResizeModel;   //基于mlu的图像尺度缩放模块
class MLUMemPool;   //mlu内存池

using MLUNet_Ptr = std::shared_ptr<BaseModelV2>;
using MLUResize_Ptr = std::shared_ptr<YuvCropResizeModel>;

// class BaseModel; //弃用


typedef enum {
    BGRA = 0,
    RGBA = 1,
    ABGR = 2,
    ARGB = 3,
}MODEL_INPUT_FORMAT;

typedef enum {
    NCHW = 0,
    NHWC = 1,
}MODEL_OUTPUT_ORDER;

class PrivateContextV2{
public:
    PrivateContextV2();
    virtual ~PrivateContextV2();
protected:
bool status_=false;
edk::MluContext* env_ = nullptr;
};

typedef struct _BASE_CONFIG_ {
    MODEL_INPUT_FORMAT model_input_fmt;//模型输入格式:RGBA/BGRA
    MODEL_OUTPUT_ORDER model_output_order;//模型输出格式:NCHW/NHWC
    bool pad_both_side = false;
    bool keep_aspect_ratio = true;
    _BASE_CONFIG_(MODEL_INPUT_FORMAT param1, MODEL_OUTPUT_ORDER param2, bool _pad_both_side = false, bool _keep_aspect_ratio = true)\
                :model_input_fmt(param1),model_output_order(param2),pad_both_side(_pad_both_side),keep_aspect_ratio(_keep_aspect_ratio) {}
    _BASE_CONFIG_(){
        model_input_fmt=MODEL_INPUT_FORMAT::RGBA;
        model_output_order=MODEL_OUTPUT_ORDER::NHWC;
        pad_both_side = false;
        keep_aspect_ratio = true;
        }
}BASE_CONFIG;


/**
 * BaseModelV2: 提供一些通用基础组件, 减少代码重复度, 便于后期维护
 * 设计目的: 需要支持多输入多输出的模型
 * 继承
 * - PrivateContext: MLU通用环境变量
 * */
class BaseModelV2: public PrivateContextV2{
public:
    BaseModelV2(){
    };
    RET_CODE base_init(const std::string &modelpath, BASE_CONFIG config);
    RET_CODE base_init(WeightData wdata, BASE_CONFIG config);
    virtual ~BaseModelV2();
    virtual void release();

public://Method
    /**
     * TRANS:
     * ([Img]x1, [ROI]x1) -> [1,C,H,W]
     * 通用前处理: 将输入的图像数据处理后存入类共享的mlu空间.
     * 仅支持yuv格式, 因为yuv格式下可以端到端实现前处理及推理, 所以前处理及推理可以放在一个循环中进行.
     * IN:
     * tvimage: 输入图像数据
     * roiRect: 限定目标区域
     * OUT:
     * aspect_ratio: 保持长宽比例的缩放比例
     * aX: 不保持长宽比例下, X方向的缩放比例
     * aY: 不保持长宽比例下, Y方向的缩放比例
     * */
    RET_CODE general_preprocess_yuv_on_mlu_union(TvaiImage &tvimage, TvaiRect roiRect, float &aspect_ratio, float &aX, float &aY);
    RET_CODE general_preprocess_bgr_on_cpu(TvaiImage &tvimage, float &aspect_ratio, float &aX, float &aY);
    
    /**
     * TRANS:
     * ([Img]x1, [ROI]xT) -> [1,C,H,W]xT -> [1,oC,oH,oW]xT
     * RGB/BGR通用前处理+推理:
     * 针对多输入框的模式
     * IN:
     * tvimage: 输入图像数据
     * roiRects: 多个限定目标区域
     * OUT:
     * [1,C,H,W]
     * model_output: 模型推理后结果
     * aspect_ratio: 每个限定目标区域的缩放(如果保持长宽比, 则每个区域一个值;如果不保持长宽比, 则每个区域X,Y两个值)
     * */
    RET_CODE general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, std::vector<TvaiRect>& roiRects, std::vector<float*> &model_output, std::vector<float> &aspect_ratios);
    RET_CODE general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox& bboxes, std::vector<float*> &model_output, std::vector<float> &aspect_ratios, std::vector<CLS_TYPE> &valid_class);

    /** 
     * TRANS:
     * ([Img],[ROI]) -> [B,C,H,W], where B > 1 and B = _N
     * IN: 
     * BATCH操作, yuv的物理地址和虚拟地址合并
     * 两种类型的BATCH
     * 第一种: 输入图像是BATCH形式的
     * 第二种: 输入单个图像, 对ROI区域进行BATCH操作
     */
    /** case I: 每个输入图像可以不同ROI
     * TRANS:
     * ([Img]xB,[ROI]xB) -> [B,C,H,W], where B > 1 and B = _N
     */
    RET_CODE general_batch_preprocess_yuv_on_mlu(BatchImageIN &batch_tvimage, std::vector<TvaiRect> &batch_roiRect,std::vector<float> &batch_aspect_ratio);
    /** case II: 单个输入图像, 多个ROI
     * TRANS:
     * ([Img]x1,[ROI]xB) -> [B,C,H,W], where B > 1 and B = _N
     */
    RET_CODE general_batch_preprocess_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox& bboxes,std::vector<float> &batch_aspect_ratio, int offset);
    
    /**
     * 通用模型推理
     * float** for Multiple Output
     * float* for each vector of output
     * */
    float** general_mlu_infer();
    // float** general_mlu_infer(MLUMemNode* ptr);
    void cpu_free(float **ptrX);


public://Variable
    MODEL_INPUT_FORMAT _model_input_fmt{MODEL_INPUT_FORMAT::RGBA};
    MODEL_OUTPUT_ORDER _model_output_order{MODEL_OUTPUT_ORDER::NCHW};
    bool _keep_aspect_ratio = true;
    bool _pad_both_side = false;

    int _MI;//MIMO输入数量
    int _MO;

    PtrHandleV2* _ptrHandle=nullptr;
    std::vector<edk::ShapeEx> m_inputShape;
    std::vector<edk::ShapeEx> m_outputShape;
    /**
     * mlu_output_, mlu_input_的数量的概念 = 模型输入输出tensor的数量例如 output1,output2 = model(input1,input2,...) 
     */
    void **mlu_output_{nullptr}, **mlu_input_{nullptr}, **cpu_output_{nullptr}, **cpu_input_{nullptr};//
    // MLUMemPool* m_mem_pool = nullptr;
};


/**
 * 20211109
 * YuvCropResizeModel: 提供一些通用基础组件, 减少代码重复度, 便于后期维护
 * 没有模型, 仅进行mlu上的图像crop
 * 继承
 * - PrivateContext: MLU通用环境变量
 * - AlgoAPI: 抽象类, 主要用于暴露接口, 无任何私有变量
 * */
class YuvCropResizeModel: public PrivateContextV2, public AlgoAPI{
public:
    YuvCropResizeModel(){};
    /**
     * 读取模型权重文件, 并设置模型输入输出格式
     * */
    RET_CODE base_init(int dstH, int dstW, BASE_CONFIG config);
    virtual ~YuvCropResizeModel();

public:
    /**
     * 通用前处理: 将输入的图像数据处理后存入类共享的mlu空间.
     * 仅支持yuv格式, 因为yuv格式下可以端到端实现前处理及推理, 所以前处理及推理可以放在一个循环中进行.
     * IN:
     * tvimage: 输入图像数据
     * roiRect: 限定目标区域
     * OUT:
     * aspect_ratio: 保持长宽比例的缩放比例
     * aX: 不保持长宽比例下, X方向的缩放比例
     * aY: 不保持长宽比例下, Y方向的缩放比例
     * */
    RET_CODE general_preprocess_yuv_on_mlu_phyAddr(TvaiImage &tvimage, TvaiRect roiRect, cv::Mat& cropped_img, float &aspect_ratio, float &aX, float &aY);
    RET_CODE general_preprocess_yuv_on_mlu(TvaiImage &tvimage, TvaiRect roiRect, cv::Mat& cropped_img, float &aspect_ratio, float &aX, float &aY);

    /**
     * RGB/BGR通用前处理+推理:
     * 针对多输入框的模式
     * IN:
     * tvimage: 输入图像数据
     * roiRects: 多个限定目标区域
     * OUT:
     * model_output: 模型推理后结果
     * aspect_ratio: 每个限定目标区域的缩放(如果保持长宽比, 则每个区域一个值;如果不保持长宽比, 则每个区域X,Y两个值)
     * */
    RET_CODE general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, std::vector<TvaiRect>& roiRects, std::vector<cv::Mat> &cropped_imgs, std::vector<float> &aspect_ratios);
    RET_CODE general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox& bboxes, std::vector<cv::Mat> &cropped_imgs, std::vector<float> &aspect_ratios, std::vector<CLS_TYPE> &valid_class);

protected:
    virtual void release();
    std::mutex _mlu_mutex;

    MODEL_INPUT_FORMAT _model_input_fmt{MODEL_INPUT_FORMAT::RGBA};
    bool _keep_aspect_ratio = true;
    bool _pad_both_side = false;

    //model input Tensor
    int _H;
    int _W;
    int _C;
    int _N;

    void* _ptrHandle=nullptr;
};


/*******************************************************************************
MLUMemPool
适配寒武纪
*******************************************************************************/
typedef struct _MLUMemNode{
    // void** cpu_input_ptr = nullptr;
    // void** cpu_output_ptr = nullptr;
    void** mlu_input_ptr = nullptr;
    void** mlu_output_ptr = nullptr;
    _MLUMemNode* next = nullptr;
} MLUMemNode;

class MLUMemPool{
public:
    MLUMemPool(){}
    /*use bind_mem_handle before init*/
    // void init(int num_of_nodes);
    void bind_mem_handle(PtrHandleV2* handle){ memHandle = handle;}
    ~MLUMemPool();
    /*外部函数, 供外部使用*/
    MLUMemNode* malloc();
    void free(MLUMemNode*);
protected:
    MLUMemNode* create(int num_of_nodes);
    PtrHandleV2* memHandle = nullptr;
    /*---内部函数, 释放节点内部开辟的空间---*/
    void _free_(MLUMemNode* ptr);
    /*---内部函数, 在节点内开辟空间, 输入必须new开辟过空间---*/
    bool _malloc_(MLUMemNode* ptr);

    int numOfNodes = 0;
    MLUMemNode* freeNodeHeader = nullptr;
    std::set<MLUMemNode*> occupiedNodes;

    std::mutex _mlock_;
};


void transform_string_to_cls_type(std::vector<std::string> &vec_str, std::vector<CLS_TYPE> &vec_t);


template<typename T>
void base_nmsBBox(std::vector<T>& input, float threshold, int type, std::vector<T>& output);
void base_nmsBBox(std::vector<VecObjBBox> &input, float threshold, int type, VecObjBBox &output);
template<typename T>
void base_transform_xyxy_xyhw(std::vector<T> &vecbox, float expand_ratio ,float aspect_ratio){
    for (int i=0 ; i < vecbox.size(); i++ ){
        float cx = (vecbox[i].x0 + vecbox[i].x1)/(2*aspect_ratio);
        float cy = (vecbox[i].y0 + vecbox[i].y1)/(2*aspect_ratio);
        float w = (vecbox[i].x1 - vecbox[i].x0)*expand_ratio/aspect_ratio;
        float h = (vecbox[i].y1 - vecbox[i].y0)*expand_ratio/aspect_ratio;
        float _x0 = cx - w/2;
        float _y0 = cy - h/2;

        vecbox[i].rect.x = int(_x0);
        vecbox[i].rect.y = int(_y0);
        vecbox[i].rect.width = int(w);
        vecbox[i].rect.height = int(h);
        for(int j=0;j<vecbox[i].Pts.pts.size(); j++){
            vecbox[i].Pts.pts[j].x /= aspect_ratio;
            vecbox[i].Pts.pts[j].y /= aspect_ratio;
        }
    }
};
template<typename T>
void base_transform_xyxy_xyhw(std::vector<T> &vecbox, float expand_ratio ,float aX, float aY){
    for (int i=0 ; i < vecbox.size(); i++ ){
        float cx = (vecbox[i].x0 + vecbox[i].x1)/(2*aX);
        float cy = (vecbox[i].y0 + vecbox[i].y1)/(2*aY);
        float w = (vecbox[i].x1 - vecbox[i].x0)*expand_ratio/aX;
        float h = (vecbox[i].y1 - vecbox[i].y0)*expand_ratio/aY;
        float _x0 = cx - w/2;
        float _y0 = cy - h/2;

        vecbox[i].rect.x = int(_x0);
        vecbox[i].rect.y = int(_y0);
        vecbox[i].rect.width = int(w);
        vecbox[i].rect.height = int(h);
        for(int j=0;j<vecbox[i].Pts.pts.size(); j++){
            vecbox[i].Pts.pts[j].x /= aX;
            vecbox[i].Pts.pts[j].y /= aY;
        }
    }
};

void base_output2ObjBox_multiCls(float* output ,std::vector<VecObjBBox> &vecbox, CLS_TYPE* cls_map, std::map<CLS_TYPE, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold=0.8, int dimOffset=5);
void base_output2ObjBox_multiCls(float* output ,std::vector<VecObjBBox> &vecbox, std::vector<CLS_TYPE> &cls_map, std::map<CLS_TYPE, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold=0.8, int dimOffset=5);
void base_output2ObjBox_multiCls(float* output ,std::vector<VecObjBBox> &vecbox, std::vector<std::string> &cls_map, std::map<std::string, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold=0.8, int dimOffset=5);

void base_output2ObjBox_multiCls_yoloface(float* output ,std::vector<VecObjBBox> &vecbox, CLS_TYPE* cls_map, std::map<CLS_TYPE, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold=0.8, int dimOffset=15);
void base_output2ObjBox_multiCls_yoloface(float* output ,std::vector<VecObjBBox> &vecbox, std::vector<CLS_TYPE> &cls_map, std::map<CLS_TYPE, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold=0.8, int dimOffset=15);

inline TvaiResolution base_get_valid_maxSize(TvaiResolution rect){
    // if width or height is set zero, then this condition will be ignored.
    unsigned int maxVal = 4096;
    TvaiResolution _rect;
    _rect.height = (rect.height==0)? maxVal:rect.height;
    _rect.width = (rect.width==0)? maxVal:rect.width;
    return _rect;
}

TvaiRect globalscaleTvaiRect(TvaiRect &rect, float scale, int W, int H);

/**
 * 根据定义的文件名开头字符,自动获取对应的模型文件
 */
RET_CODE auto_model_file_search( std::vector<std::string> &roots, std::map<InitParam, std::string> &fileBeginNameIN, std::map<InitParam, std::string> &modelpathOUT);



/**
 * BaseModel: 提供一些通用基础组件, 减少代码重复度, 便于后期维护
 * 继承
 * - PrivateContext: MLU通用环境变量
 * - AlgoAPI: 抽象类, 主要用于暴露接口, 无任何私有变量
 * */
// class BaseModel: public PrivateContext, public AlgoAPI{
// public:
//     BaseModel(){};
//     /**
//      * 读取模型权重文件, 并设置模型输入输出格式
//      * */
//     RET_CODE base_init(const std::string &modelpath, BASE_CONFIG config);
//     virtual ~BaseModel();

// protected:
//     /**
//      * TRANS:
//      * ([Img]x1) -> [1,C,H,W]
//      * 通用前处理: 将输入的图像数据处理后存入类共享的mlu空间.
//      * IN:
//      * tvimage: 输入图像数据
//      * OUT:
//      * aspect_ratio: 保持长宽比例的缩放比例
//      * aX: 不保持长宽比例下, X方向的缩放比例
//      * aY: 不保持长宽比例下, Y方向的缩放比例
//      * */
//     RET_CODE general_preprocess_yuv_on_mlu_phyAddr(TvaiImage &tvimage, float &aspect_ratio, float &aX, float &aY);
//     RET_CODE general_preprocess_bgr_on_cpu(TvaiImage &tvimage, float &aspect_ratio, float &aX, float &aY);
//     RET_CODE general_preprocess_yuv_on_mlu(TvaiImage &tvimage, float &aspect_ratio, float &aX, float &aY);
//     /**
//      * TRANS:
//      * ([Img]x1, [ROI]x1) -> [1,C,H,W]
//      * 通用前处理: 将输入的图像数据处理后存入类共享的mlu空间.
//      * 仅支持yuv格式, 因为yuv格式下可以端到端实现前处理及推理, 所以前处理及推理可以放在一个循环中进行.
//      * IN:
//      * tvimage: 输入图像数据
//      * roiRect: 限定目标区域
//      * OUT:
//      * aspect_ratio: 保持长宽比例的缩放比例
//      * aX: 不保持长宽比例下, X方向的缩放比例
//      * aY: 不保持长宽比例下, Y方向的缩放比例
//      * */
//     RET_CODE general_preprocess_yuv_on_mlu_phyAddr(TvaiImage &tvimage, TvaiRect roiRect, float &aspect_ratio, float &aX, float &aY);
//     RET_CODE general_preprocess_yuv_on_mlu(TvaiImage &tvimage, TvaiRect roiRect, float &aspect_ratio, float &aX, float &aY);
//     /**
//      * 合并general_preprocess_yuv_on_mlu, general_preprocess_yuv_on_mlu_phyAddr
//      */
//     RET_CODE general_preprocess_yuv_on_mlu_union(TvaiImage &tvimage, TvaiRect roiRect, float &aspect_ratio, float &aX, float &aY);

//     /**
//      * TRANS:
//      * ([Img]x1, [ROI]xT) -> [1,C,H,W]xT -> [1,oC,oH,oW]xT
//      * RGB/BGR通用前处理+推理:
//      * 针对多输入框的模式
//      * IN:
//      * tvimage: 输入图像数据
//      * roiRects: 多个限定目标区域
//      * OUT:
//      * [1,C,H,W]
//      * model_output: 模型推理后结果
//      * aspect_ratio: 每个限定目标区域的缩放(如果保持长宽比, 则每个区域一个值;如果不保持长宽比, 则每个区域X,Y两个值)
//      * */
//     RET_CODE general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, std::vector<TvaiRect>& roiRects, std::vector<float*> &model_output, std::vector<float> &aspect_ratios);
//     RET_CODE general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox& bboxes, std::vector<float*> &model_output, std::vector<float> &aspect_ratios, std::vector<CLS_TYPE> &valid_class);
//     /**
//      * 通用模型推理
//      * */
//     float* general_mlu_infer();
//     /**
//      * 仅供内部调用, 自动析构开辟的数据空间
//      */
//     std::shared_ptr<float> general_mlu_infer_share_ptr();

//     /** 
//      * TRANS:
//      * ([Img],[ROI]) -> [B,C,H,W], where B > 1 and B = _N
//      * IN: 
//      * BATCH操作, yuv的物理地址和虚拟地址合并
//      * 两种类型的BATCH
//      * 第一种: 输入图像是BATCH形式的
//      * 第二种: 输入单个图像, 对ROI区域进行BATCH操作
//      */
//     /** case I: 每个输入图像可以不同ROI
//      * TRANS:
//      * ([Img]xB,[ROI]xB) -> [B,C,H,W], where B > 1 and B = _N
//      */
//     RET_CODE general_batch_preprocess_yuv_on_mlu(BatchImageIN &batch_tvimage, std::vector<TvaiRect> &batch_roiRect,std::vector<float> &batch_aspect_ratio);
//     /** case II: 单个输入图像, 多个ROI
//      * TRANS:
//      * ([Img]x1,[ROI]xB) -> [B,C,H,W], where B > 1 and B = _N
//      */
//     RET_CODE general_batch_preprocess_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox& bboxes,std::vector<float> &batch_aspect_ratio, int offset);

//     /** 
//      * TRANS:
//      * ([Img]x1, [ROI]xN) -> [1,C,H,W]xN, where N > 1 and N = _MI
//      * MIMO操作, multiple input tensor, single output tensor
//      * 第一种: 单个图像输入, 单个图像中多个ROI区域组成一组输入
//      * 第二种: 多个图像输入, 每个图像中的ROI区域组成一组输入
//      */
//     RET_CODE general_preprocess_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox& bboxes,std::vector<float> &batch_aspect_ratio, int offset);

//     virtual void release();
//     std::mutex _mlu_mutex;

//     MODEL_INPUT_FORMAT _model_input_fmt{MODEL_INPUT_FORMAT::RGBA};
//     MODEL_OUTPUT_ORDER _model_output_order{MODEL_OUTPUT_ORDER::NCHW};
//     bool _keep_aspect_ratio = true;
//     bool _pad_both_side = false;

//     //model input Tensor
//     int _H;
//     int _W;
//     int _C;
//     int _N;
//     int _MI;//MIMO输入数量
//     //model output Tensor
//     int _oH, _oW, _oC;

//     void* _ptrHandle=nullptr;
//     /**
//      * mlu_output_, mlu_input_的数量的概念 = 模型输入输出tensor的数量例如 output1,output2 = model(input1,input2,...) 
//      */
//     void **mlu_output_{nullptr}, **cpu_output_{nullptr}, **mlu_input_{nullptr}, **cpu_input_{nullptr};
// };

#endif