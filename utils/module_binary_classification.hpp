#ifndef _MODULE_BINARY_CLASSIFICATION_HPP_
#define _MODULE_BINARY_CLASSIFICATION_HPP_
#include "module_base.hpp"

#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
//通用物体分类器, 输入image, 输出区域的类别概率
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;
class BinaryClassificationV4; //use BaseModelV2
class ClassificationV4; //use BaseModelV2 对齐rk3399
class PhoningClassification;
class SmokingClassification;



/*******************************************************************************
BinaryClassificationV4 use BaseModelV2
chaffee.chen@2022-10-08
 * BinaryClassificationV4
 * 解决什么问题: 输入图像, 以及候选区域, 返回候选区域属于某个类别的概率, 同时修改候选框
 * 注意:
 *  FUNC: set_filter_cls, 设定过滤条件, 只对满足过滤条件的候选框进行识别
 *  FUNC: set_primary_output_cls, 模型数据可能有多个维度, 设定仅使用某一个维度, 及对应类别.
 *  FUNC: set_expand_ratio, 对候选框进行扩大, 扩大后的结果输入到模型中进行分类
*******************************************************************************/

class BinaryClassificationV4: public AlgoAPI{
public: 
    BinaryClassificationV4(){ m_net = std::make_shared<BaseModelV2>(); }
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~BinaryClassificationV4();
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &clss);

    /**
     * set_filter_cls
     * 输入框的过滤, 空则所有框都使用
     */
    RET_CODE set_filter_cls(std::vector<CLS_TYPE> &cls_seqs);
    /**
     * softmax输出两个结果下, 以哪个为准, 且对应类别是什么
     */
    RET_CODE set_primary_output_cls(int rank, CLS_TYPE cls);
    RET_CODE set_expand_ratio(float expand_ratio){m_expand_ratio=expand_ratio;}
private:
    RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold);
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold);
    //后处理单元
    RET_CODE postprocess(float *model_output, BBox &bbox, float threshold);

private:
    MLUNet_Ptr m_net = nullptr;
    int m_primary_rank = 1;//输出类别所在模型位置(0 or 1)
    CLS_TYPE m_cls = CLS_TYPE::UNKNOWN;//输出判别类型
    std::vector<CLS_TYPE> m_in_valid_cls;//接收框的类型
    float m_expand_ratio = 1.0;//对输入框的进一步扩大, 
};



/*******************************************************************************
ClassificationV4 use BaseModelV2 对齐rk3399
chaffee.chen@2022-10-17
 * ClassificationV4
 * 解决什么问题: 输入图像, 以及候选区域, 返回候选区域属于某个类别的概率, 同时修改候选框
 * 注意:
 *  FUNC: set_expand_ratio, 对候选框进行扩大, 扩大后的结果输入到模型中进行分类
 *  set_output_cls_order 设定模型输出维度对应的类别 OTHERS表示占位符
*******************************************************************************/

class ClassificationV4: public AlgoAPI{
public: 
    ClassificationV4(){ m_net = std::make_shared<BaseModelV2>(); }
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~ClassificationV4(){}
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    /*******************************************************************************
     * set_output_cls_order
     * 例如 {OTHERS, FIRE, OTHERS} 长度必须与模型输出一致 否则后续运行会FAILED
     * 表示输出的 dim0 = OTHERS, dim1 = FIRE, dim2 = OTHERS
     * OTHERS仅表示占位, 仅dim1会被输出
     *******************************************************************************/    
    RET_CODE set_output_cls_order(std::vector<CLS_TYPE> &clss);
    RET_CODE set_output_cls_order(std::vector<std::string> &clss);
    /*******************************************************************************
     * get_class_type 返回剔除占位类型OTHERS后的有效分类类别
    *******************************************************************************/ 
    RET_CODE get_class_type(std::vector<CLS_TYPE> &clss);

    RET_CODE set_expand_ratio(float expand_ratio){m_expand_ratio=expand_ratio;}
private:
    RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold);
    RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold);
    //后处理单元
    RET_CODE postprocess(float *model_output, BBox &bbox, float threshold);

private:
    MLUNet_Ptr m_net = nullptr;
    std::vector<CLS_TYPE> m_clss;//输出判别类型
    std::vector<std::string> m_clss_str;//输出判别类型
    float m_expand_ratio = 1.0;//对输入框的进一步扩大, 
};


/**
 * BinaryClassificationExtension
 */
typedef struct _SMOKING_BOX{
    BBox body;
    BBox handl;
    BBox handr;
    BBox face;
}SMOKING_BOX;
typedef std::vector<SMOKING_BOX> VecSmokingBox;
class SmokingClassification: public AlgoAPI{
public:
    SmokingClassification(){m_net = std::make_shared<BaseModelV2>();}
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~SmokingClassification();
    //针对抽烟的识别
    RET_CODE run(TvaiImage &tvimage, VecSmokingBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &clss);

    /**
     * softmax输出两个结果下, 以哪个为准, 且对应类别是什么
     */
    RET_CODE set_expand_ratio(float expand_ratio){m_expand_ratio=expand_ratio; return RET_CODE::SUCCESS;}
private:
    RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecSmokingBox &bboxes, float threshold );
    //后处理单元
    RET_CODE postprocess(float *model_output, SMOKING_BOX &bbox, float threshold);

private:
    MLUNet_Ptr m_net = nullptr;
    int m_primary_rank = 0;//输出类别所在模型位置(0 or 1)
    CLS_TYPE m_cls = CLS_TYPE::SMOKING;//输出判别类型
    std::vector<CLS_TYPE> m_in_valid_cls;//接收框的类型
    float m_expand_ratio = 1.0;//对输入框的进一步扩大, 
};

/**
 * Not Binary classification
 * The output length = 3
 */
typedef struct _PED_BOX{
    BBox body;
    BBox head;
    BBox target;
}PED_BOX;
typedef std::vector<PED_BOX> VecPedBox;
class PhoningClassification: public AlgoAPI{
public:
    PhoningClassification(){m_net = std::make_shared<BaseModelV2>();}
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~PhoningClassification();
    //针对抽烟的识别
    RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold=0.6);
    RET_CODE run(TvaiImage &tvimage, VecPedBox &bboxes, float threshold, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &clss);

    /**
     * softmax输出两个结果下, 以哪个为准, 且对应类别是什么
     */
    RET_CODE set_expand_ratio(float expand_ratio){m_expand_ratio=expand_ratio; return RET_CODE::SUCCESS;}
private:
    RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold);
    RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecPedBox &bboxes, float threshold);
    //后处理单元
    RET_CODE postprocess(float *model_output, BBox &bbox, float threshold);
    RET_CODE postprocess(float *model_output, PED_BOX &bbox, float threshold);

private:
    MLUNet_Ptr m_net = nullptr;
    int m_primary_rank = 0;//输出类别所在模型位置(0 or 1)
    CLS_TYPE m_cls = CLS_TYPE::PHONING;//输出判别类型
    std::vector<CLS_TYPE> m_in_valid_cls;//接收框的类型
    float m_expand_ratio = 1.0;//对输入框的进一步扩大, 

};

// class BinaryClassification: public BaseModel{
// public:
//     BinaryClassification(){};
//     RET_CODE init(const std::string &modelpath);
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     ~BinaryClassification();
//     /**
//      * @IN:
//      *  tvimage: BGR/YUV_NV21 format data
//      * @OUT:
//      *  bboxes: bounding box
//      * @DESC:
//      *  Support model: firstconv(input channel=4, uint8) only.
//      *  When NV21 is input, resize and crop ops are done on mlu.
//      *  Postprocess id done on cpu. Will be moved to mlu.
//      **/
//     RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &clss);

//     /**
//      * set_filter_cls
//      * 输入框的过滤, 空则所有框都使用
//      */
//     RET_CODE set_filter_cls(std::vector<CLS_TYPE> &cls_seqs);
//     /**
//      * softmax输出两个结果下, 以哪个为准, 且对应类别是什么
//      */
//     RET_CODE set_primary_output_cls(int rank, CLS_TYPE cls);
//     RET_CODE set_expand_ratio(float expand_ratio){m_expand_ratio=expand_ratio;}
//     RET_CODE set_threshold(float threshold){m_threshold=threshold;}
// private:
//     RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     //后处理单元
//     RET_CODE postprocess(float *model_output, BBox &bbox);

// private:
//     int m_primary_rank = 1;//输出类别所在模型位置(0 or 1)
//     CLS_TYPE m_cls = CLS_TYPE::UNKNOWN;//输出判别类型
//     std::vector<CLS_TYPE> m_in_valid_cls;//接收框的类型
//     float m_expand_ratio = 1.0;//对输入框的进一步扩大, 

//     float m_threshold = 0.5;
// };

#endif