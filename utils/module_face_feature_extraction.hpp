#ifndef _MODULE_FACE_FEATURE_EXTRACTION_HPP_
#define _MODULE_FACE_FEATURE_EXTRACTION_HPP_
#include "module_base.hpp"
#include "basic.hpp"

#include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
//人脸特征提取
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;
/*******************************************************************************
FaceExtractionV4 使用BaseModelV2
*******************************************************************************/
class FaceExtractionV4: public AlgoAPI{
public:
    FaceExtractionV4();
    RET_CODE init(const std::string &modelpath);
    /**
     * 20211117
     * 新接口形式
     */
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~FaceExtractionV4();
    RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
protected:
    //后处理单元, 这里移交了指针, 所以不要释放model_output
    RET_CODE postprocess(float* model_output, BBox &bbox);

    CLS_TYPE _cls_ = CLS_TYPE::FACE;
    MLUNet_Ptr m_net = nullptr;

    Timer m_Tk;

public://yolo检测系列, 开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "resnet101"},
    };        
    bool use_auto_model = false;    
};


/*******************************************************************************
FaceAttributionV4 使用BaseModelV2
*******************************************************************************/
/**
 * FaceAttribution
 * EfficientNet 非通用模型, 输出b,H,W,C = 1,1,1,103 (2+101)
 * 2 for sex
 * 101 for age
 */
class FaceAttributionV4: public AlgoAPI{
public:
    FaceAttributionV4(){m_net=std::make_shared<BaseModelV2>();}
    RET_CODE init(std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~FaceAttributionV4();
    RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold=0.5, float nms_threshold=0.6);
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
    static float get_box_expand_ratio();
protected:
    RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes);
    RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
    //后处理单元, 这里移交了指针, 所以不要释放model_output
    RET_CODE postprocess(float* model_output, BBox &bbox);

    CLS_TYPE _cls_ = CLS_TYPE::FACE;
    MLUNet_Ptr m_net = nullptr;

    static constexpr float _expand_ratio = 2.0f;

public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
        std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "faceattr"},
    };
    bool use_auto_model = false;    

}; 


// class FaceExtractionV2: public BaseModel{
// public:
//     FaceExtractionV2(){};
//     RET_CODE init(const std::string &modelpath);
//     /**
//      * 20211117
//      * 新接口形式
//      */
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     ~FaceExtractionV2();
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
//     RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes);
//     RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
// protected:
//     RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     //后处理单元, 这里移交了指针, 所以不要释放model_output
//     RET_CODE postprocess(float* model_output, BBox &bbox);

//     CLS_TYPE _cls_ = CLS_TYPE::FACE;

// public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
//     std::vector<std::string> m_roots = {"/cambricon/model/"};
//     std::string m_basemodel_startswith = "resnet101";
//     bool use_auto_model = false;    
// protected:
//     RET_CODE auto_model_file_search(std::map<InitParam, std::string> &modelpath);    
// };

// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// //人脸属性分类(年龄、性别) 
// //2022-06-28 initial
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// /**
//  * FaceAttribution
//  * EfficientNet 非通用模型, 输出b,H,W,C = 1,1,1,103 (2+101)
//  * 2 for sex
//  * 101 for age
//  */
// class FaceAttribution: public BaseModel{
// public:
//     FaceAttribution(){};
//     RET_CODE init(std::string &modelpath);
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     ~FaceAttribution();
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
//     RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
//     static float get_box_expand_ratio();
// protected:
//     RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes);
//     RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes);
//     //后处理单元, 这里移交了指针, 所以不要释放model_output
//     RET_CODE postprocess(float* model_output, BBox &bbox);

//     CLS_TYPE _cls_ = CLS_TYPE::FACE;

//     static constexpr float _expand_ratio = 2.0f;

// public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
//     std::vector<std::string> m_roots = {"/cambricon/model/"};
//         std::map<ucloud::InitParam, std::string> m_models_startswith = {
//         {InitParam::BASE_MODEL, "faceattr"},
//     };
//     bool use_auto_model = false;    

// }; 



#endif