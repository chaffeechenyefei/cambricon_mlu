#ifndef _MODULE_MOD_UNET2DSHIFT_HPP_
#define _MODULE_MOD_UNET2DSHIFT_HPP_
#include "module_base.hpp"
#include "module_nn_match.hpp"
#include <mutex>
#include <map>
////////////////////////////////////////////////////////////////////////////////////////////////////////
// 高空抛物: 仅支持yuv
////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace ucloud;

/*******************************************************************************
MovementSegment 基于分割算法的高空抛物
chaffee.chen@2022-10-
*******************************************************************************/
class MovementSegment: public AlgoAPI{
public:
    MovementSegment(){
        m_net = std::make_shared<BaseModelV2>();
    }
    RET_CODE init(const std::string &modelpath);
    RET_CODE init(std::map<InitParam, std::string> &modelpath);
    ~MovementSegment();
    /*---对外接口---*/
    RET_CODE run(TvaiImage &batch_tvimages,VecObjBBox &bboxes, float threshold, float nms_threshold);
    /**---实际处理的函数---**/
    RET_CODE run(BatchImageIN &batch_tvimages,VecObjBBox &bboxes, float threshold, float nms_threshold);
    
    RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
private:
    RET_CODE push_back(TvaiImage &tvimage);
    RET_CODE clear();
    RET_CODE run_yuv_on_mlu(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
    RET_CODE postprocess(float* model_output, VecObjBBox &bboxes, float aspect_ratio_x, float aspect_ratio_y);


    RET_CODE init_trackor();
    RET_CODE create_trackor(int uuid_cam=-1);
    RET_CODE trackprocess(TvaiImage &tvimage, VecObjBBox &ins);

    void visual(TvaiImage& tvimage, float* model_output);

    std::vector<CLS_TYPE> _cls_ = {CLS_TYPE::FALLING_OBJ, CLS_TYPE::FALLING_OBJ_UNCERTAIN};

    float m_predict_threshold = 0.25;
    // std::vector<TvaiImage> tvimage_buffers;
    std::map<int, std::vector<TvaiImage>> m_tviamge_buffers;
    std::map<int,std::shared_ptr<BoxTraceSet>> m_Trackors;
    const int m_max_executable_bboxes = 20;//限时单帧画面中移动物体候选框的数量, 强制进行截断. 防止cpu端处理过多内容.
    bool m_test_mod = false; //on: 没有轨迹后处理, 直接显示结果, 查看mod的效果.

    MLUNet_Ptr m_net = nullptr;
    int m_batchsize = 2;


public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
    std::vector<std::string> m_roots = {"/cambricon/model/"};
    std::map<ucloud::InitParam, std::string> m_models_startswith = {
        {InitParam::BASE_MODEL, "diffunet"},
    };        
    bool use_auto_model = false;    
};



/*******************************************************************************
UNet2DShiftSegment 基于分割算法的高空抛物 继承BaseModel
chaffee.chen@2021
*******************************************************************************/
// class UNet2DShiftSegment: public BaseModel{
// public:
//     UNet2DShiftSegment(){};
//     void set_batchsize(int m){m_batchsize=m;}
//     int get_batchsize(){return 1;}//change batchsize to from m_batchsize to 1 for display
//     RET_CODE init(const std::string &modelpath);
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     ~UNet2DShiftSegment();
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
//     RET_CODE run(BatchImageIN &batch_tvimages,VecObjBBox &bboxes);
//     RET_CODE run(TvaiImage &batch_tvimages,VecObjBBox &bboxes);
//     RET_CODE set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
// private:
//     RET_CODE push_back(TvaiImage &tvimage);
//     RET_CODE clear();
//     RET_CODE run_yuv_on_mlu(BatchImageIN &batch_tvimages, VecObjBBox &bboxes);
//     //后处理单元, 不移交指针
//     RET_CODE postprocess(float* model_output, VecObjBBox &bboxes, float aspect_ratio_x, float aspect_ratio_y);
//     RET_CODE postfilter(VecObjBBox &ins, VecObjBBox &outs);

//     RET_CODE init_trackor();
//     RET_CODE create_trackor(int uuid_cam=-1);
//     RET_CODE trackprocess(TvaiImage &tvimage, VecObjBBox &ins);

//     void visual(TvaiImage& tvimage, float* model_output);

//     std::vector<CLS_TYPE> _cls_ = {CLS_TYPE::FALLING_OBJ, CLS_TYPE::FALLING_OBJ_UNCERTAIN};

//     float m_predict_threshold = 0.4;
//     TvaiResolution m_maxTargetSize{0,0};
//     TvaiResolution m_minTargetSize{0,0};

//     VecRect m_pAoiRect;

//     int m_batchsize = 2;
//     std::vector<TvaiImage> tvimage_buffers;

//     std::map<int,std::shared_ptr<BoxTraceSet>> m_Trackors;
//     int m_max_executable_bboxes = 20;//限时单帧画面中移动物体候选框的数量, 强制进行截断. 防止cpu端处理过多内容.
//     bool m_test_mod = false; //on: 没有轨迹后处理, 直接显示结果, 查看mod的效果.

// public://开通自动模型推导, 根据根目录, 以及文件开头进行推导.
//     std::vector<std::string> m_roots = {"/cambricon/model/"};
//     std::string m_basemodel_startswith = "diffunet";
//     bool use_auto_model = false;    
// protected:
//     RET_CODE auto_model_file_search(std::map<InitParam, std::string> &modelpath);
// };


#endif