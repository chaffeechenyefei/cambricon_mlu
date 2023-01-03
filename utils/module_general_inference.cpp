#include "module_general_inference.hpp"
#include "../inner_utils/inner_basic.hpp"

/*******************************************************************************
inner function
*******************************************************************************/
static TvaiRect scaleRect(TvaiRect &rect, float scale){
    TvaiRect output;
    float cx = rect.x + rect.width/2;
    float cy = rect.y + rect.height/2;
    output.width = rect.width*scale;
    output.height = rect.height*scale;
    output.x = cx - output.width/2;
    output.y = cy - output.height/2;
    return output;
}

/*******************************************************************************
GeneralInferenceSIMO
chaffee.chen@2022-10-27
*******************************************************************************/


RET_CODE GeneralInferenceSIMO::init(const std::string &modelpath){
    LOGI << "-> GeneralInferenceSIMO::init";
    bool pad_both_side = true;//双边留黑
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    //Self param
    return ret;
}

RET_CODE GeneralInferenceSIMO::init(std::map<InitParam, std::string> &modelpath){
    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    return init(modelpath[InitParam::BASE_MODEL]);
}


RET_CODE GeneralInferenceSIMO::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
        // ret = run_bgr_on_cpu(tvimage, bboxes, threshold);
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    }
    else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu(tvimage,bboxes, threshold);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE GeneralInferenceSIMO::run_yuv_on_mlu(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold){
    LOGI << "-> GeneralInferenceSIMO::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    for(int i = 0; i < bboxes.size(); i++){
        //根据设定的类型过滤bbox, 即仅对某些类别的bbox进一步进行分类
        TvaiRect roiRect = bboxes[i].rect;
        //EXPAND
        TvaiRect roiRectXL = scaleRect(roiRect, 1.0);
        float aspect_ratio = 1.0;
        float aX,aY;
        float** model_output = nullptr;
        {
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roiRectXL, aspect_ratio, aX , aY);
            if(ret!=RET_CODE::SUCCESS) return ret;
            model_output = m_net->general_mlu_infer();
        }
        if(ret!=RET_CODE::SUCCESS){ 
            m_net->cpu_free(model_output); 
            return ret; 
        }        
        //TODO post process 
        int num_tensors = m_net->_MO;
        Tensors tensors;
        tensors.num_tensors = num_tensors;
        tensors.tensors = reinterpret_cast<TvaiFeature*>(malloc(sizeof(TvaiFeature)*num_tensors));
        for(int j = 0; j < num_tensors; j++){
            tensors.tensors[j].pFeature = reinterpret_cast<unsigned char*>(model_output[j]);
            tensors.tensors[j].featureLen = m_net->m_outputShape[j].BatchDataCount();
        }
        bboxes[i].tensors = tensors;
    }
    LOGI << "<- GeneralInferenceSIMO::run_yuv_on_mlu";
    return RET_CODE::SUCCESS;
}