#include "framework_detection.hpp"


/*******************************************************************************
AnyDetection + DeepSort
chaffee.chen@2022-09-30
*******************************************************************************/
AnyDetectionV4DeepSort::AnyDetectionV4DeepSort(){
    // m_detector = std::make_shared<YoloDetectionV4>();
    LOGI << "-> AnyDetectionV4DeepSort::AnyDetectionV4DeepSort()";
    m_trackor = std::make_shared<DeepSortPool>(m_fps, m_nn_buf, m_max_cosine_dist);
    m_trackFeatExtractor = std::make_shared<ObjFeatureExtractionV2>();
    LOGI << "<- AnyDetectionV4DeepSort::AnyDetectionV4DeepSort()";
}

RET_CODE AnyDetectionV4DeepSort::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);
    DEEPSORTPARM track_param = {threshold};
    RET_CODE ret = m_detector->run(tvimage, bboxes, threshold, nms_threshold);
    if(ret!=RET_CODE::SUCCESS) {
        printf("Err: m_detector return [%d]\n", ret);
        return ret;
    }
#ifdef TIMING    
    m_Tk.start();
#endif
    if(m_trackor && m_trackFeatExtractor){
        ret = m_trackFeatExtractor->run(tvimage, bboxes);
        if(ret!=RET_CODE::SUCCESS){
            printf("ERR: m_trackFeatExtractor return [%d]\n", ret);
            return ret;
        }
        m_trackor->update(tvimage, bboxes, track_param);
    }
#ifdef TIMING    
    m_Tk.end("tracking");
#endif  
    return RET_CODE::SUCCESS;
}

RET_CODE AnyDetectionV4DeepSort::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> AnyDetectionV4DeepSort::init";
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) {
            printf("auto_model_file_search failed, return %d\n",ret);
            return ret;
        }
    }
    RET_CODE ret = m_detector->init(modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;
    if(modelpath.find(InitParam::TRACK_MODEL)!=modelpath.end()){
        ret = m_trackFeatExtractor->init(modelpath[InitParam::TRACK_MODEL]);
        if(ret!=RET_CODE::SUCCESS) return ret;
    } else{
        m_trackFeatExtractor = nullptr;
    }
    LOGI << "<- AnyDetectionV4DeepSort::init";
    return ret;
}

RET_CODE AnyDetectionV4DeepSort::init(std::map<InitParam, WeightData> &modelpath){
    LOGI << "-> AnyDetectionV4DeepSort::init";
    RET_CODE ret = m_detector->init(modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;
    if(modelpath.find(InitParam::TRACK_MODEL)!=modelpath.end()){
        ret = m_trackFeatExtractor->init(modelpath[InitParam::TRACK_MODEL]);
        if(ret!=RET_CODE::SUCCESS) return ret;
    } else{
        m_trackFeatExtractor = nullptr;
    }
    LOGI << "<- AnyDetectionV4DeepSort::init";
    return ret;
}

RET_CODE AnyDetectionV4DeepSort::init(const std::string &modelpath){
    RET_CODE ret = m_detector->init(modelpath);
    if(ret!=RET_CODE::SUCCESS) return ret;
    m_trackFeatExtractor = nullptr;
    return ret;
}

RET_CODE AnyDetectionV4DeepSort::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    return m_detector->get_class_type(valid_clss);
}

RET_CODE AnyDetectionV4DeepSort::set_detector(AlgoAPI* ptr){
    m_detector.reset(ptr);
    return RET_CODE::SUCCESS;
}

RET_CODE AnyDetectionV4DeepSort::set_output_cls_order(std::vector<CLS_TYPE> &output_clss){
    return m_detector->set_output_cls_order(output_clss);
}

float AnyDetectionV4DeepSort::clip_threshold(float x){
    if(x < 0) return m_default_threshold;
    if(x > 1) return m_default_threshold;
    return x;
}
float AnyDetectionV4DeepSort::clip_nms_threshold(float x){
    if(x < 0) return m_default_nms_threshold;
    if(x > 1) return m_default_nms_threshold;
    return x;
}

/*******************************************************************************
AnyDetection + ByteTrack
use set_trackor to switch differenct version of ByteTrack
chaffee.chen@2022-09-30
*******************************************************************************/
AnyDetectionV4ByteTrack::AnyDetectionV4ByteTrack(){
    // m_detector = std::make_shared<YoloDetectionV4>();
    m_trackor = std::make_shared<ByteTrackOriginPool>(m_fps,m_nn_buf);
}

RET_CODE AnyDetectionV4ByteTrack::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);
    BYTETRACKPARM track_param = {threshold, threshold+0.1f};
    RET_CODE ret = m_detector->run(tvimage, bboxes, threshold, nms_threshold);
#ifdef TIMING    
    m_Tk.start();
#endif
    if(m_trackor){
        m_trackor->update(tvimage, bboxes, track_param);
    }
#ifdef TIMING    
    m_Tk.end("tracking");
#endif
    return RET_CODE::SUCCESS;
}

RET_CODE AnyDetectionV4ByteTrack::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> AnyDetectionV4ByteTrack::init";
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) {
            printf("auto_model_file_search failed, return %d\n",ret);
            return ret;
        }
    }
    return m_detector->init(modelpath);
}

RET_CODE AnyDetectionV4ByteTrack::init(std::map<InitParam, WeightData> &modelpath){
    LOGI << "-> AnyDetectionV4ByteTrack::init";
    return m_detector->init(modelpath);
}

RET_CODE AnyDetectionV4ByteTrack::init(const std::string &modelpath){
    return m_detector->init(modelpath);
}

RET_CODE AnyDetectionV4ByteTrack::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    return m_detector->get_class_type(valid_clss);
}

RET_CODE AnyDetectionV4ByteTrack::set_detector(AlgoAPI* ptr){
    m_detector.reset(ptr);
    return RET_CODE::SUCCESS;
}

RET_CODE AnyDetectionV4ByteTrack::set_trackor(TRACKMETHOD trackmethod){
    switch (trackmethod)
    {
    case TRACKMETHOD::BYTETRACK_ORIGIN :
        m_trackor = std::make_shared<ByteTrackOriginPool>(m_fps,m_nn_buf);
        break;
    case TRACKMETHOD::BYTETRACK_NO_REID :
        m_trackor = std::make_shared<ByteTrackNoReIDPool>(m_fps,m_nn_buf);
        break;        
    default:
        printf("unsupported tracking method, ByteTrackOriginPool will be used\n");
        m_trackor = std::make_shared<ByteTrackOriginPool>(m_fps,m_nn_buf);
        break;
    }
    return RET_CODE::SUCCESS;
}

RET_CODE AnyDetectionV4ByteTrack::set_output_cls_order(std::vector<CLS_TYPE> &output_clss){
    return m_detector->set_output_cls_order(output_clss);
}

float AnyDetectionV4ByteTrack::clip_threshold(float x){
    if(x < 0) return m_default_threshold;
    if(x > 1) return m_default_threshold;
    return x;
}
float AnyDetectionV4ByteTrack::clip_nms_threshold(float x){
    if(x < 0) return m_default_nms_threshold;
    if(x > 1) return m_default_nms_threshold;
    return x;
}


/*******************************************************************************
PipelineNaive
chaffee.chen@2022-10-09
*******************************************************************************/
RET_CODE PipelineNaive::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    std::set<CLS_TYPE> clss;
    for(auto&& handle: m_handles){
        std::vector<CLS_TYPE> vec_tmp;
        handle->get_class_type(vec_tmp);
        clss.insert(vec_tmp.begin(),vec_tmp.end());
    }
    for(auto&& cls: clss){
        valid_clss.push_back(cls);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE PipelineNaive::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::SUCCESS;
    for(int i=0; i < m_handles.size(); i++){
        if(i==unfixed_thresholds_index)
            ret = m_handles[i]->run(tvimage,bboxes,threshold, nms_threshold);
        else
            ret = m_handles[i]->run(tvimage,bboxes,m_thresholds[i], m_nms_thresholds[i]);
        if(ret!=RET_CODE::SUCCESS){
            printf("ERR[%s][%d]::PipelineNaive::run [%d]th handle return [%d]\n",__FILE__, __LINE__, i, ret);
            return ret;
        }
    }
    return ret;
}