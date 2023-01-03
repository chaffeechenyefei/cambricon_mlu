#include "module_track.hpp"

using namespace std;

void DeepSortPool::add_trackor(cam_class_uuid uuid){
    std::lock_guard<std::mutex> lk(m_mutex);
    if(m_trackors.find(uuid)==m_trackors.end()){
        //没有找到，新建一个
        edk::FeatureMatchTrack *track = new edk::FeatureMatchTrack();
        track->SetParams(m_max_cosine_dist, m_nn_budget, m_max_iou_dist, m_fps*2, m_n_init);
        DeepSort_Ptr m_trackor(track);
        m_trackors[uuid] = m_trackor;
    }
}

void DeepSortPool::update(TvaiImage &tvimage, VecObjBBox &bboxIN, DEEPSORTPARM &params){
    int cam_uuid = tvimage.uuid_cam;
    float imgW = tvimage.width;
    float imgH = tvimage.height;
    cam_class_uuid uuid = get_cam_class_uuid(cam_uuid,  CLS_TYPE::UNKNOWN);
    add_trackor(uuid);

    std::vector<edk::DetectObject> in, out;
    for(auto &&box: bboxIN){//loop class
        edk::DetectObject obj;
        float x = CLIP(box.rect.x/imgW);
        float y = CLIP(box.rect.y/imgH);
        float w = CLIP(box.rect.width/imgW);
        float h = CLIP(box.rect.height/imgH);
        w = (x + w > 1.0) ? (1.0 - x) : w;
        h = (y + h > 1.0) ? (1.0 - y) : h;
        obj.label = box.objtype;
        obj.score = box.confidence;
        obj.bbox.x = x;
        obj.bbox.y = y;
        obj.bbox.width = w;
        obj.bbox.height = h;
        obj.feature = box.trackfeat;
        in.push_back(obj);
    }
    edk::TrackFrame tframe;
    m_trackors[uuid]->UpdateFrame(tframe, in , &out);
    LOGI << "after track " << out.size();
    for(int i = 0; i < out.size() ; i++){
        bboxIN[out[i].detect_id].track_id = out[i].track_id;
    }
}