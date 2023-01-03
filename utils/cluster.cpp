#include "cluster.hpp"
#include <math.h>
#include <algorithm>
using std::vector;

////////////////////////////////////////////////////////////////////////////////////////////////
//  RectCluster
////////////////////////////////////////////////////////////////////////////////////////////////
static inline float calc_diag_distance(TvaiRect rectA, TvaiRect rectB){
    float lx = std::min(rectA.x, rectB.x);
    float ly = std::min(rectA.y, rectB.y);
    float rx = std::max(rectA.x+rectA.width, rectB.x+rectB.width);
    float ry = std::max(rectA.y+rectA.height, rectB.y+rectB.height);
    float outerDiag = sqrtf((rx-lx)*(rx-lx)+(ry-ly)*(ry-ly)) + 1e-3;
    float innerDiag = sqrtf(1.0*(rectA.width*rectA.width+rectA.height*rectA.height)) + \
                      sqrtf(1.0*(rectB.width*rectB.width+rectB.height*rectB.height));
    return innerDiag/outerDiag;
}
bool RectCluster::insert(TvaiRect& rect){
    if(m_rects.empty()){
        update(rect);
        m_rects.push_back(rect);
        return true;
    } else {
        float diag_ratio = calc_diag_distance(rect, m_roi);
        if(diag_ratio > m_threshold ){
            update(rect);
            m_rects.push_back(rect);
            return true;
        }
    }
    return false;
}

void RectCluster::update(TvaiRect &rect){
    if(m_rects.empty()) m_roi = rect;
    int x1,y1,x2,y2;
    x1 = m_roi.x; y1 = m_roi.y;
    x2 = m_roi.x + m_roi.width; y2 = m_roi.y + m_roi.height;

    x1 = std::min(x1, rect.x);
    y1 = std::min(y1, rect.y);
    x2 = std::max(x2, rect.x+rect.width);
    y2 = std::max(y2, rect.y+rect.height);
    m_roi = TvaiRect{x1,y1, x2-x1, y2-y1};
}

TvaiRect RectCluster::update(){
    if(m_rects.empty()) m_roi = TvaiRect{0,0,0,0};
    int x1,y1,x2,y2;
    x1 = m_rects[0].x; y1 = m_rects[0].y;
    x2 = m_rects[0].x + m_rects[0].width; y2 = m_rects[0].y + m_rects[0].height;

    for (auto iter=m_rects.begin();iter!=m_rects.end();iter++){
        x1 = std::min(x1, iter->x);
        y1 = std::min(y1, iter->y);
        x2 = std::max(x2, iter->x+iter->width);
        y2 = std::max(y2, iter->y+iter->height);
    }
    return TvaiRect{x1,y1, x2-x1, y2-y1};
}

TvaiRect RectCluster::getROI(){
    if(m_rects.empty()) return TvaiRect{0,0,0,0};
    return m_roi;
}
////////////////////////////////////////////////////////////////////////////////////////////////
//  ClusterImageLevel
//////////////////////////////////////////////////////////////////////////////////////////////
void ClusterImageLevel::create_and_add_new_cluster(TvaiRect &rect){
    RectCluster tmp(m_threshold);
    tmp.insert(rect);
    m_clusters.push_back(tmp);
}

bool ClusterImageLevel::attach(TvaiRect &rect){
    for(auto iter=m_clusters.begin(); iter!=m_clusters.end(); iter++ ){
        if(iter->insert(rect)){
            return true;
        } //else continue
    }
    return false;
}

void ClusterImageLevel::attach(VecRect &rect){
    for(auto iter = rect.begin(); iter!=rect.end(); iter++ ){
        this->attach(*iter);
    }
}

void ClusterImageLevel::insert(TvaiRect &rect){
    if(m_clusters.empty()){
        create_and_add_new_cluster(rect);
    } else{
        bool flg_insert = false;
        for(auto iter=m_clusters.begin(); iter!=m_clusters.end(); iter++ ){
            if(iter->insert(rect)){
                flg_insert = true;
                break;
            } //else continue
        }
        if(!flg_insert){
            create_and_add_new_cluster(rect);
        }
    }
}

static bool sortRect(const TvaiRect &a, const TvaiRect &b ){
    return (a.x + a.y) < (b.x + b.y);
}
void ClusterImageLevel::insert(VecRect &rect){
    std::sort(rect.begin(), rect.end(), sortRect );
    for(auto iter = rect.begin(); iter!=rect.end(); iter++ ){
        this->insert(*iter);
    }
}

void ClusterImageLevel::merge(){
    VecRect tmp_roi;
    VecRect tmp_cand;
    //insert and filter
    if(m_clusters.empty()) { m_roi.clear(); return; }
    for(auto iter=m_clusters.begin(); iter!=m_clusters.end(); iter++){
        if(iter->getRectNum() <= 0) continue;
        TvaiRect roi = iter->getROI();
        if(iter->getRectNum() == 1) {
            tmp_cand.push_back(roi);
        } else {
            tmp_roi.push_back(roi);
        }
    }
    //merge
    if( tmp_roi.empty() ) { m_roi.clear(); return; }
    std::sort(tmp_cand.begin(), tmp_cand.end(), sortRect );
    ClusterImageLevel tmpHandle = ClusterImageLevel(m_threshold);
    tmpHandle.insert(tmp_roi);
    tmpHandle.attach(tmp_cand);
    m_roi = tmpHandle.getOriginROI();
    //TODO
}

VecRect ClusterImageLevel::getOriginROI(){
    VecRect rois;
    for(auto iter=m_clusters.begin(); iter!=m_clusters.end(); iter++ ){
        if(iter->getRectNum() >= 1)
            rois.push_back(iter->getROI());
    }
    return rois;
}

////////////////////////////////////////////////////////////////////////////////////////////////
//  ClusterImageSetLevel
//////////////////////////////////////////////////////////////////////////////////////////////
void ClusterImageSetLevel::insert(BatchBBoxIN &batch_bboxes){
    for(auto biter=batch_bboxes.begin(); biter!=batch_bboxes.end(); biter++){
        ClusterImageLevel imgCluster(m_threshold);
        VecRect t;
        for( auto iter=biter->begin(); iter!=biter->end(); iter++ ){
            t.push_back(iter->rect);
        }
        imgCluster.insert(t);
        imgCluster.merge();
        VecRect roi_ = imgCluster.getROI();
        m_roi_before_merge.insert(m_roi_before_merge.end(), roi_.begin(), roi_.end());
    }
}

void ClusterImageSetLevel::insert(std::vector<VecRect> &batch_rects){
    for(auto biter=batch_rects.begin(); biter!=batch_rects.end(); biter++){
        ClusterImageLevel imgCluster(m_threshold);
        imgCluster.insert(*biter);
        imgCluster.merge();
        VecRect roi_ = imgCluster.getROI();
        m_roi_before_merge.insert(m_roi_before_merge.end(), roi_.begin(), roi_.end());
    }    
}

void ClusterImageSetLevel::merge(){
    ClusterImageLevel tmphandle(m_threshold);
    tmphandle.insert(m_roi_before_merge);
    tmphandle.merge();
    m_roi = tmphandle.getROI();
}