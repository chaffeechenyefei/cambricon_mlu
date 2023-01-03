#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>
#include <stdio.h>
#include "libai_core.hpp"


using namespace std;
using namespace ucloud;

#ifndef VAR
#define VAR private
#endif

#include <math.h>
#include <algorithm>
#include <string.h>

namespace unitest
{
class RectCluster{
public:
    RectCluster(float threshold = 0.8):m_threshold(threshold){}
    ~RectCluster(){
        ucloud::VecRect().swap(m_rects);
        m_rects.clear();
    }
    /**
     * 每次insert都会出发增量update, 更新m_roi
     */
    bool insert(TvaiRect& rect);
    /**
     * 返回最大外接矩形 
     */
    TvaiRect getROI();
    int getRectNum(){return m_rects.size();}

TvaiRect update();//全量更新m_roi, check时候用
private:
    void update(TvaiRect &rect);//增量更新m_roi
    

VAR:
    TvaiRect m_roi;//m_rects的外接矩形
    ucloud::VecRect m_rects;
    float m_threshold = 0.8;//对角线距离比
};

/**
 * 一个图像含有多个Cluster
 */
class ClusterImageLevel{
public:
    ClusterImageLevel(float threshold = 0.8):m_threshold(threshold){}
    ~ClusterImageLevel(){
        m_clusters.clear();
    }
    void insert(TvaiRect &rect);
    void insert(VecRect &rect);
    //attach::只有距离接近才加入, 不新建cluster
    bool attach(TvaiRect &rect);
    void attach(VecRect &rect);
    /**
     * 将m_clusters进行一次聚合和过滤--> m_roi
     */
    void merge();
    int getROINum(){return m_roi.size();}
    std::vector<TvaiRect> getROI(){return m_roi;}
    VecRect getOriginROI();
private:
    void create_and_add_new_cluster(TvaiRect &rect);

VAR:
    float m_threshold = 0.8;
    std::vector<RectCluster> m_clusters; 
    std::vector<TvaiRect> m_roi;
};

class ClusterImageSetLevel{
public:
    ClusterImageSetLevel(float threshold = 0.8):m_threshold(threshold){}
    ~ClusterImageSetLevel(){m_roi_before_merge.clear(); m_roi.clear();}

    void insert(BatchBBoxIN &batch_bboxes);
    void merge();
    VecRect getROI(){return m_roi;}

VAR:
    float m_threshold = 0.8;
    std::vector<TvaiRect> m_roi;
    VecRect m_roi_before_merge;
};

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

void ClusterImageSetLevel::merge(){
    ClusterImageLevel tmphandle(m_threshold);
    tmphandle.insert(m_roi_before_merge);
    tmphandle.merge();
    m_roi = tmphandle.getROI();
}
} // namespace unitest

#include <time.h> 
int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL)); 
    int W = 640, H = 480;
    int _W = W/10, _H = H/10;
    int N = 5; int B = 5;
    //generate random rects

    BatchBBoxIN batch_bboxes;    
    VecObjBBox bboxes;
    for(int n = 0; n < N; n++ ){
        int x = rand()%W;
        int y = rand()%H;
        int w = rand()%_W;
        int h = rand()%_H;
        w = (x+w >= W ) ? W-x:w;
        h = (y+h >= H ) ? H-y:h;
        TvaiRect t = TvaiRect{x,y,w,h};
        BBox box;
        box.rect = t;
        bboxes.push_back(box);
    }
    batch_bboxes.push_back(bboxes);

    
    for(int b = 1; b < B; b++ ){
        VecObjBBox _bboxes;
        for(int n = 0; n < N; n++ ){
            int x = bboxes[n].rect.x + rand()%10;
            int y = bboxes[n].rect.y + rand()%10;
            int w = bboxes[n].rect.width + rand()%10;
            int h = bboxes[n].rect.height + rand()%10;
            w = (x+w >= W ) ? W-x:w;
            h = (y+h >= H ) ? H-y:h;
            TvaiRect t = TvaiRect{x,y,w,h};
            BBox box;
            box.rect = t;
            _bboxes.push_back(box);
        }
        batch_bboxes.push_back(_bboxes);
    }

    unsigned char* imgPtr = (unsigned char*)malloc(W*H*3*sizeof(unsigned char));
    unsigned char* imgPtr2 = (unsigned char*)malloc(W*H*3*sizeof(unsigned char));
    memset(imgPtr, 0, W*H*3*sizeof(unsigned char));
    memset(imgPtr2, 0, W*H*3*sizeof(unsigned char));
    for(int b=0;b<B;b++){
        drawImg(imgPtr,W,H,batch_bboxes[b]);
    }
    writeImg("result/001.jpg", imgPtr, W, H );

    unitest::ClusterImageSetLevel m;
    m.insert(batch_bboxes);
    m.merge();
    VecRect rois = m.getROI();
    cout << rois.size() << endl;
    VecObjBBox roiBoxes;
    for(int i = 0; i < rois.size(); i++ ){
        BBox box;
        box.rect = rois[i];
        roiBoxes.push_back(box);
    }
    drawImg(imgPtr, W, H, roiBoxes);
    writeImg("result/001.jpg", imgPtr, W, H);
    free(imgPtr);
    free(imgPtr2);
}