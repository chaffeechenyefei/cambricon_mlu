#ifndef _CLUSTER_HPP_
#define _CLUSTER_HPP_
#include "module_base.hpp"

#ifndef VAR
#define VAR private
#endif

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
    void insert(std::vector<VecRect> &batch_rects);
    void merge();
    VecRect getROI(){return m_roi;}

VAR:
    float m_threshold = 0.8;
    std::vector<TvaiRect> m_roi;
    VecRect m_roi_before_merge;
};


#endif