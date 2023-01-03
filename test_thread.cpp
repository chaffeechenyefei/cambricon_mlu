#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>
#include "libai_core.hpp"
#include "config.hpp"

using namespace std;
using namespace ucloud;

int num_times_each_thread = 10000;
float total_fps = 0;
std::mutex cmutex;


void create_thread_for_task(int thread_id, TASKNAME task_id, string datapath, int total_thread_num, int W, int H){
    thread thread_source([=](){
        AInfo detailTime = {0,0,0};
        int detail_cnt = 0;

        double tm_cost = 0;
        int num_result = 0;
        float threshold, nms_threshold;
        AlgoAPIName apiName;
        InitParamMap modelConfig;
        RET_CODE ret = RET_CODE::FAILED;
        int use_batch = 1;
        task_parser(task_id, threshold, nms_threshold, apiName, modelConfig, use_batch);
        AlgoAPISPtr ptrObjDetectHandle = ucloud::AICoreFactory::getAlgoAPI(apiName);
        {
            std::lock_guard<std::mutex> lk(cmutex);
            printf("tid[%d] loading model\n",thread_id );
        }
        ret = ptrObjDetectHandle->init(modelConfig);
        if(ret!=RET_CODE::SUCCESS){
            std::lock_guard<std::mutex> lk(cmutex);
            printf("tid[%d] init failed return [%d]\n",thread_id, ret );
            return;
        }

        ifstream infile;
        string filename = datapath + "/list.txt";
        infile.open(filename, std::ios::in);
        string imgname;
        vector<string> vec_imgnames;
        while(infile >> imgname){
            std::string imgname_full = datapath + "/" + imgname;
            vec_imgnames.push_back(imgname_full);
        }
        infile.close();
        int disp_step = num_times_each_thread/10;
        for(int b = 0; b < num_times_each_thread; b++){
            if(b%disp_step==1){
                std::lock_guard<std::mutex> lk(cmutex);
                printf("tid[%03d][%.2f%%][%07d/%07d]::avg time %.3fs, %d detected\n", 
                thread_id, (float(b))/num_times_each_thread*100, b, num_times_each_thread,
                tm_cost/b, num_result
                );
            }
            int idx = b%(vec_imgnames.size());
            VecObjBBox bboxes;
            int width, height, stride;
            unsigned char* tmp = nullptr;
            tmp = readImg_to_NV21(vec_imgnames[idx], W, H ,width, height, stride);
            int inputdata_sz = 3*stride*height/2*sizeof(unsigned char);
            TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,tmp, inputdata_sz};
            if(task_id==TASKNAME::FACE_FEAT){
                BBox box;
                box.rect = {0,0,W,H};
                box.objtype = CLS_TYPE::FACE;
                bboxes.push_back(box);
            }
            auto start = chrono::system_clock::now();
            ptrObjDetectHandle->run(tvimage, bboxes);
            auto end = chrono::system_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
            tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
            num_result += bboxes.size();

            if(!bboxes.empty()){
                AInfo _tmp_ = bboxes[0].tmInfo;
                detailTime.preprocess_time += _tmp_.preprocess_time;
                detailTime.npu_inference_time += _tmp_.npu_inference_time;
                detailTime.postprocess_time += _tmp_.postprocess_time;
                detail_cnt++;
            }
            releaseVecObjBBox(bboxes);

            free(tmp);
        }//end for
        {
            std::lock_guard<std::mutex> lk(cmutex);
            printf("** tid[%03d][100%%][%07d]::avg time %.3fs, %d detected\n", 
                thread_id,  num_times_each_thread,
                tm_cost/num_times_each_thread, num_result);
            total_fps += num_times_each_thread/tm_cost;
            printf("-------------------------------------------------------\n");
            printf("%d threads, total fps = %.1f, avg fps per thread = %.1f\n", total_thread_num, total_fps, total_fps/total_thread_num);
            printf("Average preprocess time = %.1fms, npu inference time = %.1fms, postprocess time = %.1fms\n", 
            1000*detailTime.preprocess_time/detail_cnt,1000*detailTime.npu_inference_time/detail_cnt,1000*detailTime.postprocess_time/detail_cnt);
            printf("-------------------------------------------------------\n");
        }
        
    });//end of thread
    thread_source.detach();
}

/**
 * ./test_thread {datapath} {task_id} {img_size} {thread_num}
 */
int main(int argc, char* argv[]) {
    cout << "Multi Thread Test" << endl;
    int thread_num = 1;
    std::string datapath = "car_data/";
    TASKNAME taskid = TASKNAME::PED_CAR_NONCARV2;
    int W{1920}, H{1080};

    for(int i=0; i< argc; i++){
        std::string keycmd = std::string(argv[i]);
        if(keycmd=="-help"){
            printf("-data(%s) -task(%d) -thread(%d) -loop(%d) -w(%d) -h(%d)\n",
            datapath.c_str(), taskid, thread_num, num_times_each_thread, W, H
            );
            return 0;
        }
    }

    for(int i=0; i< argc - 1; i++){
        std::string keycmd = std::string(argv[i]);
        if(keycmd=="-data"){
            datapath = std::string(argv[i+1]);
        }
        else if(keycmd=="-task"){
            taskid = TASKNAME(atoi(argv[i+1]));
        }
        else if(keycmd=="-thread"){
            thread_num = atoi(argv[i+1]);
        }
        else if( keycmd=="-loop"){
            num_times_each_thread = atoi(argv[i+1]);
        }
        else if( keycmd=="-w"){
            W = atoi(argv[i+1]);
        }
        else if( keycmd=="-h" ){
            H = atoi(argv[i+1]);
        }
    }
    printf("** datapath = %s\n", datapath.c_str());
    printf("**taskid = %d\n", taskid);
    printf("** thread num = %d\n", thread_num);
    printf("** thread loop times = %d\n", num_times_each_thread);    
    printf("** input width %d, height %d\n", W, H);    


    for (int i = 0; i < thread_num ; i++ ){
        create_thread_for_task(i, taskid, datapath, thread_num, W, H);
    }

    pthread_exit(NULL);
    return 0;
}