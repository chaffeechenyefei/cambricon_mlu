#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>
using namespace std;


#include "config.hpp"
using namespace ucloud;

std::mutex cmutex;

void create_thread_for_yolo_task(int thread_id, TASKNAME taskid ,string datapath ,bool use_track=false){
    thread thread_source([=](){
        float threshold, nms_threshold;
        int use_batch = 0;
        AlgoAPIName apiName, apiSubName;
        task_parser(taskid, threshold, nms_threshold, apiName, apiSubName, use_batch);
        ucloud::TvaiResolution maxTraget={0,0};
        ucloud::TvaiResolution minTarget={0,0};
        std::vector<ucloud::TvaiRect> pRoi;
        double tm_cost = 0;
        int num_result = 0;
        AlgoAPISPtr ptrObjDetectHandle = ucloud::AICoreFactory::getAlgoAPI(apiName);
        cout << "yolo::loading model for thread #" << thread_id << endl;
        string modelfile = modelInfo[apiName];
        string trackmodelfile = modelInfo[AlgoAPIName::GENERAL_TRACKOR];
        ptrObjDetectHandle->init(modelfile);
        // if(!use_track)
        //     ptrObjDetectHandle->init(modelfile);
        // else
        //     ptrObjDetectHandle->init(modelfile, trackmodelfile);
        ptrObjDetectHandle->set_param(threshold, nms_threshold, maxTraget, minTarget, pRoi );

        ifstream infile;
        string filename = datapath + "list.txt";
        infile.open(filename, std::ios::in);
        string imgname;
        vector<string> vec_imgnames;
        vector<string> vec_imgbasenames;
        while(infile >> imgname){
            std::string imgname_full = datapath + imgname;
            vec_imgnames.push_back(imgname_full);
            vec_imgbasenames.push_back(imgname);
        }
        infile.close();
        for(int b = 0; b < vec_imgnames.size(); b++){
            if(b%100==0){
                std::lock_guard<std::mutex> lk(cmutex);
                cout << "#" << thread_id << ":" << b << ", " << num_result << " detected"<< endl;
            }
            int idx = b%(vec_imgnames.size());
            VecObjBBox bboxes;
            int width, height, stride;
            unsigned char* tmp = nullptr;
            tmp = readImg_to_NV21(vec_imgnames[idx],width, height, stride);
            int inputdata_sz = 3*width*height/2*sizeof(unsigned char);
            TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,width,tmp, inputdata_sz};
            auto start = chrono::system_clock::now();
            ptrObjDetectHandle->run(tvimage, bboxes);
            auto end = chrono::system_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
            tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
            num_result += bboxes.size();
            free(tmp);
            if(!bboxes.empty()){
                tmp = readImg(vec_imgnames[idx],width, height);
                drawImg(tmp, width, height, bboxes, false, true);
                string savename = datapath + "result/" + vec_imgbasenames[idx];
                writeImg(savename, tmp, width, height, true);
                freeImg(&tmp);
            }
            
        }//end for
        cout << "yolo::thread #" << thread_id << " : [" << num_result << "] cost(s) per image: " << tm_cost/vec_imgnames.size() << endl;
    });//end of thread
    thread_source.detach();
}

/**
 * ./test_case_img_list {datapath} {apiIdx} {use tracking or not}
 * {thread_num_for_face} {thread_num_for_yolo}
 */
int main(int argc, char* argv[]) {
    bool use_track = false;
    string datapath;
    TASKNAME taskid = TASKNAME::PED_CAR_NONCAR;
    if(argc>=2){
        string _tmp(argv[1]);
        datapath = _tmp;
    }
    if(argc >= 3){
        int _taskid = atoi(argv[2]);
        taskid = TASKNAME(_taskid);
    }
    if (argc >= 4){
        use_track = true;
        cout << "use tracking" << endl;
    } else {
        use_track = false;
        cout << "no tracking" << endl;
    }

    create_thread_for_yolo_task(0, taskid , datapath, use_track);

    pthread_exit(NULL);
    return 0;
}