#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>
#include "libai_core.hpp"
#include <string.h>

using namespace std;
using namespace ucloud;

typedef struct APIConfig{
    AlgoAPIName apiName;
    std::string modelName;
    APIConfig(AlgoAPIName _apiName, std::string _modelName):apiName(_apiName),modelName(_modelName){};
}APIConfig;

#ifdef  MLU220
vector<APIConfig> config = {
  APIConfig(AlgoAPIName::FACE_DETECTOR, "retinaface_736x416_mlu220.cambricon"),
  APIConfig(AlgoAPIName::FACE_EXTRACTOR, "resnet101_112x112_mlu220_fp16.cambricon"),
  APIConfig(AlgoAPIName::GENERAL_DETECTOR, "yolov5s-conv-9_736x416_mlu220_bs1c1_fp16.cambricon"),
  APIConfig(AlgoAPIName::SKELETON_DETECTOR, "pose_resnet_50_256x192_mlu220_bs1c1_fp16.cambricon")
};
#else//mlu270
vector<APIConfig> config = {
  APIConfig(AlgoAPIName::FACE_DETECTOR, "/project/workspace/samples/mlu_videofacerec/weights/face_det/retinaface_736x416_mlu270.cambricon"),
  APIConfig(AlgoAPIName::FACE_EXTRACTOR, "/project/workspace/samples/mlu_videofacerec/weights/face_rec/resnet101_mlu270.cambricon"),
  APIConfig(AlgoAPIName::GENERAL_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-9_736x416_mlu270_bs1c1_fp16.cambricon"),
  APIConfig(AlgoAPIName::SKELETON_DETECTOR, "/project/workspace/samples/deep-high-resolution-net/mlu270/pose_resnet_50_256x192_mlu270_bs1c1_fp16.cambricon")
};
#endif

/**
 * ./test_general_multithread {thread_num} {apiIdx} {use fake yuv if given}
 * 
 */
int main(int argc, char* argv[]) {
    int apiIdx = 2;
    bool use_fake_yuv = false;
    
    cout << "Multi Thread Test" << endl;
    int thread_num = 1;
    int N = 1;
    string modelfile;
    if (argc >= 2){
        string _thread_num(argv[1]);
        cout << "Create #" << _thread_num << " threads" << endl;
        thread_num = atoi(_thread_num.c_str());
    } else {
    cout << "Create #" << thread_num << " threads" << endl;
    }
    if (argc >= 3){
        apiIdx = atoi(argv[2]);
    }
    cout << "Test AlgoAPIName["<< apiIdx << "] algorithm" << endl;
    // if (argc >= 4){
    //     use_fake_yuv = true;
    // }

    modelfile = config[apiIdx].modelName;
    std::cout << modelfile << std::endl;

    for (int i = 0; i < thread_num; i++ ){
        cout << "#Loop:" << i << endl;
        thread thread_source([=](){
        double tm_cost = 0;
        int tm_count = 0;
        AlgoAPISPtr ptrObjDetectHandle = ucloud::AICoreFactory::getAlgoAPI(config[apiIdx].apiName);
        cout << "loading model for thread #" << i << endl;
        ptrObjDetectHandle->init(modelfile);
        //names: [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ]
        // CLS_TYPE yolov5s_conv_9[] = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
        // ptrObjDetectHandle->set_output_cls_order(yolov5s_conv_9, 9);
        std::vector<CLS_TYPE> class_to_be_detected;
        ptrObjDetectHandle->get_class_type(class_to_be_detected);
        std::cout << "Class will be detected: ";
        for (int i = 0; i < class_to_be_detected.size(); i++ ){
            cout << class_to_be_detected[i] << ", ";
        }
        std::cout << std::endl;
        std::string datapath = "image_cache/";
        std::vector<std::string> rootpath = {datapath};
        for( int n = 0; n < rootpath.size() ; n++ ){//rootpath
            std::ifstream infile;
            std::string filename = rootpath[n] + "list.txt";
            int cnt = 0;
            while (cnt<60){//while
                infile.open(filename, std::ios::in);
                std::string imgname;
                std::vector<string> vec_imgnames;
                bool use_yuv = false;
                while(infile >> imgname){
                    if(imgname.find(".yuv")>=0 && imgname.find(".yuv")!=std::string::npos){
                        use_yuv = true;
                    }
                    std::string imgname_full = rootpath[n] + imgname;
                    vec_imgnames.push_back(imgname_full);
                }

                ucloud::TvaiImageFormat fmt = (!use_yuv)? ucloud::TvaiImageFormat::TVAI_IMAGE_FORMAT_BGR : ucloud::TvaiImageFormat::TVAI_IMAGE_FORMAT_NV21;
                
                for(int b = 0; b < vec_imgnames.size(); b++){
                    std::vector<unsigned char*> batch_data;
                    std::vector<TvaiImage> batch_in;
                    std::vector<VecObjBBox> batch_out;
                    VecObjBBox bboxes;
                    for(int bs=0; bs<N; bs++){
                        int idx = (b + bs)%vec_imgnames.size();
                        int width, height, stride;
                        unsigned char* tmp = nullptr;
                        if(use_yuv){
                            width = 1920; height = 1080;
                            tmp = yuv_reader(vec_imgnames[idx]);
                        } else {
                            if(use_fake_yuv)
                                tmp = readImg_to_NV21(vec_imgnames[idx], width, height,stride);
                            else
                                tmp = readImg(vec_imgnames[idx], width, height);
                        }
                        batch_data.push_back(tmp);
                        int inputdata_sz = (!use_yuv) ? (width*height*3*sizeof(unsigned char)) :(3*width*height/2*sizeof(unsigned char));
                        TvaiImage tvimage{fmt,width,height,width,tmp, inputdata_sz};
                        batch_in.push_back(tvimage);
                    }

                    auto start = chrono::system_clock::now();
                    if (N==1)
                        ptrObjDetectHandle->run(batch_in[0], bboxes);
                    auto end = chrono::system_clock::now();
                    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
                    tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
                    tm_count++;
                    cnt++;

                    for(int bs=0; bs<N; bs++){
                        free(batch_data[bs]);
                    }
                }
            infile.close();    
            }//while
        }//rootpath
        cout << "thread #" << i << " : cost(s) per image: " << tm_cost/tm_count/N << endl;

        });//thread lambda end

        thread_source.detach();
    }
    pthread_exit(NULL);
    return 0;
}