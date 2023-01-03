/**
 * This main funtion is used to extract face features from face database where all the face image has been cropped.
 * For image name xxx.jpg, its feature file will be named xxx.jpg.txt, which will be stored in the same directory of image.
 * Feature comparison is done with python script for simplicity.
 * Chaffee@20210721
 * */
#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include "libai_core.hpp"

using namespace ucloud;

#define LOGI LOG(INFO) 

void write_vector_to_file(float* feature, int dims, std::string filename){
    std::ofstream ofile;
    ofile.open(filename, std::ios::out | std::ios::trunc);
    for(int i=0 ; i < dims; i++ ){
        ofile << feature[i] << std::endl;
    }
    ofile.close();
}


int main(int argc, char* argv[]) {
    FaceExtractor* ptrExtractorHandle = new FaceExtractor();
#ifdef MLU220
  LOG(INFO) << "loading MLU220_EDGE files";
  ptrExtractorHandle->init("resnet101_mlu220edge.cambricon");
#else
  ptrExtractorHandle->init("resnet101_mlu270.cambricon");
#endif
    LOGI << "model loaded...";

    std::string datapath = "hegui_faces/";
    std::string gallerypath = datapath + "register/";
    std::string probepath = datapath + "test/";

    std::vector<std::string> rootpath = {gallerypath,probepath};

    for( int n = 0; n < rootpath.size() ; n++ ){
        std::ifstream infile;
        std::string filename = rootpath[n] + "list.txt";
        infile.open(filename, std::ios::in);
        std::string imgname;
        while(infile >> imgname){
            std::cout << imgname << std::endl;
            std::string imgname_full = rootpath[n] + imgname;
            std::string resultname_full = imgname_full + "1.txt";
            int width, height;
            unsigned char* imgPtr = ucloud::readImg(imgname_full, width, height);
            if (imgPtr==nullptr){
                std::cout << imgname << " Not Found" << std::endl;
                continue;
            }
            TvaiFeature feat;
            ptrExtractorHandle->run(imgPtr, width, height, feat);
            float* _feat = reinterpret_cast<float*>(feat.pFeature);
            write_vector_to_file( _feat, feat.featureLen/sizeof(float), resultname_full );
            
            std::string resultname_full2 = imgname_full + "2.txt";
            VecFeat feat2;
            FaceInfo bbox;
            bbox.rect.x = 0; bbox.rect.y = 0;
            bbox.rect.height = height; bbox.rect.width = width;
            VecFaceBBox bboxes;
            bboxes.push_back(bbox);
            ptrExtractorHandle->run(imgPtr, width, height, bboxes,feat2);
            _feat = reinterpret_cast<float*>(feat2[0].pFeature);
            write_vector_to_file( _feat, feat2[0].featureLen/sizeof(float), resultname_full2 );

            releaseTvaiFeature(feat2);
            free(feat.pFeature);
            ucloud::freeImg(&imgPtr);
        }
        infile.close();
    }
    delete ptrExtractorHandle;
}