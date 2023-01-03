#include "module.hpp"
#include <stdlib.h>
#include <iostream>

using namespace std;
bool MemCtrl::check_leakying(){
    std::lock_guard<std::mutex> lk(_mutex);
    bool ret = true;
    if(!_memPool.empty()){
        for(auto iter = _memPool.begin(); iter!= _memPool.end(); iter++){
            cout << iter->second.file << ":" << iter->second.line << " leaking" << endl;
        }
        ret = false;
    }
    return ret;
}


bool MemCtrl::check_and_release(){
    std::lock_guard<std::mutex> lk(_mutex);
    bool ret = true;
    if(!_memPool.empty()){
        for(auto iter = _memPool.begin(); iter!= _memPool.end(); iter++){
            cout << iter->second.file << ":" << iter->second.line << " leaking" << endl;
            free( reinterpret_cast<void*>(iter->first) );
        }
        ret = false;
    }
    return ret;
}

bool MemCtrl::insert(void* memPtr, string file, int line){
    cout << file << ":" << line << endl;
    std::lock_guard<std::mutex> lk(_mutex);
    if (memPtr!=nullptr){
        auto iter = _memPool.find( (size_t)memPtr);
        if(iter==_memPool.end()){
            //insert
            MemInfo a{file,line};
            _memPool.insert(std::pair<size_t,MemInfo>((size_t)memPtr, a));
            return true;
        } else {
            cout << file << ":" << line << " mallocing a unfree space" << endl;
            return false;
        }
    }
    return true;
}

bool MemCtrl::del(void* memPtr){
    std::lock_guard<std::mutex> lk(_mutex);
    if(memPtr==nullptr)
        return true;
    auto iter = _memPool.find((size_t)memPtr);
    if (iter==_memPool.end())
        return false;
    else{
        _memPool.erase(iter);
        return true;
    }
}

MemCtrl::MemCtrl(){
    // cout << "memory leaky detection" << endl;
}

MemCtrl::~MemCtrl(){
    // cout << "releasing" << endl;
    check_and_release();
}

static MemCtrl memCtrl;

void *debug_malloc(size_t size,std::string file, int line){
    void* ptr = malloc(size);
    memCtrl.insert(ptr, file, line);
    return ptr;
}

void debug_free(void* ptr){
    if(ptr!=nullptr){
        memCtrl.del(ptr);
        free(ptr);
    }
}