#ifndef _MODULE_HPP_
#define _MODULE_HPP_
#include <stddef.h>
#include <string>
#include <map>
#include <mutex>
#include <vector>

// #define NSLogRect(rect) NSLog(@"%s x:%.4f, y:%.4f, w:%.4f, h:%.4f", #rect, rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
// #define NSLogSize(size) NSLog(@"%s w:%.4f, h:%.4f", #size, size.width, size.height)
// #define NSLogPoint(point) NSLog(@"%s x:%.4f, y:%.4f", #point, point.x, point.y)

////////////////////////////////////////////////////////////////////////////////
/**
 * 内存泄漏检测模组, 似乎存在BUG
 **/
typedef struct _MemInfo {
    std::string file;
    int line;
} MemInfo;

class MemCtrl{
public:
    MemCtrl();
    ~MemCtrl();
    bool check_and_release();
    bool check_leakying();
    bool insert(void* memPtr, std::string file, int line);
    bool del(void* memPtr);

protected:
    std::map<size_t,MemInfo> _memPool;
    std::mutex _mutex;
};


void *debug_malloc(size_t size, std::string file, int line);
void debug_free(void* ptr);

// #ifdef DEBUG
// #define malloc(s) debug_malloc( (s), __FILE__, __LINE__)
// #define free(s) debug_free((s));
// #endif
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

typedef void (*FreePtr)(void*);
/**
 * 适用于分配好空间后的指针,且不能在外部自行释放,否则存储的是野指针
 */
template<typename T>
class TemporalPtrPool{
    typedef T (*FreePtrT)(void*);
public:
    TemporalPtrPool(){}
    ~TemporalPtrPool(){tpp_free();}
    void add(void* ptr_addr){
        m_ptr_addr.push_back(ptr_addr);
        m_ptr_func.push_back(free);
    }
    void add(void* ptr_addr, FreePtrT free_func){
        m_ptr_addr.push_back(ptr_addr);
        m_ptr_func.push_back(free_func);
    }
    void tpp_free(){
        for(int i = 0; i < m_ptr_addr.size(); i++){
            m_ptr_func[i](m_ptr_addr[i]);
            m_ptr_addr[i] = nullptr;
        }
    }

private:
    std::vector<void*> m_ptr_addr;
    std::vector<FreePtrT> m_ptr_func;
};

#endif