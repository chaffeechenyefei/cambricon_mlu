#include "config.hpp"
#include "libai_core.hpp"

#include <stdlib.h>
#include <time.h> 

using namespace ucloud;
// using namespace cv;

using ucloud::VecObjBBox;
using ucloud::BatchBBoxIN;
using ucloud::BBox;
using ucloud::TvaiRect;
using std::vector;

/////////////////////////////////////////////////////////////////////////////////////
// HEADER
/////////////////////////////////////////////////////////////////////////////////////
class A{
public:
    A(){printf("A construct\n");}
    virtual ~A(){printf("A deconstruct\n");}
    virtual void init(){
        printf("A::init()\n");
        func();
        return;
    }
    virtual void print(){
        printf("A::print %s\n", m_x.c_str());
    }
protected:
    virtual void func(){
        printf("A::func(), %s\n", m_x.c_str());
        return;
    }
private:
    std::string m_x = "a";
};
class B: public A{
public:
    B(){printf("B construct\n");}
    virtual ~B(){printf("B deconstruct\n");}
protected:
    virtual void func(){
        printf("B::func(), %s\n", m_x.c_str());
        return;
    }
private:
    std::string m_x = "b";    
};

/////////////////////////////////////////////////////////////////////////////////////
// MAIN
/////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
    A* ptr = new B();
    ptr->init();

    ptr->print();
    delete ptr;
    printf("\033[31mThis text is red\033[0m\n");
}


/////////////////////////////////////////////////////////////////////////////////////
// IMP
/////////////////////////////////////////////////////////////////////////////////////
