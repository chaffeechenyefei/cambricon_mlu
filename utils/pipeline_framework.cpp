#include "pipeline_framework.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <utility>
#include <vector>
#include <thread>
#include <atomic>

using namespace std;

// class printSink : public Sink<string, int> {
//  public:
//     string getName() override {
//         return "printSink";
//     }

//  protected:
//     int doExecute(string t) override {
//         return INT_MIN;
//     }
// };

// class intStringChannel : public Channel<int, string> {
//  public:
//     void init(std::string config) override {

//     }

//     string getName() override {
//         return "intStringChannel";
//     }

//     void startUp() override {

//     }

//     void shutDown() override {

//     }

//  protected:
//     string doExecute(int t) override {
//         return to_string(t + 100);
//     }
// };

// class IntSource : public Source<int, int> {
//  private:
//     int val = 0;
//  public:
//     void init(std::string config) override {
//         cout << "--------- " + getName() + " init --------- ";
//         val = 1;
//     }

//     string getName() override {
//         return "Int Source";
//     }

//     void startUp() override {
//         this->execute(val);
//     }

//  protected:
//     int doExecute(int) override {
//         return val + 1;
//     }
// };

// template<typename R, typename T>
// class pipeline : public LifeCycle {
//  private:
//     shared_ptr<Source<R, T>> source;

//  public:
//     void setSource(shared_ptr<Source<R, T>> component) {
//         source = component;
//     }

//     void init(std::string config) override {
//     }

//     void startUp() override {
//         assert(source.get() != nullptr);
//         source->startUp();
//     }

//     void shutDown() override {
//         source->shutDown();
//     }
// };

// int main() {
//     pipeline<int, int> p;
//     // source
//     auto is = make_shared<IntSource>();

//     // channel
//     auto isc = make_shared<intStringChannel>();

//     // sink
//     auto ps = make_shared<printSink>();

//     is->addDownStream(isc);
//     isc->addDownStream(ps);

//     // 设置 source
//     p.setSource(is);

//     // 启动
//     p.startUp();
// }
