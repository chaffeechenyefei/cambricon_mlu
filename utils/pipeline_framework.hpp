#ifndef _PIPELINE_FRAMEWORK_HPP_
#define _PIPELINE_FRAMEWORK_HPP_

#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <string>
#include <memory>
#include <utility>

namespace ucloud{
class LifeCycle {
 public:
    virtual void init(std::string config) = 0;

    virtual void startUp() = 0;

    virtual void shutDown() = 0;
};

template<typename T>
class Component : public LifeCycle {
 public:
    virtual std::string getName() = 0;

    virtual void execute(T t) = 0;
};

template<typename T, typename R>
class AbstractComponent : public Component<T> {
 private:
    std::unordered_set<std::shared_ptr<Component<R>>> down_stream;
 protected:
    const std::unordered_set<std::shared_ptr<Component<R>>> &getDownStream() {
        return down_stream;
    }

 protected:
    virtual R doExecute(T t) = 0;

 public:
    void addDownStream(std::shared_ptr<Component<R>> component) {
        down_stream.insert(component);
    }

    void init(std::string config) override {

    }

    void execute(T t) override {
        R r = doExecute(t);
        std::cout << this->getName() + "\treceive\t" << typeid(t).name() << "\t" << t << "\treturn\t" << typeid(r).name() << "\t" << r << std::endl;
        if (std::is_same<R, void>::value){
            return;
        }
        for (auto &&obj : getDownStream()) {
            obj->execute(r);
        }
    }

    void startUp() override {
        for (auto &&obj : this->getDownStream()) {
            obj->startUp();
        }
        std::cout << "------------------ " + this->getName() + " is starting ----------------------" << std::endl;
    }

    void shutDown() override {
        auto downStreams = this->getDownStream();
        for (auto &&obj : downStreams) {
            obj->shutDown();
        }
        std::cout << "------------------ " + this->getName() + " is starting ----------------------" << std::endl;
    }
};


template<typename T, typename R>
using Source = AbstractComponent<T, R>;

template<typename T, typename R>
using Channel = AbstractComponent<T, R>;

template<typename T, typename R>
using Sink = AbstractComponent<T, R>;

class printSink;

class intStringChannel;

}


#endif