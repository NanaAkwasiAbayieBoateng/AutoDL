#ifndef _SINGLETON_H_
#define _SINGLETON_H_

#include <assert.h>
#include <pthread.h>
//for aexit
#include <stdlib.h>

namespace shouter {

// use c++11 style 
class Noncopyable {
public:
    Noncopyable() = default;
private:
    Noncopyable(const Noncopyable&) = delete; 
    Noncopyable& operator=(const Noncopyable&) = delete;
};

// use pthread_once to implement singleton.
template<typename T>
class Singleton: public Noncopyable
{
public:
    static T& instance() {
        pthread_once(&_ponce, &Singleton::init);
        return *_value;
    }

    static void init() {
        _value = new T();
        ::atexit(destroy);
    }
    
    static void destroy() {
        __attribute__ ((unused)) char T_must_be_complete_type[sizeof(T) == 0 ? -1 : 1];
        delete _value;
    }
private:
    static pthread_once_t _ponce;
    static T*             _value;
};

template<typename T>
pthread_once_t Singleton<T>::_ponce = PTHREAD_ONCE_INIT;

template<typename T>
T* Singleton<T>::_value = NULL;


// template class for threadlocal
template<typename T>
class ThreadLocal : public Noncopyable
{
public:

    ThreadLocal() {
        pthread_key_create(&_pkey, &ThreadLocal::destructor);
    }

    ~ThreadLocal() {
        pthread_key_delete(_pkey);
    }

    T& value() {
        T* perThreadValue = static_cast<T*>(pthread_getspecific(_pkey));
        if (!perThreadValue) {
            T* new_obj = new T();
            pthread_setspecific(_pkey, new_obj);
            perThreadValue = new_obj;
        }
        return *perThreadValue;
    }

    void set(T* new_obj){
        assert(pthread_getspecific(_pkey) == NULL);
        pthread_setspecific(_pkey, new_obj);
    }

    bool is_null(){
        return static_cast<T*>(pthread_getspecific(_pkey)) ==NULL;
    }

private:

    static void destructor(void *x) {
        T* obj = static_cast<T*>(x);
        __attribute__ ((unused))  char T_must_be_complete_type[sizeof(T) == 0 ? -1 : 1];
        delete obj;
    }

private:
    pthread_key_t _pkey;
};


template<typename T>
class ThreadLocalSingleton : Noncopyable
{
public:
    static T& instance() {
        if (!t_value_) {
            t_value_ = new T();
            _deleter.set(t_value_);
        }
        return *t_value_;
    }

    static T* ptr() {
        return t_value_;
    }

private:
    static void destructor(void* obj)
    {
        assert(obj == t_value_);
        __attribute__ ((unused))  char T_must_be_complete_type[sizeof(T) == 0 ? -1 : 1];
        delete t_value_;
        t_value_ = 0;
    }

    static __thread T* t_value_;
    static ThreadLocal<T> _deleter;
};

template<typename T>
__thread T* ThreadLocalSingleton<T>::t_value_ = 0;

template<typename T>
ThreadLocal<T> ThreadLocalSingleton<T>::_deleter;
}

#endif