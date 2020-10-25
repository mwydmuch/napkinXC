/*
 * ThreadPool:
 * Copyright (c) 2012 by Jakob Progsch, Vaclav Zeman
 * All rights reserved.
 * https://github.com/progschj/ThreadPool
 *
 * ThreadSet:
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>


// Simple pool of threads
class ThreadPool {
public:
    ThreadPool(size_t);
    ~ThreadPool();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    void stopAll();

private:
    // Keeps track of threads
    std::vector<std::thread> workers;

    // The task queue
    std::queue<std::function<void()>> tasks;

    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// The constructor just launches some number of workers
inline ThreadPool::ThreadPool(size_t threads): stop(false){
    for(size_t i = 0; i < threads; ++i)
        workers.emplace_back([this]{
                for(;;){
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            }
        );
}

// The destructor joins all threads
inline ThreadPool::~ThreadPool(){
    stopAll();
}

// Add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Don't allow enqueueing after stopping the pool
        if(stop) throw std::runtime_error("Enqueue on stopped ThreadPool!");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

inline void ThreadPool::stopAll(){
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers) worker.join();
}


//Simple set of threads
class ThreadSet {
public:
    ThreadSet();
    ~ThreadSet();

    template<class F, class... Args>
    auto add(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    void joinAll();

private:
    std::vector<std::thread> workers;
};

inline ThreadSet::ThreadSet(){ }

inline ThreadSet::~ThreadSet(){
    joinAll();
}

// Add new thread to set
template<class F, class... Args>
auto ThreadSet::add(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

    std::future<return_type> res = task->get_future();
    workers.push_back(std::thread([task](){ (*task)(); }));
    return res;
}

inline void ThreadSet::joinAll(){
    for(auto &worker: workers)
        worker.join();
    workers.clear();
}
