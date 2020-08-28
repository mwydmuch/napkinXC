/*
 Copyright (c) 2019-2020 by Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

// Time & resources utils

#include "resources.h"

#include <fstream>


#if defined(__linux__) || defined(__APPLE__)
#include <sys/times.h>
#include <unistd.h>
#endif

Resources getResources() {
    Resources rc;
    rc.timePoint = std::chrono::steady_clock::now();
    rc.cpuTime = static_cast<double>(clock()) / CLOCKS_PER_SEC;

#if defined(__linux__) || defined(__APPLE__)
    const long ticks = sysconf(_SC_CLK_TCK);
    tms t;
    times(&t);

    rc.userCpuTime = static_cast<double>(t.tms_utime) / ticks;
    rc.systemCpuTime = static_cast<double>(t.tms_stime) / ticks;
#endif

#ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string next;
    while (status >> next) {
        status >> next;
        if (next == "VmPeak:")
            status >> rc.peakVirtualMem;
        else if (next == "VmSize:")
            status >> rc.currentVirtualMem;
        else if (next == "VmHWM:")
            status >> rc.peakRealMem;
        else if (next == "VmRSS:")
            status >> rc.currentRealMem;
        else if (next == "VmData:")
            status >> rc.dataMemory;
        else if (next == "VmStk:")
            status >> rc.stackMemory;
    }

    status.close();
#endif

// TODO
#ifdef __APPLE__
    rc.peakVirtualMem = 0;
    rc.currentVirtualMem = 0;
    rc.peakRealMem = 0;
    rc.currentRealMem = 0;
#endif

    return rc;
}

int getCpuCount() { return std::thread::hardware_concurrency(); }

unsigned long long getSystemMemory() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}