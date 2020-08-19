/**
 * Copyright (c) 2020 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <ctime>
#include <cstdio>
#include <iostream>

// Logging
enum LogLevel {
    NONE,
    COUT,
    CERR
};

// Log config
extern LogLevel logLevel;
extern bool logTime;
extern bool logLabel;

class LOG {
public:
    LOG() {}
    LOG(LogLevel level): level(level) {
        if(logTime) operator << (getTime() + " ");
        if(logLabel) operator << ("[" + getLabel(level) + "] : ");
    }

    ~LOG() {
        //if(opened) cout << std::endl;
        opened = false;
    }

    template<class T>
    LOG &operator<<(const T &msg) {
        if(level <= logLevel) {
            switch(level){
                case NONE: break;
                case COUT: std::cout << msg; opened = true; break;
                case CERR: std::cerr << msg; opened = true; break;
            }
        }
        return *this;
    }

private:
    bool opened = false;
    LogLevel level = CERR;

    inline std::string getTime(){
        time_t now = time(NULL);
        return std::string(ctime(&now));
    }

    inline std::string getLabel(LogLevel level) {
        std::string label;
        switch(level) {
            case NONE: break;
            case COUT: label = "COUT"; break;
            case CERR: label = "CERR"; break;
        }
        return label;
    }
};
