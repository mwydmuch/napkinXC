/*
 Copyright (c) 2020 by Marek Wydmuch

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

// Minimal logging tool

#pragma once

#include <ctime>
#include <cstdio>
#include <iostream>
#include <string>

// Logging
enum LogLevel {
    NONE,
    COUT,
    CERR,
    CERR_DEBUG,
};



class Log {
public:
    // Log config
    static inline LogLevel logLevel = NONE;
    static inline bool logTime = false;
    static inline bool logLabel = false;
    static inline int logIndent = 0;

    
    Log() {}
    Log(LogLevel level, int indent = 0, bool time = false, bool label = false): level(level) {
        if(time || logTime) operator << (getTime() + " ");
        if(label || logLabel) operator << ("[" + getLabel(level) + "] : ");
        if((logIndent + indent) > 0) operator << (std::string(logIndent + indent, ' '));
    }

    ~Log() {
        //if(opened) cout << std::endl;
        opened = false;
    }

    template<class T>
    Log &operator<<(const T &msg) {
        if(level <= logLevel) {
            switch(level){
                case NONE: break;
                case COUT: std::cout << msg; opened = true; break;
                case CERR: std::cerr << msg; opened = true; break;
                case CERR_DEBUG: std::cerr << msg; opened = true; break;
            }
        }
        return *this;
    }

    static LogLevel getLogLevel() { return logLevel; }
    static void setLogLevel(LogLevel level) { logLevel = level; }

    static int getGlobalIndent() { return logIndent; }
    static void setGlobalIndent(int indent) { logIndent = indent; }
    static void updateGlobalIndent(int indent) { logIndent += indent; }

    static std::string newLine(int indent = 0) { return "\n" + std::string(logIndent + indent, ' '); }

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
            case CERR_DEBUG: label = "DEBUG"; break;
        }
        return label;
    }
};
