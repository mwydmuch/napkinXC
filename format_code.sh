#!/usr/bin/env bash
find src \( -path src/blas -o -path src/liblinear -o -path src/utils \) -prune -o -type f -print | xargs clang-format -style=file -i