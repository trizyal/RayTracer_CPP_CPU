#!/usr/bin/bash

make config=release_x64 -j6

./bin/main-release-x64-gcc.exe objects/cornell_box.obj objects/cornell_box.mtl