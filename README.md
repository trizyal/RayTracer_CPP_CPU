The easiest way to build this CPU based Raytracer is:

./premake5 gmake2
make config=release_x64 -j6
./bin/main-release-x64-gcc.exe objects/cornell_box.obj objects/cornell_box.mtl


![Alt text](/progress_images/fresnelrefraction.png)
