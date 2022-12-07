# parallel-exercises
Repository for the stuffs I made while learning to program with parallel processors

## Pre-requisites
**NOTE**: Currently I have been working on it with Windows 11 OS
- CUDA toolkit and GPU with compute capability 2.1+
- MSVC Build Tools 2015 or newer (be sure of nvcc and MSVC's cl compatibility)
- SDL2

## Modules
**lena_bw** does a grayscale conversion on a color tiff file of 512 x 512 resolution (not shared in repo right now) named "lena_color.tiff" in the directory with executable

## How to run
### Windows 10/11
```
#set up environment variables in MSVC, consider vcvarsall
nmake <module_name> 
<module_name>.exe
```