#pragma once

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_


#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#include <windows.h>
#include <math.h>


LARGE_INTEGER beginClock, endClock, clockFreq;
LARGE_INTEGER tot_beginClock, tot_endClock, tot_clockFreq;

cl_mem   d_input;
cl_mem   d_kernel;
cl_mem   d_output;
// 내가 추가

cl_platform_id platform;

cl_context          context;
cl_device_id        device;
cl_command_queue    queue;

cl_program program;

cl_kernel  simpleKernel;
cl_kernel  simpleKernel2;
cl_kernel  simpleKernel3;

#endif  /* #ifndef HISTOGRAM_H_ */
