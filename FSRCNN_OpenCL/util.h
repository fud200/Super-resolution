#pragma warning(disable:4996)
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <Windows.h>
#include <math.h>
#include <io.h>
#include <time.h>
#include <direct.h>
#include <cstdio>
#include <CL/cl.h>
#include <cmath>

typedef struct strs {
    char strs[255];
}; strs;

typedef struct _element {
    int cols;
    float w;
}; _element;

typedef struct _minmax {
    float mincc;
    float maxcc;
}; _minmax;

typedef struct _SR {
    _element element;
}; _SR;

typedef struct _SM {
    int n;
    _SR* SR;
}; _SM;

typedef struct _Output {
    float* feature_map;
}; _Output;

typedef struct _Output_B {
    byte* feature_map;
}; _Output_B;



typedef struct _bcelement {
    int x;
    int y;
    int w;
}; _bcelement;

typedef struct _bcSR {
    _bcelement element[16];
}; _bcSR;

typedef struct _bcSM {
    int n;
    _bcSR* SR;
}; _bcSM;


float u(float s, float a) {
    if ((fabs(s) >= 0) && (fabs(s) <= 1))
        return (a + 2) * (fabs(s) * fabs(s) * fabs(s)) - (a + 3) * (fabs(s) * fabs(s)) + 1;

    else if ((fabs(s) > 1) && (fabs(s) <= 2))
        return a * (fabs(s) * fabs(s) * fabs(s)) - (5 * a) * (fabs(s) * fabs(s)) + (8 * a) * fabs(s) - 4 * a;
    //return 0 - a * (fabs(s) * fabs(s) * fabs(s)) - (5 * a) * (fabs(s) * fabs(s)) + (8 * a) * fabs(s) - 4 * a;

    return 0;

}
char* readSource(char* kernelPath) {

    cl_int status;
    FILE* fp;
    char* source;
    long int size;

    printf("Program file is: %s\n", kernelPath);

    fp = fopen(kernelPath, "rb");
    if (!fp) {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    status = fseek(fp, 0, SEEK_END);
    if (status != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    size = ftell(fp);
    if (size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }

    rewind(fp);

    source = (char*)malloc(size + 1);

    int i;
    for (i = 0; i < size + 1; i++) {
        source[i] = '\0';
    }

    if (source == NULL) {
        printf("Error allocating space for the kernel source\n");
        exit(-1);
    }

    fread(source, 1, size, fp);
    source[size] = '\0';

    return source;
}

void CLInit()
{
    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (i = 0; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("platform %d. Device %d: %s\n", i + 1, j + 1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", i + 1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", i + 1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", i + 1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", i + 1, 4, maxComputeUnits);
        }
    }
    int platformNum;
    int deviceNum;
    printf("\n\nSELECT PLATFORM('1' ~ '%d') : ", platformCount);
    scanf("%d", &platformNum);
    printf("\n");
    printf("SELECT DEVICE('1' ~ '%d') : ", deviceCount);
    scanf("%d", &deviceNum);
    printf("\n");
    clGetDeviceIDs(platforms[platformNum - 1], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

    device = devices[deviceNum - 1];

    //create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    //create command queue
    queue = clCreateCommandQueue(context, device, 0, NULL);

    // 텍스트파일로부터 프로그램 읽기
    char* source = readSource("convolution.cl");

    // compile program
    program = clCreateProgramWithSource(context, 1,
        (const char**)&source, NULL, NULL);
    cl_int build_status;
    build_status = clBuildProgram(program, 1, &device, NULL, NULL,
        NULL);

    //커널 포인터 생성
    simpleKernel = clCreateKernel(program, "on_g_mat", NULL);
    simpleKernel2 = clCreateKernel(program, "conv1", NULL);
    simpleKernel3 = clCreateKernel(program, "convt", NULL);

}
void bufferWrite(float* input, float* kernel, float* output)
{
    //queue = clCreateCommandQueue(context, device, 0, NULL);
    long long int input_size = _msize(input);
    long long int kernel_size = _msize(kernel);
    long long int output_size = _msize(output);
    //printf("%d ",ksize);
    // 메모리 버퍼 생성

    /*d_inputArray_A = clCreateBuffer(context, CL_MEM_READ_WRITE,
        10*sizeof(int), NULL, NULL);
    d_inputArray_B = clCreateBuffer(context, CL_MEM_READ_WRITE,
        10*sizeof(int), NULL, NULL);
    d_outputArray = clCreateBuffer(context, CL_MEM_READ_WRITE,
        10 * sizeof(int), NULL, NULL);*/
        /*d_SR = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ksize * ksize * Imgsize * sizeof(_SR), NULL, NULL);*/
    d_input = clCreateBuffer(context, CL_MEM_READ_WRITE,
        input_size, NULL, NULL);
    d_kernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
        kernel_size, NULL, NULL);
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE,
        output_size, NULL, NULL);
    //clEnqueueWriteBuffer(queue, d_SR, CL_TRUE, 0, ksize * ksize * Imgsize * sizeof(_SR), SM.SR, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    //이게 터져
    clEnqueueWriteBuffer(queue, d_kernel, CL_TRUE, 0, kernel_size , kernel, 0, NULL, NULL);


    //clEnqueueWriteBuffer(queue, d_image, CL_TRUE, 0, 2000 * sizeof(char), image, 0, NULL, NULL);
    /*clEnqueueWriteBuffer(queue, d_output, CL_TRUE, 0, 2000 * 2000 * sizeof(int),
        B, 0, NULL, NULL);*/


}
void bufferWrite_1(float* input, float* kernel, float* output)
{
    //queue = clCreateCommandQueue(context, device, 0, NULL);
    long long int input_size = _msize(input);
    long long int kernel_size = _msize(kernel);
    long long int output_size = _msize(output);
    //printf("%d ",ksize);
    // 메모리 버퍼 생성

    /*d_inputArray_A = clCreateBuffer(context, CL_MEM_READ_WRITE,
        10*sizeof(int), NULL, NULL);
    d_inputArray_B = clCreateBuffer(context, CL_MEM_READ_WRITE,
        10*sizeof(int), NULL, NULL);
    d_outputArray = clCreateBuffer(context, CL_MEM_READ_WRITE,
        10 * sizeof(int), NULL, NULL);*/
        /*d_SR = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ksize * ksize * Imgsize * sizeof(_SR), NULL, NULL);*/
    d_input = clCreateBuffer(context, CL_MEM_READ_WRITE,
        input_size, NULL, NULL);
    d_kernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
        kernel_size, NULL, NULL);
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE,
        output_size, NULL, NULL);
    //clEnqueueWriteBuffer(queue, d_SR, CL_TRUE, 0, ksize * ksize * Imgsize * sizeof(_SR), SM.SR, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    //이게 터져
    clEnqueueWriteBuffer(queue, d_kernel, CL_TRUE, 0, kernel_size, kernel, 0, NULL, NULL);


    //clEnqueueWriteBuffer(queue, d_image, CL_TRUE, 0, 2000 * sizeof(char), image, 0, NULL, NULL);
    /*clEnqueueWriteBuffer(queue, d_output, CL_TRUE, 0, 2000 * 2000 * sizeof(int),
        B, 0, NULL, NULL);*/


}
void bufferWrite_trans(float* input, float* kernel, float* output)
{
    //queue = clCreateCommandQueue(context, device, 0, NULL);
    long long int input_size = _msize(input);
    long long int kernel_size = _msize(kernel);
    long long int output_size = _msize(output);
    //printf("%d ",ksize);
    // 메모리 버퍼 생성

    /*d_inputArray_A = clCreateBuffer(context, CL_MEM_READ_WRITE,
        10*sizeof(int), NULL, NULL);
    d_inputArray_B = clCreateBuffer(context, CL_MEM_READ_WRITE,
        10*sizeof(int), NULL, NULL);
    d_outputArray = clCreateBuffer(context, CL_MEM_READ_WRITE,
        10 * sizeof(int), NULL, NULL);*/
        /*d_SR = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ksize * ksize * Imgsize * sizeof(_SR), NULL, NULL);*/
    d_input = clCreateBuffer(context, CL_MEM_READ_WRITE,
        input_size, NULL, NULL);
    d_kernel = clCreateBuffer(context, CL_MEM_READ_WRITE,
        kernel_size, NULL, NULL);
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE,
        output_size, NULL, NULL);
    //clEnqueueWriteBuffer(queue, d_SR, CL_TRUE, 0, ksize * ksize * Imgsize * sizeof(_SR), SM.SR, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, input_size, input, 0, NULL, NULL);
    //이게 터져
    clEnqueueWriteBuffer(queue, d_kernel, CL_TRUE, 0, kernel_size, kernel, 0, NULL, NULL);


    //clEnqueueWriteBuffer(queue, d_image, CL_TRUE, 0, 2000 * sizeof(char), image, 0, NULL, NULL);
    /*clEnqueueWriteBuffer(queue, d_output, CL_TRUE, 0, 2000 * 2000 * sizeof(int),
        B, 0, NULL, NULL);*/


}


void runKernel(float* output, int W, int H, int Ksize, int padding)
{

    long long int output_size = _msize(output);
    //int totalWorkItemsX = ImgSize;
    int totalWorkItemsX = W;
    //int totalWorkItemsX = 16;
    int totalWorkItemsY = H;

    size_t globalSize[2] = { totalWorkItemsX, totalWorkItemsY };

    float* minVal, * maxVal;
    // 커널 매개변수 설정 
    clSetKernelArg(simpleKernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(simpleKernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(simpleKernel, 2, sizeof(cl_mem), &d_kernel);
    clSetKernelArg(simpleKernel, 3, sizeof(int), &H);
    clSetKernelArg(simpleKernel, 4, sizeof(int), &W);
    clSetKernelArg(simpleKernel, 5, sizeof(int), &Ksize);
    clSetKernelArg(simpleKernel, 6, sizeof(int), &padding);
    clEnqueueNDRangeKernel(queue, simpleKernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    // 완료 대기
    clFinish(queue);
    //output = (float*)malloc(sizeof(float) * ImgSize);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size, output, 0, NULL, NULL);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_kernel);
    clReleaseMemObject(d_output);
    //clReleaseCommandQueue(queue);
    //for (int i = 0; i < 50; i++) {
    //    for (int j = 0; j < 3; j++) {
    //        cout << (output[i + j]) << " ";
    //    }
    //    cout << endl;
    //}
    /*for (int i = 1056; i < 2000; i=i+1) {
        cout << (output[i]) << " ";
        if (i != 0 && i % 3 == 0)
            printf("\n");
    }*/
}
void runKernel_1(float* output, int W, int H)
{

    long long int output_size = _msize(output);
    //int totalWorkItemsX = ImgSize;
    int totalWorkItemsX = W;
    //int totalWorkItemsX = 16;
    int totalWorkItemsY = H;

    size_t globalSize[2] = { totalWorkItemsX, totalWorkItemsY };

    float* minVal, * maxVal;
    // 커널 매개변수 설정 
    clSetKernelArg(simpleKernel2, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(simpleKernel2, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(simpleKernel2, 2, sizeof(cl_mem), &d_kernel);
    clSetKernelArg(simpleKernel2, 3, sizeof(int), &H);
    clSetKernelArg(simpleKernel2, 4, sizeof(int), &W);
    clEnqueueNDRangeKernel(queue, simpleKernel2, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    // 완료 대기
    clFinish(queue);
    //output = (float*)malloc(sizeof(float) * ImgSize);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size, output, 0, NULL, NULL);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_kernel);
    clReleaseMemObject(d_output);
    //clReleaseCommandQueue(queue);
    //for (int i = 0; i < 50; i++) {
    //    for (int j = 0; j < 3; j++) {
    //        cout << (output[i + j]) << " ";
    //    }
    //    cout << endl;
    //}
    /*for (int i = 1056; i < 2000; i=i+1) {
        cout << (output[i]) << " ";
        if (i != 0 && i % 3 == 0)
            printf("\n");
    }*/
}void runKernel_trans(float* output, int OH, int OW,int PW3, int k, int ks, int y)
{

    long long int output_size = _msize(output);
    //int totalWorkItemsX = ImgSize;
    int totalWorkItemsX = OW;
    //int totalWorkItemsX = 16;
    int totalWorkItemsY = OH;

    size_t globalSize[2] = { totalWorkItemsX, totalWorkItemsY };

    float* minVal, * maxVal;
    // 커널 매개변수 설정 
    clSetKernelArg(simpleKernel3, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(simpleKernel3, 1, sizeof(cl_mem), &d_kernel);
    clSetKernelArg(simpleKernel3, 2, sizeof(cl_mem), &d_output);
    clSetKernelArg(simpleKernel3, 3, sizeof(int), &OH);
    clSetKernelArg(simpleKernel3, 4, sizeof(int), &OW);
    clSetKernelArg(simpleKernel3, 5, sizeof(int), &PW3);
    clSetKernelArg(simpleKernel3, 6, sizeof(int), &k);
    clSetKernelArg(simpleKernel3, 7, sizeof(int), &ks);
    clSetKernelArg(simpleKernel3, 8, sizeof(int), &y);
    clEnqueueNDRangeKernel(queue, simpleKernel3, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    // 완료 대기
    clFinish(queue);
    //output = (float*)malloc(sizeof(float) * ImgSize);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size, output, 0, NULL, NULL);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_kernel);
    clReleaseMemObject(d_output);
    //clReleaseCommandQueue(queue);
    //for (int i = 0; i < 50; i++) {
    //    for (int j = 0; j < 3; j++) {
    //        cout << (output[i + j]) << " ";
    //    }
    //    cout << endl;
    //}
    /*for (int i = 1056; i < 2000; i=i+1) {
        cout << (output[i]) << " ";
        if (i != 0 && i % 3 == 0)
            printf("\n");
    }*/
}

void Release()
{
    // 릴리즈
    clReleaseProgram(program);
    //clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
void matmul_5(float* Padding,float* kernel,float* feature_map,int H, int W, int ksize,int pad) {
    for (int z = 0; z < H; z++) { // z: Height, y축
        for (int t = 0; t < W; t++) { // t: Width, x축
            _element sr[25];
            for (int i = 0; i < ksize; i++) { // y축 kernel size
                for (int j = 0; j < ksize; j++) { // x축 kernel size
                    sr[i * ksize + j].cols = (t + ((W + pad + pad) * i) + j + (z * (W + pad + pad)));
                    sr[i * ksize + j].w = kernel[i * ksize + j];
                }
            }

            float sumb = 0;
            sumb = 0;
            for (int j = 0; j < ksize; j++) {
                for (int k = 0; k < ksize; k++) {
                    sumb = sumb + sr[j * ksize + k].w * (float)Padding[sr[j * ksize + k].cols];
                }
            }
            feature_map[z * W + t] = sumb;

        }
    }
}
void matmul_3(float* Padding, float* kernel, float* feature_map, int H, int W, int ksize, int pad) {
    for (int z = 0; z < H; z++) { // z: Height, y축
        for (int t = 0; t < W; t++) { // t: Width, x축
            _element sr[9];
            for (int i = 0; i < ksize; i++) { // y축 kernel size
                for (int j = 0; j < ksize; j++) { // x축 kernel size
                    sr[i * ksize + j].cols = (t + ((W + pad + pad) * i) + j + (z * (W + pad + pad)));
                    sr[i * ksize + j].w = kernel[i * ksize + j];
                }
            }

            float sumb = 0;
            sumb = 0;
            for (int j = 0; j < ksize; j++) {
                for (int k = 0; k < ksize; k++) {
                    sumb = sumb + sr[j * ksize + k].w * (float)Padding[sr[j * ksize + k].cols];
                }
            }
            feature_map[z * W + t] = sumb;

        }
    }
}
void matmul_1(float* prefeature_map,float* kernel,float* feature_map,int H,int W) {
    for (int z = 0; z < H; z++) { // z: Height, y축
        for (int t = 0; t < W; t++) { // t: Width, x축
            feature_map[z * W + t] = kernel[0] * float(prefeature_map[z * W + t]);
        }
    }
}
void trans_matmul(float* Padding,float* kernel,float* TempY,int OH, int OW,int PW, int k,int ksize,int y3) {
    for (int z = 0; z < OH; z++) { // z: Height, y축
        int h_start = z * PW * 1;
        int h_end = h_start + k;
        for (int t = 0; t < OW; t++) { // t: Width, x축
            int w_start = t * 1;
            int w_end = w_start + ksize;
            float sumb = 0;
            int kx = 0;
            int ky = 0;
            for (int i = h_start; i < h_end; i += PW) {
                for (int j = w_start; j < w_end; j++) {
                    sumb = sumb + kernel[ky + kx] * float(Padding[i + j]);
                    //printf("kernel index: %d, padding[%d]: %f\n", kx + ky, i + j, Padding3[i + j]);
                    kx++;

                }
                kx = 0;
                ky += ksize;
            }
            TempY[y3 + t] = sumb;
            //printf("%d\n", y3 + t);
        }
        y3 += OW;
    }
}

// bicubic Sparse Matrix
void BC_MakeSparseMatrix(BYTE* Image, BYTE* Output, _bcSM SM, int W, int H, float scale)
{
    clock_t start;
    clock_t end;

    start = clock();

    float IH = H;
    float IW = W;
    float in_size = H * W;

    float OH = H * scale;
    float OW = W * scale;
    float out_size = OH * OW;
    printf("%f %f\n", OW, OH);
    float a = -0.5;
    float h = 1 / scale;

    float ux, ux1, ux2, ux3, uy, uy1, uy2, uy3;
    int cx, cy;

    float rpix = 0;

    float _x[4];
    float _y[4];

    for (int y = 0; y < OH; y++) {
        for (int x = 0; x < OW; x++) {
            int index = x + (OW * y);

            // 0-0. Output에 와야하는 원본 이미지 좌표를 역연산
            cx = x * h;
            cy = y * h;
            // printf("cx, cy: %f, %f\n", cx, cy);
            // ** 가중치 넣는 데 필요한 부분 ** //
            // 0-1. 입력 이미지 위치에 따라 다른 가중치 값 부여
            if (fmodf(cx, 1) == 0) {
                ux = u(1, a);
                ux1 = u(0, a);
                ux2 = u(1, a);
                ux3 = u(2, a);
            }
            else {
                ux = u(1.5, a);
                ux1 = u(0.5, a);
                ux2 = u(0.5, a);
                ux3 = u(1.5, a);
            }

            if (fmodf(cy, 1) == 0) {
                uy = u(1, a);
                uy1 = u(0, a);
                uy2 = u(1, a);
                uy3 = u(2, a);
            }
            else {
                uy = u(1.5, a);
                uy1 = u(0.5, a);
                uy2 = u(0.5, a);
                uy3 = u(1.5, a);
            }

            // cx, cy : 참고해야 되는 input 이미지의 위치
            // j,i : output 이미지의 위치
            // cx-1, cx-1 : 가중치가 곱해지는 시작 위치(mat_w[0]에 대응)

            _x[0] = 1 + cx - floor(cx); // 1.5
            _x[1] = cx - floor(cx); // 0.5
            _x[2] = floor(cx) + 1 - cx; // 0.5
            _x[3] = floor(cx) + 2 - cx; // 1.5

            _y[0] = 1 + cy - floor(cy);
            _y[1] = cy - floor(cy);
            _y[2] = floor(cy) + 1 - cy;
            _y[3] = floor(cy) + 2 - cy;
            /*
            int px1 = cx - _x[0];
            int py1 = cy - _y[0];
            int w1 = ux3 * uy3 * 100;

            int px2 = cx - _x[1];
            int py2 = cy - _y[0];
            int w2 = ux3 * uy2 * 100;

            int px3 = cx + _x[2];
            int py3 = cy - _y[0];
            int w3 = ux3 * uy1 * 100;

            int px4 = cx + _x[3];
            int py4 = cy - _y[0];
            int w4 = ux3 * uy * 100;

            int px5 = cx - _x[0];
            int py5 = cy - _y[1];
            int w5 = ux2 * uy3 * 100;

            int px6 = cx - _x[1];
            int py6 = cy - _y[1];
            int w6 = ux2 * uy3 * 100;

            int px7 = cx + _x[2];
            int py7 = cy - _y[1];
            int w7 = ux2 * uy1 * 100;

            int px8 = cx + _x[3];
            int py8 = cy - _y[1];
            int w8 = ux2 * uy * 100;

            int px9 = cx - _x[0];
            int py9 = cy + _y[2];
            int w9 = ux1 * uy3 * 100;

            int px10 = cx - _x[1];
            int py10 = cy + _y[2];
            int w10 = ux1 * uy2 * 100;

            int px11 = cx + _x[2];
            int py11 = cy + _y[2];
            int w11 = ux1 * uy1 * 100;

            int px12 = cx + _x[3];
            int py12 = cy + _y[2];
            int w12 = ux1 * uy * 100;

            int px13 = cx - _x[0];
            int py13 = cy + _y[3];
            int w13 = ux * uy3 * 100;

            int px14 = cx - _x[1];
            int py14 = cy + _y[3];
            int w14 = ux * uy2 * 100;

            int px15 = cx + _x[2];
            int py15 = cy + _y[3];
            int w15 = ux * uy1 * 100;

            int px16 = cx + _x[3];
            int py16 = cy + _y[3];
            int w16 = ux * uy * 100;

            SM.SR[index].element[0].x = px1;
            SM.SR[index].element[0].y = py1;
            SM.SR[index].element[0].w = w1;

            SM.SR[index].element[1].x = px2;
            SM.SR[index].element[1].y = py2;
            SM.SR[index].element[1].w = w2;

            SM.SR[index].element[2].x = px3;
            SM.SR[index].element[2].y = py3;
            SM.SR[index].element[2].w = w3;

            SM.SR[index].element[3].x = px4;
            SM.SR[index].element[3].y = py4;
            SM.SR[index].element[3].w = w4;

            SM.SR[index].element[4].x = px5;
            SM.SR[index].element[4].y = py5;
            SM.SR[index].element[4].w = w5;

            SM.SR[index].element[5].x = px6;
            SM.SR[index].element[5].y = py6;
            SM.SR[index].element[5].w = w6;

            SM.SR[index].element[6].x = px7;
            SM.SR[index].element[6].y = py7;
            SM.SR[index].element[6].w = w7;

            SM.SR[index].element[7].x = px8;
            SM.SR[index].element[7].y = py8;
            SM.SR[index].element[7].w = w8;

            SM.SR[index].element[8].x = px9;
            SM.SR[index].element[8].y = py9;
            SM.SR[index].element[8].w = w9;

            SM.SR[index].element[9].x = px10;
            SM.SR[index].element[9].y = py10;
            SM.SR[index].element[9].w = w10;

            SM.SR[index].element[10].x = px11;
            SM.SR[index].element[10].y = py11;
            SM.SR[index].element[10].w = w11;

            SM.SR[index].element[11].x = px12;
            SM.SR[index].element[11].y = py12;
            SM.SR[index].element[11].w = w12;

            SM.SR[index].element[12].x = px13;
            SM.SR[index].element[12].y = py13;
            SM.SR[index].element[12].w = w13;

            SM.SR[index].element[13].x = px14;
            SM.SR[index].element[13].y = py14;
            SM.SR[index].element[13].w = w14;

            SM.SR[index].element[14].x = px15;
            SM.SR[index].element[14].y = py15;
            SM.SR[index].element[14].w = w15;

            SM.SR[index].element[15].x = px16;
            SM.SR[index].element[15].y = py16;
            SM.SR[index].element[15].w = w16;
            */

            SM.SR[index].element[0].x = cx - _x[0];
            SM.SR[index].element[0].y = cy - _y[0];
            SM.SR[index].element[0].w = ux3 * uy3 * 100;

            SM.SR[index].element[1].x = cx - _x[1];
            SM.SR[index].element[1].y = cy - _y[0];
            SM.SR[index].element[1].w = ux3 * uy2 * 100;

            SM.SR[index].element[2].x = cx + _x[2];
            SM.SR[index].element[2].y = cy - _y[0];
            SM.SR[index].element[2].w = ux3 * uy1 * 100;

            SM.SR[index].element[3].x = cx + _x[3];
            SM.SR[index].element[3].y = cy - _y[0];
            SM.SR[index].element[3].w = ux3 * uy * 100;



            SM.SR[index].element[4].x = cx - _x[0];
            SM.SR[index].element[4].y = cy - _y[1];
            SM.SR[index].element[4].w = ux2 * uy3 * 100;

            SM.SR[index].element[5].x = cx - _x[1];
            SM.SR[index].element[5].y = cy - _y[1];
            SM.SR[index].element[5].w = ux2 * uy2 * 100;

            SM.SR[index].element[6].x = cx + _x[2];
            SM.SR[index].element[6].y = cy - _y[1];
            SM.SR[index].element[6].w = ux2 * uy1 * 100;

            SM.SR[index].element[7].x = cx + _x[3];
            SM.SR[index].element[7].y = cy - _y[1];
            SM.SR[index].element[7].w = ux2 * uy * 100;



            SM.SR[index].element[8].x = cx - _x[0];
            SM.SR[index].element[8].y = cy + _y[2];
            SM.SR[index].element[8].w = ux1 * uy3 * 100;

            SM.SR[index].element[9].x = cx - _x[1];
            SM.SR[index].element[9].y = cy + _y[2];
            SM.SR[index].element[9].w = ux1 * uy2 * 100;

            SM.SR[index].element[10].x = cx + _x[2];
            SM.SR[index].element[10].y = cy + _y[2];
            SM.SR[index].element[10].w = ux1 * uy1 * 100;

            SM.SR[index].element[11].x = cx + _x[2];
            SM.SR[index].element[11].y = cy + _y[2];
            SM.SR[index].element[11].w = ux1 * uy * 100;



            SM.SR[index].element[12].x = cx - _x[0];
            SM.SR[index].element[12].y = cy + _y[3];
            SM.SR[index].element[12].w = ux * uy3 * 100;

            SM.SR[index].element[13].x = cx - _x[1];
            SM.SR[index].element[13].y = cy + _y[3];
            SM.SR[index].element[13].w = ux * uy2 * 100;

            SM.SR[index].element[14].x = cx + _x[2];
            SM.SR[index].element[14].y = cy + _y[2];
            SM.SR[index].element[14].w = ux * uy1 * 100;

            SM.SR[index].element[15].x = cx + _x[3];
            SM.SR[index].element[15].y = cy + _y[3];
            SM.SR[index].element[15].w = ux * uy * 100;

            if (cx - _x[0] < 0) {
                SM.SR[index].element[0].x = 0;
                SM.SR[index].element[0].y = 0;
                SM.SR[index].element[0].w = 0;

                SM.SR[index].element[4].x = 0;
                SM.SR[index].element[4].y = 0;
                SM.SR[index].element[4].w = 0;

                SM.SR[index].element[8].x = 0;
                SM.SR[index].element[8].y = 0;
                SM.SR[index].element[8].w = 0;

                SM.SR[index].element[12].x = 0;
                SM.SR[index].element[12].y = 0;
                SM.SR[index].element[12].w = 0;
            }

            if (cy - _y[0] < 0) {
                SM.SR[index].element[0].x = 0;
                SM.SR[index].element[0].y = 0;
                SM.SR[index].element[0].w = 0;

                SM.SR[index].element[1].x = 0;
                SM.SR[index].element[1].y = 0;
                SM.SR[index].element[1].w = 0;

                SM.SR[index].element[2].x = 0;
                SM.SR[index].element[2].y = 0;
                SM.SR[index].element[2].w = 0;

                SM.SR[index].element[3].x = 0;
                SM.SR[index].element[3].y = 0;
                SM.SR[index].element[3].w = 0;
            }

            if (cy + _y[2] >= IH) {
                SM.SR[index].element[8].x = 0;
                SM.SR[index].element[8].y = 0;
                SM.SR[index].element[8].w = 0;

                SM.SR[index].element[9].x = 0;
                SM.SR[index].element[9].y = 0;
                SM.SR[index].element[9].w = 0;

                SM.SR[index].element[10].x = 0;
                SM.SR[index].element[10].y = 0;
                SM.SR[index].element[10].w = 0;

                SM.SR[index].element[11].x = 0;
                SM.SR[index].element[11].y = 0;
                SM.SR[index].element[11].w = 0;

            }

            if (cy + _y[3] >= IH) {

                SM.SR[index].element[12].x = 0;
                SM.SR[index].element[12].y = 0;
                SM.SR[index].element[12].w = 0;

                SM.SR[index].element[13].x = 0;
                SM.SR[index].element[13].y = 0;
                SM.SR[index].element[13].w = 0;

                SM.SR[index].element[14].x = 0;
                SM.SR[index].element[14].y = 0;
                SM.SR[index].element[14].w = 0;

                SM.SR[index].element[15].x = 0;
                SM.SR[index].element[15].y = 0;
                SM.SR[index].element[15].w = 0;
            }
        }


    }

    end = clock();
    printf("BICUIBC 희소행렬 생성 완료: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);

}

void TESTBC(BYTE* Image, BYTE* Output, int W, int H, int size, float scale, _bcSM SM)
{
    /*
    Bicubic matrix 곱 수행 시 변수에 값을 할당한 후 Output 값을 결정하는 게 약 0.1~ 0.2 초 정도 덜 소요됨
    따라서 값을 바로 부르지말고 변수에 대입 후 Output 값 결정하기!
    */
    clock_t start;
    clock_t end;


    start = clock();

    int IW = W;
    int IH = H;

    int OW = scale * W;
    int OH = scale * H;

    printf("%d\n", OW);

    for (int y = 0; y < OH; y++) {
        for (int x = 0; x < OW; x++) {

            int sm_index = x + (y * OW);
            int out_index = x * 3 + (y * OW * 3);

            // 변수 선언 하지말고 바로 때려박기
            // w: float --> int
            // w를 곱하기 100을 해서 integer 로 바꾸고
            // 다 곱하고 output image를 만들 때 나눠주기 /100을 하기(소수점은 무시하기)
            // 100, 1000, 10000 나눠보기

            /*
            printf("%d\n", Image[(SM.SR[sm_index].element[0].x * 3) + (SM.SR[sm_index].element[0].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[1].x * 3) + (SM.SR[sm_index].element[1].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[2].x * 3) + (SM.SR[sm_index].element[2].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[3].x * 3) + (SM.SR[sm_index].element[3].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[4].x * 3) + (SM.SR[sm_index].element[4].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[5].x * 3) + (SM.SR[sm_index].element[5].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[6].x * 3) + (SM.SR[sm_index].element[6].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[7].x * 3) + (SM.SR[sm_index].element[7].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[8].x * 3) + (SM.SR[sm_index].element[8].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[9].x * 3) + (SM.SR[sm_index].element[9].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[10].x * 3) + (SM.SR[sm_index].element[10].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[11].x * 3) + (SM.SR[sm_index].element[11].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[12].x * 3) + (SM.SR[sm_index].element[12].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[13].x * 3) + (SM.SR[sm_index].element[13].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[14].x * 3) + (SM.SR[sm_index].element[14].y * W * 3)]);
            printf("%d\n", Image[(SM.SR[sm_index].element[15].x * 3) + (SM.SR[sm_index].element[15].y * W * 3)]);
            */

            int px1 = SM.SR[sm_index].element[0].x;
            int py1 = SM.SR[sm_index].element[0].y;
            int w1 = SM.SR[sm_index].element[0].w;

            int px2 = SM.SR[sm_index].element[1].x;
            int py2 = SM.SR[sm_index].element[1].y;
            int w2 = SM.SR[sm_index].element[1].w;

            int px3 = SM.SR[sm_index].element[2].x;
            int py3 = SM.SR[sm_index].element[2].y;
            int w3 = SM.SR[sm_index].element[2].w;

            int px4 = SM.SR[sm_index].element[3].x;
            int py4 = SM.SR[sm_index].element[3].y;
            int w4 = SM.SR[sm_index].element[3].w;

            int px5 = SM.SR[sm_index].element[4].x;
            int py5 = SM.SR[sm_index].element[4].y;
            int w5 = SM.SR[sm_index].element[4].w;

            int px6 = SM.SR[sm_index].element[5].x;
            int py6 = SM.SR[sm_index].element[5].y;
            int w6 = SM.SR[sm_index].element[5].w;

            int px7 = SM.SR[sm_index].element[6].x;
            int py7 = SM.SR[sm_index].element[6].y;
            int w7 = SM.SR[sm_index].element[6].w;

            int px8 = SM.SR[sm_index].element[7].x;
            int py8 = SM.SR[sm_index].element[7].y;
            int w8 = SM.SR[sm_index].element[7].w;

            int px9 = SM.SR[sm_index].element[8].x;
            int py9 = SM.SR[sm_index].element[8].y;
            int w9 = SM.SR[sm_index].element[8].w;

            int px10 = SM.SR[sm_index].element[9].x;
            int py10 = SM.SR[sm_index].element[9].y;
            int w10 = SM.SR[sm_index].element[9].w;

            int px11 = SM.SR[sm_index].element[10].x;
            int py11 = SM.SR[sm_index].element[10].y;
            int w11 = SM.SR[sm_index].element[10].w;

            int px12 = SM.SR[sm_index].element[11].x;
            int py12 = SM.SR[sm_index].element[11].y;
            int w12 = SM.SR[sm_index].element[11].w;

            int px13 = SM.SR[sm_index].element[12].x;
            int py13 = SM.SR[sm_index].element[12].y;
            int w13 = SM.SR[sm_index].element[12].w;

            int px14 = SM.SR[sm_index].element[13].x;
            int py14 = SM.SR[sm_index].element[13].y;
            int w14 = SM.SR[sm_index].element[13].w;

            int px15 = SM.SR[sm_index].element[14].x;
            int py15 = SM.SR[sm_index].element[14].y;
            int w15 = SM.SR[sm_index].element[14].w;

            int px16 = SM.SR[sm_index].element[15].x;
            int py16 = SM.SR[sm_index].element[15].y;
            int w16 = SM.SR[sm_index].element[15].w;

            int b_v = (Image[(px1 * 3) + (py1 * W * 3)] * w1
                + Image[(px2 * 3) + (py2 * W * 3)] * w2
                + Image[(px3 * 3) + (py3 * W * 3)] * w3
                + Image[(px4 * 3) + (py4 * W * 3)] * w4
                + Image[(px5 * 3) + (py5 * W * 3)] * w5
                + Image[(px6 * 3) + (py6 * W * 3)] * w6
                + Image[(px7 * 3) + (py7 * W * 3)] * w7
                + Image[(px8 * 3) + (py8 * W * 3)] * w8
                + Image[(px9 * 3) + (py9 * W * 3)] * w9
                + Image[(px10 * 3) + (py10 * W * 3)] * w10
                + Image[(px11 * 3) + (py11 * W * 3)] * w11
                + Image[(px12 * 3) + (py12 * W * 3)] * w12
                + Image[(px13 * 3) + (py13 * W * 3)] * w13
                + Image[(px14 * 3) + (py14 * W * 3)] * w14
                + Image[(px15 * 3) + (py15 * W * 3)] * w15
                + Image[(px16 * 3) + (py16 * W * 3)] * w16) / 100;

            int g_v = (Image[(px1 * 3) + (py1 * W * 3) + 1] * w1
                + Image[(px2 * 3) + (py2 * W * 3) + 1] * w2
                + Image[(px3 * 3) + (py3 * W * 3) + 1] * w3
                + Image[(px4 * 3) + (py4 * W * 3) + 1] * w4
                + Image[(px5 * 3) + (py5 * W * 3) + 1] * w5
                + Image[(px6 * 3) + (py6 * W * 3) + 1] * w6
                + Image[(px7 * 3) + (py7 * W * 3) + 1] * w7
                + Image[(px8 * 3) + (py8 * W * 3) + 1] * w8
                + Image[(px9 * 3) + (py9 * W * 3) + 1] * w9
                + Image[(px10 * 3) + (py10 * W * 3) + 1] * w10
                + Image[(px11 * 3) + (py11 * W * 3) + 1] * w11
                + Image[(px12 * 3) + (py12 * W * 3) + 1] * w12
                + Image[(px13 * 3) + (py13 * W * 3) + 1] * w13
                + Image[(px14 * 3) + (py14 * W * 3) + 1] * w14
                + Image[(px15 * 3) + (py15 * W * 3) + 1] * w15
                + Image[(px16 * 3) + (py16 * W * 3) + 1] * w16) / 100;

            int r_v = (Image[(px1 * 3) + (py1 * W * 3) + 2] * w1
                + Image[(px2 * 3) + (py2 * W * 3) + 2] * w2
                + Image[(px3 * 3) + (py3 * W * 3) + 2] * w3
                + Image[(px4 * 3) + (py4 * W * 3) + 2] * w4
                + Image[(px5 * 3) + (py5 * W * 3) + 2] * w5
                + Image[(px6 * 3) + (py6 * W * 3) + 2] * w6
                + Image[(px7 * 3) + (py7 * W * 3) + 2] * w7
                + Image[(px8 * 3) + (py8 * W * 3) + 2] * w8
                + Image[(px9 * 3) + (py9 * W * 3) + 2] * w9
                + Image[(px10 * 3) + (py10 * W * 3) + 2] * w10
                + Image[(px11 * 3) + (py11 * W * 3) + 2] * w11
                + Image[(px12 * 3) + (py12 * W * 3) + 2] * w12
                + Image[(px13 * 3) + (py13 * W * 3) + 2] * w13
                + Image[(px14 * 3) + (py14 * W * 3) + 2] * w14
                + Image[(px15 * 3) + (py15 * W * 3) + 2] * w15
                + Image[(px16 * 3) + (py16 * W * 3) + 2] * w16) / 100;

            /*
            printf("%d, %d\n", px1, py1);
            printf("%d, %d\n", px2, py2);
            printf("%d, %d\n", px3, py3);
            printf("%d, %d\n", px4, py4);
            printf("%d, %d\n", px5, py5);
            printf("%d, %d\n", px6, py6);
            printf("%d, %d\n", px7, py7);
            printf("%d, %d\n", px8, py8);
            printf("%d, %d\n", px9, py9);
            printf("%d, %d\n", px10, py10);
            printf("%d, %d\n", px11, py11);
            printf("%d, %d\n", px12, py12);
            printf("%d, %d\n", px13, py13);
            printf("%d, %d\n", px14, py14);
            printf("%d, %d\n", px15, py15);
            printf("%d, %d\n", px16, py16);
            */
            /*
            int b_v = (Image[(SM.SR[sm_index].element[0].x * 3) + (SM.SR[sm_index].element[0].y * W * 3)] * SM.SR[sm_index].element[0].w
                + Image[(SM.SR[sm_index].element[1].x * 3) + (SM.SR[sm_index].element[1].y * W * 3)] * SM.SR[sm_index].element[1].w
                + Image[(SM.SR[sm_index].element[2].x * 3) + (SM.SR[sm_index].element[2].y * W * 3)] * SM.SR[sm_index].element[2].w
                + Image[(SM.SR[sm_index].element[3].x * 3) + (SM.SR[sm_index].element[3].y * W * 3)] * SM.SR[sm_index].element[3].w
                + Image[(SM.SR[sm_index].element[4].x * 3) + (SM.SR[sm_index].element[4].y * W * 3)] * SM.SR[sm_index].element[4].w
                + Image[(SM.SR[sm_index].element[5].x * 3) + (SM.SR[sm_index].element[5].y * W * 3)] * SM.SR[sm_index].element[5].w
                + Image[(SM.SR[sm_index].element[6].x * 3) + (SM.SR[sm_index].element[6].y * W * 3)] * SM.SR[sm_index].element[6].w
                + Image[(SM.SR[sm_index].element[7].x * 3) + (SM.SR[sm_index].element[7].y * W * 3)] * SM.SR[sm_index].element[7].w
                + Image[(SM.SR[sm_index].element[8].x * 3) + (SM.SR[sm_index].element[8].y * W * 3)] * SM.SR[sm_index].element[8].w
                + Image[(SM.SR[sm_index].element[9].x * 3) + (SM.SR[sm_index].element[9].y * W * 3)] * SM.SR[sm_index].element[9].w
                + Image[(SM.SR[sm_index].element[10].x * 3) + (SM.SR[sm_index].element[10].y * W * 3)] * SM.SR[sm_index].element[10].w
                + Image[(SM.SR[sm_index].element[11].x * 3) + (SM.SR[sm_index].element[11].y * W * 3)] * SM.SR[sm_index].element[11].w
                + Image[(SM.SR[sm_index].element[12].x * 3) + (SM.SR[sm_index].element[12].y * W * 3)] * SM.SR[sm_index].element[12].w
                + Image[(SM.SR[sm_index].element[13].x * 3) + (SM.SR[sm_index].element[13].y * W * 3)] * SM.SR[sm_index].element[13].w
                + Image[(SM.SR[sm_index].element[14].x * 3) + (SM.SR[sm_index].element[14].y * W * 3)] * SM.SR[sm_index].element[14].w
                + Image[(SM.SR[sm_index].element[15].x * 3) + (SM.SR[sm_index].element[15].y * W * 3)] * SM.SR[sm_index].element[15].w);

            int g_v = (Image[(SM.SR[sm_index].element[0].x * 3) + (SM.SR[sm_index].element[0].y * W * 3) + 1] * SM.SR[sm_index].element[0].w
                + Image[(SM.SR[sm_index].element[1].x * 3) + (SM.SR[sm_index].element[1].y * W * 3) + 1] * SM.SR[sm_index].element[1].w
                + Image[(SM.SR[sm_index].element[2].x * 3) + (SM.SR[sm_index].element[2].y * W * 3) + 1] * SM.SR[sm_index].element[2].w
                + Image[(SM.SR[sm_index].element[3].x * 3) + (SM.SR[sm_index].element[3].y * W * 3) + 1] * SM.SR[sm_index].element[3].w
                + Image[(SM.SR[sm_index].element[4].x * 3) + (SM.SR[sm_index].element[4].y * W * 3) + 1] * SM.SR[sm_index].element[4].w
                + Image[(SM.SR[sm_index].element[5].x * 3) + (SM.SR[sm_index].element[5].y * W * 3) + 1] * SM.SR[sm_index].element[5].w
                + Image[(SM.SR[sm_index].element[6].x * 3) + (SM.SR[sm_index].element[6].y * W * 3) + 1] * SM.SR[sm_index].element[6].w
                + Image[(SM.SR[sm_index].element[7].x * 3) + (SM.SR[sm_index].element[7].y * W * 3) + 1] * SM.SR[sm_index].element[7].w
                + Image[(SM.SR[sm_index].element[8].x * 3) + (SM.SR[sm_index].element[8].y * W * 3) + 1] * SM.SR[sm_index].element[8].w
                + Image[(SM.SR[sm_index].element[9].x * 3) + (SM.SR[sm_index].element[9].y * W * 3) + 1] * SM.SR[sm_index].element[9].w
                + Image[(SM.SR[sm_index].element[10].x * 3) + (SM.SR[sm_index].element[10].y * W * 3) + 1] * SM.SR[sm_index].element[10].w
                + Image[(SM.SR[sm_index].element[11].x * 3) + (SM.SR[sm_index].element[11].y * W * 3) + 1] * SM.SR[sm_index].element[11].w
                + Image[(SM.SR[sm_index].element[12].x * 3) + (SM.SR[sm_index].element[12].y * W * 3) + 1] * SM.SR[sm_index].element[12].w
                + Image[(SM.SR[sm_index].element[13].x * 3) + (SM.SR[sm_index].element[13].y * W * 3) + 1] * SM.SR[sm_index].element[13].w
                + Image[(SM.SR[sm_index].element[14].x * 3) + (SM.SR[sm_index].element[14].y * W * 3) + 1] * SM.SR[sm_index].element[14].w
                + Image[(SM.SR[sm_index].element[15].x * 3) + (SM.SR[sm_index].element[15].y * W * 3) + 1] * SM.SR[sm_index].element[15].w);

            int r_v = (Image[(SM.SR[sm_index].element[0].x * 3) + (SM.SR[sm_index].element[0].y * W * 3) + 2] * SM.SR[sm_index].element[0].w
                + Image[(SM.SR[sm_index].element[1].x * 3) + (SM.SR[sm_index].element[1].y * W * 3) + 2] * SM.SR[sm_index].element[1].w
                + Image[(SM.SR[sm_index].element[2].x * 3) + (SM.SR[sm_index].element[2].y * W * 3) + 2] * SM.SR[sm_index].element[2].w
                + Image[(SM.SR[sm_index].element[3].x * 3) + (SM.SR[sm_index].element[3].y * W * 3) + 2] * SM.SR[sm_index].element[3].w
                + Image[(SM.SR[sm_index].element[4].x * 3) + (SM.SR[sm_index].element[4].y * W * 3) + 2] * SM.SR[sm_index].element[4].w
                + Image[(SM.SR[sm_index].element[5].x * 3) + (SM.SR[sm_index].element[5].y * W * 3) + 2] * SM.SR[sm_index].element[5].w
                + Image[(SM.SR[sm_index].element[6].x * 3) + (SM.SR[sm_index].element[6].y * W * 3) + 2] * SM.SR[sm_index].element[6].w
                + Image[(SM.SR[sm_index].element[7].x * 3) + (SM.SR[sm_index].element[7].y * W * 3) + 2] * SM.SR[sm_index].element[7].w
                + Image[(SM.SR[sm_index].element[8].x * 3) + (SM.SR[sm_index].element[8].y * W * 3) + 2] * SM.SR[sm_index].element[8].w
                + Image[(SM.SR[sm_index].element[9].x * 3) + (SM.SR[sm_index].element[9].y * W * 3) + 2] * SM.SR[sm_index].element[9].w
                + Image[(SM.SR[sm_index].element[10].x * 3) + (SM.SR[sm_index].element[10].y * W * 3) + 2] * SM.SR[sm_index].element[10].w
                + Image[(SM.SR[sm_index].element[11].x * 3) + (SM.SR[sm_index].element[11].y * W * 3) + 2] * SM.SR[sm_index].element[11].w
                + Image[(SM.SR[sm_index].element[12].x * 3) + (SM.SR[sm_index].element[12].y * W * 3) + 2] * SM.SR[sm_index].element[12].w
                + Image[(SM.SR[sm_index].element[13].x * 3) + (SM.SR[sm_index].element[13].y * W * 3) + 2] * SM.SR[sm_index].element[13].w
                + Image[(SM.SR[sm_index].element[14].x * 3) + (SM.SR[sm_index].element[14].y * W * 3) + 2] * SM.SR[sm_index].element[14].w
                + Image[(SM.SR[sm_index].element[15].x * 3) + (SM.SR[sm_index].element[15].y * W * 3) + 2] * SM.SR[sm_index].element[15].w);
            */

            if (b_v < 0) {
                b_v = 0;
            }
            if (b_v > 255) {
                b_v = 255;
            }
            if (g_v < 0) {
                g_v = 0;
            }
            if (g_v > 255) {
                g_v = 255;
            }
            if (r_v < 0) {
                r_v = 0;
            }
            if (r_v > 255) {
                r_v = 255;
            }


            Output[out_index] = b_v;
            Output[out_index + 1] = g_v;
            Output[out_index + 2] = r_v;

        }

    }

    end = clock();
    printf("BICUBIC 연산: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);
}


void padding_initialize(float* Padding, float* Image, int PW, int PH, int pad) {

    clock_t padding_S = clock();
    //bitmap 파일은 상하 반전으로 저장되기 때문에 padding과 동시에 상하 반전되도록 구현 성공
    int W = PW - (pad * 2);
    int H = PH - (pad * 2);
    // Zero Padding
    /*
    for (int i = 0; i < PH; i++) {
        for (int j = 0; j < PW; j++) {
            Padding[i * PW + j] = 0;
        }
    }
    */
    // Origin Pixel Value 
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            Padding[(i + pad) * PW + (j + pad)] = Image[i * W + j]; // y

        }
    }

    clock_t padding_E = clock();
    printf("패딩 소요 시간: %.3f\n", (float)(padding_E - padding_S) / CLOCKS_PER_SEC);
}

void transpose_padding_initialize(float* Padding, float* Image, int W, int H, int PW, int PH, int pad, int stride) {

    clock_t padding_S = clock();


    // Origin Pixel Value 
    int px = pad;
    int py = pad;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            Padding[py * PW + px] = Image[i * W + j];
            px = px + stride;
        }
        px = pad;
        py = py + stride;
    }

    clock_t padding_E = clock();
    printf("패딩 소요 시간: %.3f\n", (float)(padding_E - padding_S) / CLOCKS_PER_SEC);
}

void padding_initialize_b(float* Padding, byte* Image, int PW, int PH, int pad) {

    clock_t padding_S = clock();
    //bitmap 파일은 상하 반전으로 저장되기 때문에 padding과 동시에 상하 반전되도록 구현 성공
    int W = PW - (pad * 2);
    int H = PH - (pad * 2);
    // Zero Padding
    for (int i = 0; i < PH; i++) {
        for (int j = 0; j < PW; j++) {
            Padding[i * PW + j] = 0;
        }
    }

    // Origin Pixel Value 
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            Padding[(i + pad) * PW + (j + pad)] = Image[i * W + j] / 255.0; // y
        }
    }

    clock_t padding_E = clock();
    printf("패딩 소요 시간: %.3f\n", (float)(padding_E - padding_S) / CLOCKS_PER_SEC);
}

void realign(BYTE* Y, BYTE* Image, int W, int H) {
    clock_t start = clock();
    // Origin Pixel Value 
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            //Row 반전 동시에
            Y[i * W * 3 + j * 3] = Image[(H - i - 1) * W * 3 + j * 3]; // Blue
            Y[i * W * 3 + j * 3 + 1] = Image[(H - i - 1) * W * 3 + j * 3 + 1]; // Green
            Y[i * W * 3 + j * 3 + 2] = Image[(H - i - 1) * W * 3 + j * 3 + 2]; // Red
        }
    }
    printf("이미지 상하반전 소요 시간: %.3f\n", (float)(clock() - start) / CLOCKS_PER_SEC);
}

void realign_F(BYTE* Y, float* Image, int W, int H) {
    clock_t start = clock();
    // Origin Pixel Value 
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            //Row 반전 동시에
            Y[i * W * 3 + j * 3] = Image[(H - i - 1) * W * 3 + j * 3]; // Blue
            Y[i * W * 3 + j * 3 + 1] = Image[(H - i - 1) * W * 3 + j * 3 + 1]; // Green
            Y[i * W * 3 + j * 3 + 2] = Image[(H - i - 1) * W * 3 + j * 3 + 2]; // Red
        }
    }
    printf("이미지 상하반전 소요 시간: %.3f\n", (float)(clock() - start) / CLOCKS_PER_SEC);
}

void SaveBMPFile(BITMAPFILEHEADER hf, BITMAPINFOHEADER hInfo,
    RGBQUAD* hRGB, BYTE* Output, int W, int H, const char* FileName, int scale)
{
    clock_t start = clock();
    hInfo.biWidth = W;
    hInfo.biHeight = H;
    //printf("%d, %d\n", W, H);
    FILE* fp = fopen(FileName, "wb");
    if (hInfo.biBitCount == 24) {
        fwrite(&hf, sizeof(BYTE), sizeof(BITMAPFILEHEADER), fp);
        fwrite(&hInfo, sizeof(BYTE), sizeof(BITMAPINFOHEADER), fp);
        fwrite(Output, sizeof(BYTE), (W) * (H) * 3, fp);
    }
    else if (hInfo.biBitCount == 8) {
        fwrite(&hf, sizeof(BYTE), sizeof(BITMAPFILEHEADER), fp);
        fwrite(&hInfo, sizeof(BYTE), sizeof(BITMAPINFOHEADER), fp);
        fwrite(hRGB, sizeof(RGBQUAD), 256, fp);
        fwrite(Output, sizeof(BYTE), W * H, fp);
    }
    fclose(fp);
    printf("이미지 저장 소요 시간: %.3f\n", (float)(clock() - start) / CLOCKS_PER_SEC);
}

// 시간 꽤 걸리는 코드
void SaveFeatureMap(int W, int H, byte* B, int k_n, const char* FileName) { //k_n: kernel list number(kernel 가리키는 숫자)

    clock_t savefm_S = clock(); //featuremap 저장 시간 측정
    FILE* file;
    char kernel_num[100];
    char result_feature[255] = "D:/연구실/SR(202109-202212)/SRCNN/feature/";
    strcat(result_feature, FileName);

    printf("%s\n", result_feature);
    int fResult = mkdir(result_feature);
    printf("%d\n", fResult);
    char feature[300] = "feature0_";
    char number[20] = "";
    strcat(result_feature, strcat(strcat(feature, itoa(k_n, number, 10)), ".txt"));
    printf("%s\n", result_feature);

    file = fopen(result_feature, "wb");
    int index = 0;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            char temp[300];
            index = (x + y * W);
            float k = B[index];
            sprintf(temp, "%f", k);
            //printf("%f\n", k);
            fputs(strcat(temp, " "), file);
        }
        fputs("\n", file);
    }

    fclose(file);

    clock_t savefm_E = clock(); //featuremap 저장 시간 측정
    printf("Featuremap 저장 시간: %.3f\n", (float)(savefm_E - savefm_S) / CLOCKS_PER_SEC);
}

void SaveFeatureMap_f(int W, int H, float* B, int k_n, const char* FileName) { //k_n: kernel list number(kernel 가리키는 숫자)

    clock_t savefm_S = clock(); //featuremap 저장 시간 측정
    FILE* file;
    char kernel_num[100];
    //char result_feature[255] = "C:/Users/siewe/OneDrive/바탕 화면/FSRCNN/feature/";
    char result_feature[255] = "D:/연구실/SR(202109-202212)/FSRCNN/feature/";
    strcat(result_feature, FileName);

    printf("%s\n", result_feature);
    int fResult = mkdir(result_feature);
    printf("%d\n", fResult);
    char feature[300] = "feature0_";
    char number[20] = "";
    strcat(result_feature, strcat(strcat(feature, itoa(k_n, number, 10)), ".txt"));
    printf("%s\n", result_feature);

    file = fopen(result_feature, "wb");
    int index = 0;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            char temp[300];
            index = (x + y * W);
            float k = B[index];
            sprintf(temp, "%f", k);
            //printf("%f\n", k);
            fputs(strcat(temp, " "), file);
        }
        fputs("\n", file);
    }

    fclose(file);

    clock_t savefm_E = clock(); //featuremap 저장 시간 측정
    printf("Featuremap 저장 시간: %.3f\n", (float)(savefm_E - savefm_S) / CLOCKS_PER_SEC);
}

// 9x9 kernel
int getMatrix_9(const char* filename, float(*matrix)[9])
{

    clock_t getkernel_S = clock();
    FILE* stream;

    int chk = 0;
    int i = 0, j = 0;
    int mj = 0;
    stream = fopen(filename, "r");
    if (stream == NULL)
        return 1;

    while (chk != EOF)
    {
        j = 0;
        chk = 0;
        while (chk != '\n' && chk != EOF)
        {
            fscanf(stream, "%f", &matrix[i][j]);
            chk = fgetc(stream);
            fseek(stream, -1, SEEK_CUR);
            j++;
        }
        i++;
        if (mj < j)
            mj = j;
    }

    fclose(stream);
    clock_t getkernel_E = clock();
    printf("커널 불러오기 소요 시간(9x9): %.3f\n", getkernel_E - getkernel_S);

    return 0;

}

// 5x5 kernel
int getMatrix_5(const char* filename, float(*matrix)[5])
{
    clock_t start = clock();
    FILE* stream;

    int chk = 0;
    int i = 0, j = 0;
    int mj = 0;
    stream = fopen(filename, "r");
    if (stream == NULL)
        return 1;

    while (chk != EOF)
    {
        j = 0;
        chk = 0;
        while (chk != '\n' && chk != EOF)
        {
            fscanf(stream, "%f", &matrix[i][j]);
            printf("matrix[%d][%d]: %f\n", i, j, matrix[i][j]);
            chk = fgetc(stream);
            fseek(stream, -1, SEEK_CUR);
            j++;
        }
        i++;
        if (mj < j)
            mj = j;
    }

    fclose(stream);
    clock_t end = clock();
    printf("커널 불러오기 소요 시간(5x5): %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);
    return 0;
}

// 30초 잡아먹는 코드
int getFeatureMap(int W, int H, const char* filename, float* matrix)
{
    clock_t start = clock();
    FILE* stream;

    int chk = 0;
    int i = 0, j = 0;
    int mj = 0;
    stream = fopen(filename, "r");
    if (stream == NULL)
        return 1;

    while (chk != EOF)
    {
        j = 0;
        chk = 0;
        while (chk != '\n' && chk != EOF)
        {
            fscanf(stream, "%f", &matrix[i * W + j]);
            chk = fgetc(stream);
            fseek(stream, -1, SEEK_CUR);
            j++;
        }
        i++;
        if (mj < j)
            mj = j;
    }

    fclose(stream);
    clock_t end = clock();
    printf("feature map 불러오기 소요 시간: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);
    return 0;
}

int getBias(const char* filename, float* bias) {

    clock_t getbias_S = clock();

    FILE* stream;

    int chk = 0;
    int i = 0, j = 0;
    int mj = 0;
    stream = fopen(filename, "r");
    if (stream == NULL) {
        printf("BIAS FILE NOT FOUND!!\n");
        return 1;
    }

    while (chk != EOF)
    {
        j = 0;
        chk = 0;
        while (chk != '\n' && chk != EOF)
        {
            fscanf(stream, "%f", &bias[i]);
            //printf("bias[%d]: %lf\n",i, bias[i]);
            chk = fgetc(stream);
            fseek(stream, -1, SEEK_CUR);
            j++;
        }
        i++;
        if (mj < j)
            mj = j;
    }
    fclose(stream);

    return 0;

    clock_t getbias_E = clock();
    printf("bias GET 소요 시간: %.3f\n", (float)(getbias_E - getbias_S) / CLOCKS_PER_SEC);
}

void getbias_(const char* filename, float* bias) {
    clock_t start = clock();
    printf("%s\n", filename);
    FILE* fp = NULL;
    int i = 0;
    if (fopen_s(&fp, filename, "rt") == 0) {
        while (fscanf_s(fp, "%f", &bias[i]) == 1) {
            //printf("확인1 %f\n",bias[i]);
            //printf("bias[%d]: %f\n", i, bias[i]);
            i++;
        }
        fclose(fp);
        printf("count: %d\n", i);
    }
    else {
        printf("FILE NOT FOUND!\n");
    }
    clock_t end = clock();
    printf("get bias 소요시간: %f\n", (float)(end - start) / CLOCKS_PER_SEC);
}
_minmax getkernel_(const char* filename, float* bias) {
    clock_t start = clock();
    printf("%s\n", filename);
    FILE* fp = NULL;
    int i = 0;
    _minmax cur;
    cur.mincc =5;
    cur.maxcc = -1;
    if (fopen_s(&fp, filename, "rt") == 0) {
        while (fscanf_s(fp, "%f", &bias[i]) == 1) {
            if (bias[i] > cur.maxcc) {
                cur.maxcc = bias[i];
                printf("이거야%f\n", bias[i]);
            }
            if (abs(bias[i]) < abs(cur.mincc))
                cur.mincc = abs(bias[i]);
            //printf("확인1 %f\n",bias[i]);
            //printf("bias[%d]: %f\n", i, bias[i]);
            i++;
        }
        fclose(fp);
        printf("count: %d\n", i);
    }
    else {
        printf("FILE NOT FOUND!\n");
    }
    clock_t end = clock();
    printf("get bias 소요시간: %f\n", (float)(end - start) / CLOCKS_PER_SEC);
    return cur;
}


void Matmul(float* Image, float* Output, float* matrix, int PW, int PH, int ksize, int pad)
{
    clock_t start = clock();
    int W = PW - (pad * 2);
    int H = PH - (pad * 2);

    for (int Y = 0; Y < W * H; Y++) {
        float temp = 0;
        for (int X = 0; X < PW * PH; X++) {
            temp += matrix[X + Y * (PW * PH)] * Image[X];
        }
        Output[Y] = temp;
        //printf("output: %f\n", Output[Y]);
    }

    clock_t end = clock();
    printf("MatMuL 소요 시간: %f\n", (float)(end - start) / CLOCKS_PER_SEC);
}

void MakeMatrix_9(float(*kernel)[9], float* matrix, int W, int H, int padding, int ksize) {
    clock_t start;
    clock_t end;
    int PW = W + (padding * 2);
    int PH = H + (padding * 2);
    int width = PW * PH;
    int height = W * H;
    printf("PW, PH: %d, %d\n", PW, PH);
    start = clock();

    for (int y = 0; y < H; y++) {
        int xcnt = PW * y;
        for (int x = 0; x < W; x++) { // matmul y축(output 사이즈)
            int index = xcnt;

            for (int ky = 0; ky < ksize; ky++) {
                for (int kx = 0; kx < ksize; kx++) {

                    //printf("--%d\n", index);
                    matrix[(y * W + x) * width + index] = kernel[ky][kx];

                    //printf("%d\n", (y * W + x)* width + index);
                    index++;
                }
                index = index + (PW - ksize);
            }
            xcnt = xcnt + 1;
        }
    }


    end = clock();
    printf("Matrix 생성 소요 시간: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);

}

void MakeMatrix_5(float(*kernel)[5], float* matrix, int W, int H, int padding, int ksize) {
    clock_t start;
    clock_t end;
    int PW = W + (padding * 2);
    int PH = H + (padding * 2);
    int width = PW * PH;
    int height = W * H;

    start = clock();
    int cnt = 0;

    for (int y = 0; y < H; y++) {
        int xcnt = PW * y;
        for (int x = 0; x < W; x++) { // matmul y축(output 사이즈)
            int index = xcnt;

            for (int ky = 0; ky < ksize; ky++) {
                for (int kx = 0; kx < ksize; kx++) {

                    //printf("--%d\n", index);
                    matrix[(y * W + x) * width + index] = kernel[ky][kx];

                    //printf("%d\n", (y * W + x)* width + index);
                    index++;
                }
                index = index + (PW - ksize);
            }
            xcnt = xcnt + 1;
        }
    }
    end = clock();
    printf("Matrix 생성 소요 시간: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);

}


/*CONVOLUTION*/
/*
void Matmul_9(float* Image, float* Output, float(*kernel)[9], int PW, int PH, int ksize, int pad)
{
    clock_t start = clock();
    int W = PW - (pad * 2);
    int H = PH - (pad * 2);

    for (int y = pad; y < H + pad; y++) {
        for (int x = pad; x < W + pad; x++) {
            int index = (y - pad) * W + (x - pad);
            for (int ky = -(ksize/2); ky < (ksize/2)+1; ky++) {
                for (int kx = -(ksize / 2); kx < (ksize / 2) + 1; kx++) {
                    Output[index] += Image[x + kx + (y + ky) * PW] * kernel[ky + (ksize/2)][kx + (ksize / 2)];
                }
            }
        }
    }
    clock_t end = clock();
    printf("MatMuL 소요 시간: %f\n", (float)(end - start) / CLOCKS_PER_SEC);
}

void Matmul_5(float* Image, float* Output, float(*kernel)[5], int PW, int PH, int ksize, int pad)
{
    clock_t start = clock();
    int W = PW - (pad * 2);
    int H = PH - (pad * 2);

    for (int y = pad; y < H + pad; y++) {
        for (int x = pad; x < W + pad; x++) {
            int index = (y - pad) * W + (x - pad);
            for (int ky = -(ksize / 2); ky < (ksize / 2) + 1; ky++) {
                for (int kx = -(ksize / 2); kx < (ksize / 2) + 1; kx++) {
                    Output[index] += Image[x + kx + (y + ky) * PW] * kernel[ky + (ksize / 2)][kx + (ksize / 2)];
                }
            }
        }
    }
    clock_t end = clock();
    printf("MatMuL 소요 시간: %f\n", (float)(end - start) / CLOCKS_PER_SEC);
}
*/