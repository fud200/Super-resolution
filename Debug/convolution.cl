// TODO: Add OpenCL kernel code here.

typedef struct _element {
	int cols;
	float w;
}_element;

typedef struct _SR {
   struct _element element;
};

typedef struct _SM {
    int n;
    struct _SR* SR;
};

typedef struct _Output {
    float* feature_map;
};

typedef struct _Output_B {
    unsigned char* feature_map;
};
__kernel void convt(__global float* dev_Padding, __global float* dev_mykernel, __global float* dev_TempY, int OH, int OW, int PW, int k, int ksize, int y3) {
    //int t = blockIdx.x * blockDim.x + threadIdx.x;
    //int z = blockIdx.y * blockDim.y + threadIdx.y;
    printf("도냐");
    int t=get_global_id(0);
	int z=get_global_id(1);
    if(z < OH) {
        int h_start = z * PW;
        int h_end = h_start + k;
        y3 = OW * z;
        if(t < OW) {
            int w_start = t;
            int w_end = w_start + ksize;
            float sumb = 0;
            int kx = 0;
            int ky = 0;
            for (int i = h_start; i < h_end; i += PW) {
                for (int j = w_start; j < w_end; j++) {
                    sumb = sumb + dev_mykernel[ky + kx] * (float)(dev_Padding[i + j]);
                    kx++;
                }
                kx = 0;
                ky += ksize;
            }
            dev_TempY[y3 + t] = sumb;
        }
    }
}
__kernel void on_g_mat(__global float* dev_Padding, __global float* dev_feature_map, __global float *mykernel, int H, int W, int ksize, int pad) {
    //int t = blockIdx.x * blockDim.x + threadIdx.x;
    //int z = blockIdx.y * blockDim.y + threadIdx.y; 
    int t=get_global_id(0);
	int z=get_global_id(1);
    if(z < H && t < W) {
        _element sr[25];
        for (int i = 0; i < ksize; i++) { // y�� mykernel size
            for (int j = 0; j < ksize; j++) { // x�� mykernel size
                sr[i * ksize + j].cols = (t + ((W + pad + pad) * i) + j + (z * (W + pad + pad)));
                sr[i * ksize + j].w = mykernel[i * ksize + j];
            }
        }

        float sumb = 0;
        sumb = 0;
        for (int j = 0; j < ksize; j++) {
            for (int k = 0; k < ksize; k++) {
                sumb = sumb + sr[j * ksize + k].w * (float)(dev_Padding[sr[j * ksize + k].cols]);
            }
        }
        dev_feature_map[z * W + t] = sumb;
    }
}
__kernel void conv1(__global float* dev_prefeature_map, __global float* dev_feature_map, __global float *mykernel, int H, int W) {
    //int t = blockIdx.x * blockDim.x + threadIdx.x;
    //int z = blockIdx.y * blockDim.y + threadIdx.y; 
    int t=get_global_id(0);
	int z=get_global_id(1);
    
    if(z < H && t < W) {
        dev_feature_map[z * W + t] = (mykernel[0] * (float)(dev_prefeature_map[z * W + t]));
    }
}
