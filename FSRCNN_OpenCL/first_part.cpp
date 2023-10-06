
#include "deviceinfo.h"
#include "util.h"

int scale = 2;

int main() {
    CLInit();
    float maxc = -1;
    float minc = 1;
    clock_t getimage_S = clock();

    char file_dir[255] = "./input/SR_Train_spp32_camera0.bmp";
    BITMAPFILEHEADER hf; // 14바이트
    BITMAPINFOHEADER hInfo; // 40바이트
    RGBQUAD hRGB[256]; // 1024바이트
    FILE* fp;
    fp = fopen(file_dir, "rb");
    printf("%s\n", file_dir);

    if (fp == NULL) {
        printf("File not found!\n");
        return -1;
    }

    fread(&hf, sizeof(BITMAPFILEHEADER), 1, fp);
    fread(&hInfo, sizeof(BITMAPINFOHEADER), 1, fp);
    float first_conv;
    float first_prelu;
    float conv2_1;
    float prelu2_1;
    float conv2_2;
    float prelu2_2;
    float conv2_3;
    float prelu2_3;
    float conv2_4;
    float prelu2_4;
    float conv2_5;
    float prelu2_5;
    float conv2_6;
    float prelu2_6;
    float conv_l;
    float biast;
    int ImgSize = hInfo.biWidth * hInfo.biHeight;
    int size = hInfo.biSizeImage; // 픽셀 데이터 크기
    int H = hInfo.biHeight, W = hInfo.biWidth;

    BYTE* Image;

    if (hInfo.biBitCount == 24) { // 트루컬러
        Image = (BYTE*)malloc(ImgSize * 3);
        fread(Image, sizeof(BYTE), ImgSize * 3, fp);
    }

    else { // 인덱스(그레이)
        fread(hRGB, sizeof(RGBQUAD), 256, fp);
        Image = (BYTE*)malloc(ImgSize);
        fread(Image, sizeof(BYTE), ImgSize, fp);
    }

    clock_t getimage_E = clock();
    printf("이미지 GET 소요 시간: \t%.3f\n", (float)(getimage_E - getimage_S) / CLOCKS_PER_SEC);

    int outsize = W * H * scale * scale;
    int OW = W * scale;
    int OH = H * scale;

    // bicubic 으로 upscaling 후 Cb, Cr 값 추출
    BYTE* bc_output = (BYTE*)malloc(outsize * 3);
    _bcSM SM; // Sparse Matrix 선언
    SM.n = scale * scale * W * H;
    SM.SR = (_bcSR*)calloc(SM.n, sizeof(_bcSR));
    BC_MakeSparseMatrix(Image, bc_output, SM, W, H, (float)scale);
    TESTBC(Image, bc_output, W, H, ImgSize, (float)scale, SM);

    // 상하 반전 제거
    BYTE* realign_ = (BYTE*)malloc(sizeof(BYTE) * ImgSize * 3);
    BYTE* bc_realign = (BYTE*)malloc(sizeof(BYTE) * outsize * 3);

    realign(realign_, Image, W, H);
    realign(bc_realign, bc_output, W * scale, H * scale);

    // bgr --> YCbCr로 변환하기(Y만 쓸 것)
    float* Y = (float*)malloc(sizeof(float) * ImgSize); // YCbCr 중에 Y만 사용
    float* Cb = (float*)malloc(sizeof(float) * outsize); // YCbCr 중에 Y만 사용
    float* Cr = (float*)malloc(sizeof(float) * outsize); // YCbCr 중에 Y만 사용

    clock_t toycbcr_S = clock();
    int k = 0;
    for (int i = 0; i < ImgSize; i++) {
        Y[i] = (16.0 + (realign_[k] * 25.064 + realign_[k + 1] * 129.057 + realign_[k + 2] * 64.738) / 256.0) / 255.0;
        k += 3;
    }
    k = 0;
    for (int i = 0; i < ImgSize * scale * scale; i++) {
        Cb[i] = 128.0 + (bc_realign[k] * 112.439 + bc_realign[k + 1] * -74.494 + bc_realign[k + 2] * -37.945) / 256.0;
        Cr[i] = 128.0 + (bc_realign[k] * -18.285 + bc_realign[k + 1] * -94.154 + bc_realign[k + 2] * 112.439) / 256.0;
        k += 3;
    }
    clock_t toycbcr_E = clock();
    printf("채널 변환 소요시간: %.3f\n", (float)(toycbcr_E - toycbcr_S) / CLOCKS_PER_SEC);

    /* first part */
    clock_t totalS = clock();

    printf("\n========== FIRST PART ==========\n");
    // conv1 layer
    printf("\nCONV1 LAYER\n");
    int kn1 = 56;
    int ks1 = 5;
    float* kernel1 = (float*)malloc(sizeof(float) * ks1 * ks1);
    int stride1 = 1;
    int pad1 = 2;
    int PW1 = W + (pad1 * 2);
    int PH1 = H + (pad1 * 2);
    float* Padding1 = (float*)malloc(sizeof(float) * PW1 * PH1);
    memset(Padding1, 0, sizeof(float) * PW1 * PH1);
    padding_initialize(Padding1, Y, PW1, PH1, pad1);

    char buffer[300];
    // bias 불러오기
    float bias1[56];
    getbias_("./parameter/bias/first/bias0_0.txt", bias1);

    // PReLU weight 불러오기
    float PReLU_weight1[56];
    getbias_("./parameter/PReLU_weight/first/weight0_0.txt", PReLU_weight1);

    // first_part: Convolution 연산 수행(conv1: kernel 56개)
    clock_t conv1_s = clock();
    _Output output1[56];

    for (int fn = 0; fn < kn1; fn++) {

        output1[fn].feature_map = (float*)malloc(sizeof(float) * ImgSize);
        memset(output1[fn].feature_map, 0, sizeof(float) * ImgSize);

        // kernel 1개씩 불러오기 
        char kernelfile_dir[255] = "./parameter/kernel/first/weight0_";
        strcat(kernelfile_dir, strcat(itoa(fn, buffer, 10), ".txt"));
        _minmax cur;
        cur=getkernel_(kernelfile_dir, kernel1);
        if (maxc < cur.maxcc)
            maxc = cur.maxcc;
        if (minc > cur.mincc)
            minc = cur.mincc;
        //printf("확인 %f\n", kernel1[0]);
        //printf("확인 %f\n", Padding1[200]);
        //CLInit();

        clock_t start;
        clock_t end;

        start = clock();
        //conv 5x5
        bufferWrite(Padding1, kernel1,output1[fn].feature_map);
        runKernel(output1[fn].feature_map, W, H, ks1, pad1);
        //matmul_5(Padding1, kernel1, output1[fn].feature_map, H, W, ks1, pad1);
        //for (int z = 0; z < H; z++) { // z: Height, y축
        //    for (int t = 0; t < W; t++) { // t: Width, x축
        //        _element sr[25];
        //        for (int i = 0; i < ks1; i++) { // y축 kernel size
        //            for (int j = 0; j < ks1; j++) { // x축 kernel size
        //                sr[i * ks1 + j].cols = (t + ((W + pad1 + pad1) * i) + j + (z * (W + pad1 + pad1)));
        //                sr[i * ks1 + j].w = kernel1[i * ks1 + j];
        //            }
        //        }

        //        float sumb = 0;
        //        sumb = 0;
        //        for (int j = 0; j < ks1; j++) {
        //            for (int k = 0; k < ks1; k++) {
        //                sumb = sumb + sr[j * ks1 + k].w * (float)Padding1[sr[j * ks1 + k].cols];
        //            }
        //        }
        //        output1[fn].feature_map[z * W + t] = sumb;

        //    }
        //}
        /*
        int y1 = 0;
        int k = W * ks1;
        for (int z = 0; z < H; z++) { // z: Height, y축
            int h_start = z * PW1 * 1;
            int h_end = h_start + k;
            for (int t = 0; t < W; t++) { // t: Width, x축
                int w_start = t * 1;
                int w_end = w_start + ks1;
                float sumb = 0;
                int kx = 0;
                int ky = 0;
                for (int i = h_start; i < h_end; i += PW1) {
                    for (int j = w_start; j < w_end; j++) {
                        sumb = sumb + kernel1[ky + kx] * float(Padding1[i + j]);
                        //printf("kernel index: %d, padding[%d]: %f\n", kx + ky, i + j, Padding3[i + j]);
                        kx++;

                    }
                    kx = 0;
                    ky += ks1;
                }
                output1[fn].feature_map[y1 + t] = sumb;
            }
            y1 += W;
        }
        */
        end = clock();
        printf("convolution: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);
        int curr;
        first_conv = (float)(end - start) / CLOCKS_PER_SEC;


        // bias 더하기 진행 후 PReLU 진행
        clock_t relu_S = clock();
        for (int index = 0; index < ImgSize; index++) {
            if (output1[fn].feature_map[index] + bias1[fn] >= 0) { output1[fn].feature_map[index] = output1[fn].feature_map[index] + bias1[fn]; }
            else { output1[fn].feature_map[index] = PReLU_weight1[fn] * (output1[fn].feature_map[index] + bias1[fn]); 
            //printf("확인좀2 %f\n", output1[fn].feature_map[index]);
            //여기까진 오케
            }

        }

        clock_t relu_E = clock();
        printf("Bias 덧셈 후 ReLU 소요 시간: %.3f\n", (float)(relu_E - relu_S) / CLOCKS_PER_SEC);
        first_prelu = (float)(relu_E - relu_S) / CLOCKS_PER_SEC;
        //SaveFeatureMap_f(W, H, output1[fn].feature_map, fn, "first_part/");


    }
    free(Y);
    free(realign_);
    free(Padding1);
    free(Image);
    clock_t conv1_e = clock();
    printf("convolution layer 1 소요 시간: %.3f\n", (float)(conv1_e - conv1_s) / CLOCKS_PER_SEC);


    /* mid part */
    printf("\n========== MID PART ==========\n");

    // conv1 layer
    printf("\nCONV1 LAYER\n");
    int kn2_1 = 12;
    int ks2_1 = 1;
    float* kernel2_1 = (float*)malloc(sizeof(float) * ks2_1 * ks2_1);
    int stride2_1 = 1;

    // bias 불러오기
    float bias2_1[12];
    getbias_("./parameter/bias/mid/bias0_0.txt", bias2_1);

    // PReLU weight 불러오기
    float PReLU_weight2_1[12];
    getbias_("./parameter/PReLU_weight/mid/weight0_0.txt", PReLU_weight2_1);

    // conv1 layer: Convolution 연산 수행(conv1: kernel 56개)
    clock_t conv2_S = clock();
    _Output output2_1[12];

    for (int on = 0; on < kn2_1; on++) {
        output2_1[on].feature_map = (float*)malloc(sizeof(float) * ImgSize);
        memset(output2_1[on].feature_map, 0, sizeof(float) * ImgSize);

        for (int kn = 0; kn < kn1; kn++) {

            float* TempY = (float*)malloc(sizeof(float) * ImgSize);
            memset(TempY, 0, sizeof(float) * ImgSize);

            char onnum[10];
            char temponnum[10];
            itoa(on, temponnum, 10);
            itoa(on, onnum, 10);
            char kernelfile_dir[255] = "./parameter/kernel/mid/conv0/conv0_";

            strcat(kernelfile_dir, strcat(onnum, "/"));
            char buffer[30];
            char kernel[30] = "kernel";
            itoa(kn, buffer, 10);
            printf("%d\n", kn);
            char sign[10] = "_";
            strcat(kernelfile_dir, strcat(kernel, strcat(buffer, strcat(strcat(sign, temponnum), ".txt"))));
            printf("kernel file: %s\n", kernelfile_dir);
            _minmax cur;
            cur=getkernel_(kernelfile_dir, kernel2_1);
            if (maxc < cur.maxcc)
                maxc = cur.maxcc;
            if (minc > cur.mincc)
                minc = cur.mincc;

            clock_t start;
            clock_t end;

            start = clock();
            //여기 바꿔
            // int cnt = 0;
            bufferWrite_1(output1[kn].feature_map, kernel2_1, TempY);
            runKernel_1(TempY, W, H);
            //for (int z = 0; z < H; z++) { // z: Height, y축
            //    for (int t = 0; t < W; t++) { // t: Width, x축
            //        TempY[z * W + t] = output1[kn].feature_map[z * W + t] * kernel2_1[0];
            //    }
            //}
            end = clock();
            //printf("convolve 소요 시간: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);
            //float conv2_1 = (float)(end - start) / CLOCKS_PER_SEC;

            clock_t plus_S = clock();
            for (int index = 0; index < ImgSize; index++) {
                output2_1[on].feature_map[index] = output2_1[on].feature_map[index] + TempY[index];
            }
            //printf("convolution 연산 소요 시간: %.3f\n", (float)(clock() - start) / CLOCKS_PER_SEC);
            conv2_1 = (float)(clock() - plus_S) / CLOCKS_PER_SEC;
            free(TempY);
        }
        clock_t relu2_s = clock();

        for (int index = 0; index < ImgSize; index++) {
            if (output2_1[on].feature_map[index] + bias2_1[on] >= 0) { output2_1[on].feature_map[index] = output2_1[on].feature_map[index] + bias2_1[on]; }
            else { 
                //printf("확인좀 하죠, %f\n", output2_1[on].feature_map[index]); 
                output2_1[on].feature_map[index] = PReLU_weight2_1[on] * (output2_1[on].feature_map[index] + bias2_1[on]);
            }

        }
        //SaveFeatureMap_f(W, H, output2_1[on].feature_map, on, "mid_part/conv1/");
        prelu2_1 = (float)(clock() - relu2_s) / CLOCKS_PER_SEC;
        printf("ReLU 및 bias 덧셈 소요 시간: %.3f\n", (float)(clock() - relu2_s) / CLOCKS_PER_SEC);

    }
    clock_t conv2_e = clock();
    printf("MID PART - conv layer 1 소요 시간: %.3f\n", (float)(conv2_e - conv2_S) / CLOCKS_PER_SEC);

    // conv2 layer
    printf("\nCONV2 LAYER\n");
    int kn2_2 = 12;
    int ks2_2 = 3;
    float* kernel2_2 = (float*)malloc(sizeof(float) * ks2_2 * ks2_2);
    int stride2_2 = 1;
    int pad2_2 = 1;
    int PW2_2 = W + (pad2_2 * 2);
    int PH2_2 = H + (pad2_2 * 2);

    // bias 불러오기
    float bias2_2[12];
    getbias_("./parameter/bias/mid/bias0_1.txt", bias2_2);

    // PReLU weight 불러오기
    float PReLU_weight2_2[12];
    getbias_("./parameter/PReLU_weight/mid/weight0_1.txt", PReLU_weight2_2);

    // conv2 layer: Convolution 연산 수행(conv1: kernel 56개)
    clock_t conv2_2S = clock();
    _Output output2_2[12];

    for (int on = 0; on < kn2_2; on++) {
        output2_2[on].feature_map = (float*)malloc(sizeof(float) * ImgSize);
        memset(output2_2[on].feature_map, 0, sizeof(float) * ImgSize);

        for (int kn = 0; kn < kn2_1; kn++) {

            float* Padding2_2 = (float*)malloc(sizeof(float) * PW2_2 * PH2_2);
            float* TempY = (float*)malloc(sizeof(float) * ImgSize);
            memset(Padding2_2, 0, sizeof(float) * PW2_2 * PH2_2);
            memset(TempY, 0, sizeof(float) * ImgSize);

            char onnum[10];
            char temponnum[10];
            itoa(on, temponnum, 10);
            itoa(on, onnum, 10);
            char kernelfile_dir[255] = "./parameter/kernel/mid/conv1/conv1_";
            strcat(kernelfile_dir, strcat(onnum, "/"));
            char buffer[30];
            char kernel[30] = "kernel";
            itoa(kn, buffer, 10);
            printf("%d\n", kn);
            char sign[10] = "_";
            strcat(kernelfile_dir, strcat(kernel, strcat(buffer, strcat(strcat(sign, temponnum), ".txt"))));
            _minmax cur;
            cur=getkernel_(kernelfile_dir, kernel2_2);
            if (maxc < cur.maxcc)
                maxc = cur.maxcc;
            if (minc > cur.mincc)
                minc = cur.mincc;

            // featuremap 불러오기

            padding_initialize(Padding2_2, output2_1[kn].feature_map, PW2_2, PH2_2, pad2_2);

            clock_t start;
            clock_t end;

            start = clock();
            // 여기 바꿔야대 conv1 3x3


            //matmul_3(Padding2_2, kernel2_2, TempY, H, W, ks2_2, pad2_2);
            bufferWrite(Padding2_2, kernel2_2, TempY);
            runKernel(TempY, W, H, ks2_2, pad2_2);
            //for (int z = 0; z < H; z++) { // z: Height, y축
            //    for (int t = 0; t < W; t++) { // t: Width, x축
            //        _element sr[9];
            //        for (int i = 0; i < ks2_2; i++) { // y축 kernel size
            //            for (int j = 0; j < ks2_2; j++) { // x축 kernel size
            //                sr[i * ks2_2 + j].cols = (t + ((W + pad2_2 + pad2_2) * i) + j + (z * (W + pad2_2 + pad2_2)));
            //                sr[i * ks2_2 + j].w = kernel2_2[i * ks2_2 + j];
            //                // cnt = cnt + 1;
            //            }
            //        }

            //        float sumb = 0;
            //        sumb = 0;
            //        for (int j = 0; j < ks2_2; j++) {
            //            for (int k = 0; k < ks2_2; k++) {
            //                sumb = sumb + sr[j * ks2_2 + k].w * float(Padding2_2[sr[j * ks2_2 + k].cols]);

            //            }
            //        }
            //        TempY[z * W + t] = sumb;
            //    }
            //}
            /*
            int y2_2 = 0;
            int k = W * ks2_2;
            for (int z = 0; z < H; z++) { // z: Height, y축
                int h_start = z * PW2_2 * 1;
                int h_end = h_start + k;
                for (int t = 0; t < W; t++) { // t: Width, x축
                    int w_start = t * 1;
                    int w_end = w_start + ks2_2;
                    float sumb = 0;
                    int kx = 0;
                    int ky = 0;
                    for (int i = h_start; i < h_end; i += PW2_2) {
                        for (int j = w_start; j < w_end; j++) {
                            sumb = sumb + kernel2_2[ky + kx] * float(Padding2_2[i + j]);
                            //printf("kernel index: %d, padding[%d]: %f\n", kx + ky, i + j, Padding3[i + j]);
                            kx++;

                        }
                        kx = 0;
                        ky += ks2_2;
                    }
                    TempY[y2_2 + t] = sumb;
                }
                y2_2 += W;
            }
            */
            clock_t plus_S = clock();
            for (int index = 0; index < ImgSize; index++) {
                output2_2[on].feature_map[index] = output2_2[on].feature_map[index] + TempY[index];
            }

            end = clock();
            conv2_2 = (float)(end - start) / CLOCKS_PER_SEC;
            //printf("convolution 연산 소요 시간: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);

            //free(SM.SR);

            free(Padding2_2);
            free(TempY);
        }
        clock_t relu2_s = clock();

        for (int index = 0; index < ImgSize; index++) {
            if (output2_2[on].feature_map[index] + bias2_2[on] >= 0) { output2_2[on].feature_map[index] = output2_2[on].feature_map[index] + bias2_2[on]; }
            else { output2_2[on].feature_map[index] = PReLU_weight2_2[on] * (output2_2[on].feature_map[index] + bias2_2[on]); }
        }
        //SaveFeatureMap_f(W, H, output2_2[on].feature_map, on, "mid_part/conv2/");
        prelu2_2 = (float)(clock() - relu2_s) / CLOCKS_PER_SEC;
        //printf("ReLU 및 bias 덧셈 소요 시간: %.3f\n", (float)(clock() - relu2_s) / CLOCKS_PER_SEC);

    }

    clock_t conv2_2e = clock();
    printf("MID PART - conv layer 2 소요 시간: %.3f\n", (float)(conv2_2e - conv2_2S) / CLOCKS_PER_SEC);

    // conv3 layer
    printf("\nCONV3 LAYER\n");
    int kn2_3 = 12;
    int ks2_3 = 3;
    float* kernel2_3 = (float*)malloc(sizeof(float) * ks2_3 * ks2_3);
    int stride2_3 = 1;
    int pad2_3 = 1;
    int PW2_3 = W + (pad2_3 * 2);
    int PH2_3 = H + (pad2_3 * 2);

    // bias 불러오기
    float bias2_3[12];
    getbias_("./parameter/bias/mid/bias0_2.txt", bias2_3);

    // PReLU weight 불러오기
    float PReLU_weight2_3[12];
    getbias_("./parameter/PReLU_weight/mid/weight0_2.txt", PReLU_weight2_3);

    // conv3 layer: Convolution 연산 수행
    clock_t conv2_3S = clock();
    _Output output2_3[12];

    for (int on = 0; on < kn2_3; on++) {
        output2_3[on].feature_map = (float*)malloc(sizeof(float) * ImgSize);
        memset(output2_3[on].feature_map, 0, sizeof(float) * ImgSize);

        for (int kn = 0; kn < kn2_2; kn++) {

            float* Padding2_3 = (float*)malloc(sizeof(float) * PW2_3 * PH2_3);
            float* TempY = (float*)malloc(sizeof(float) * ImgSize);
            memset(Padding2_3, 0, sizeof(float) * PW2_3 * PH2_3);
            memset(TempY, 0, sizeof(float) * ImgSize);

            // conv2 layer kernel 가져오기
            char onnum[10];
            char temponnum[10];
            itoa(on, temponnum, 10);
            itoa(on, onnum, 10);
            char kernelfile_dir[255] = "./parameter/kernel/mid/conv2/conv2_";
            strcat(kernelfile_dir, strcat(onnum, "/"));
            char buffer[30];
            char kernel[30] = "kernel";
            itoa(kn, buffer, 10);
            printf("%d\n", kn);
            char sign[10] = "_";
            strcat(kernelfile_dir, strcat(kernel, strcat(buffer, strcat(strcat(sign, temponnum), ".txt"))));
            printf("kernel file: %s\n", kernelfile_dir);
            _minmax cur;
            cur=getkernel_(kernelfile_dir, kernel2_3);
            if (maxc < cur.maxcc)
                maxc = cur.maxcc;
            if (minc > cur.mincc)
                minc = cur.mincc;

            // featuremap 불러오기

            padding_initialize(Padding2_3, output2_2[kn].feature_map, PW2_3, PH2_3, pad2_3);


            clock_t start;
            clock_t end;

            start = clock();
            //여기 conv 3x3
            bufferWrite(Padding2_3, kernel2_3, TempY);
            runKernel(TempY, W, H, ks2_3, pad2_3);
            //matmul_3(Padding2_3, kernel2_3, TempY, H, W, ks2_3, pad2_3);
            //for (int z = 0; z < H; z++) { // z: Height, y축
            //    for (int t = 0; t < W; t++) { // t: Width, x축
            //        _element sr[9];
            //        for (int i = 0; i < ks2_3; i++) { // y축 kernel size
            //            for (int j = 0; j < ks2_3; j++) { // x축 kernel size
            //                sr[i * ks2_3 + j].cols = (t + ((W + pad2_3 + pad2_3) * i) + j + (z * (W + pad2_3 + pad2_3)));
            //                sr[i * ks2_3 + j].w = kernel2_3[i * ks2_3 + j];
            //            }
            //        }

            //        float sumb = 0;
            //        sumb = 0;
            //        for (int j = 0; j < ks2_3; j++) {
            //            for (int k = 0; k < ks2_3; k++) {
            //                sumb = sumb + sr[j * ks2_3 + k].w * float(Padding2_3[sr[j * ks2_3 + k].cols]);
            //            }
            //        }
            //        TempY[z * W + t] = sumb;
            //    }
            //}
            /*
            int y2_3 = 0;
            int k = W * ks2_3;
            for (int z = 0; z < H; z++) { // z: Height, y축
                int h_start = z * PW2_3 * 1;
                int h_end = h_start + k;
                for (int t = 0; t < W; t++) { // t: Width, x축
                    int w_start = t * 1;
                    int w_end = w_start + ks2_3;
                    float sumb = 0;
                    int kx = 0;
                    int ky = 0;
                    for (int i = h_start; i < h_end; i += PW2_3) {
                        for (int j = w_start; j < w_end; j++) {
                            sumb = sumb + kernel2_3[ky + kx] * float(Padding2_3[i + j]);
                            //printf("kernel index: %d, padding[%d]: %f\n", kx + ky, i + j, Padding3[i + j]);
                            kx++;

                        }
                        kx = 0;
                        ky += ks2_3;
                    }
                    TempY[y2_3 + t] = sumb;
                }
                y2_3 += W;
            }
            */
            clock_t plus_S = clock();
            for (int index = 0; index < ImgSize; index++) {
                output2_3[on].feature_map[index] = output2_3[on].feature_map[index] + TempY[index];
            }
            end = clock();
            conv2_3 = (float)(end - start) / CLOCKS_PER_SEC;
            printf("convolution 소요 시간: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);


            //free(SM.SR);

            free(Padding2_3);
            free(TempY);
        }
        clock_t relu2_s = clock();

        for (int index = 0; index < ImgSize; index++) {
            if (output2_3[on].feature_map[index] + bias2_3[on] >= 0) { output2_3[on].feature_map[index] = output2_3[on].feature_map[index] + bias2_3[on]; }
            else { output2_3[on].feature_map[index] = PReLU_weight2_3[on] * (output2_3[on].feature_map[index] + bias2_3[on]); }
        }
        //SaveFeatureMap_f(W, H, output2_3[on].feature_map, on, "mid_part/conv3/");
        prelu2_3 = (float)(clock() - relu2_s) / CLOCKS_PER_SEC;
        printf("ReLU 및 bias 덧셈 소요 시간: %.3f\n", (float)(clock() - relu2_s) / CLOCKS_PER_SEC);

    }
    clock_t conv2_3e = clock();
    printf("MID PART - conv layer 3 소요 시간: %.3f\n", (float)(conv2_3e - conv2_3S) / CLOCKS_PER_SEC);


    // conv4 layer
    printf("\nCONV4 LAYER\N");
    int kn2_4 = 12;
    int ks2_4 = 3;
    float* kernel2_4 = (float*)malloc(sizeof(float) * ks2_4 * ks2_4);
    int stride2_4 = 1;
    int pad2_4 = 1;
    int PW2_4 = W + (pad2_4 * 2);
    int PH2_4 = H + (pad2_4 * 2);

    // bias 불러오기
    float bias2_4[12];
    getbias_("./parameter/bias/mid/bias0_3.txt", bias2_4);

    // PReLU weight 불러오기
    float PReLU_weight2_4[12];
    getbias_("./parameter/PReLU_weight/mid/weight0_3.txt", PReLU_weight2_4);

    // conv3 layer: Convolution 연산 수행(conv1: kernel 56개)
    clock_t conv2_4S = clock();
    _Output output2_4[12];

    for (int on = 0; on < kn2_4; on++) {
        output2_4[on].feature_map = (float*)malloc(sizeof(float) * ImgSize);
        memset(output2_4[on].feature_map, 0, sizeof(float) * ImgSize);

        for (int kn = 0; kn < kn2_3; kn++) {

            float* Padding2_4 = (float*)malloc(sizeof(float) * PW2_4 * PH2_4);
            float* TempY = (float*)malloc(sizeof(float) * ImgSize);
            memset(Padding2_4, 0, sizeof(float) * PW2_4 * PH2_4);
            memset(TempY, 0, sizeof(float) * ImgSize);

            // conv2 layer kernel 가져오기
            char onnum[10];
            char temponnum[10];
            itoa(on, temponnum, 10);
            itoa(on, onnum, 10);
            char kernelfile_dir[255] = "./parameter/kernel/mid/conv3/conv3_";
            strcat(kernelfile_dir, strcat(onnum, "/"));
            char buffer[30];
            char kernel[30] = "kernel";
            itoa(kn, buffer, 10);
            printf("%d\n", kn);
            char sign[10] = "_";
            strcat(kernelfile_dir, strcat(kernel, strcat(buffer, strcat(strcat(sign, temponnum), ".txt"))));
            _minmax cur;
            cur=getkernel_(kernelfile_dir, kernel2_4);
            if (maxc < cur.maxcc)
                maxc = cur.maxcc;
            if (minc > cur.mincc)
                minc = cur.mincc;


            // featuremap 불러오기

            padding_initialize(Padding2_4, output2_3[kn].feature_map, PW2_4, PH2_4, pad2_4);


            clock_t start;
            clock_t end;

            start = clock();
            //여기 conv 3x3
            //matmul_3(Padding2_4, kernel2_4, TempY, H, W, ks2_4, pad2_4);
            bufferWrite(Padding2_4, kernel2_4, TempY);
            runKernel(TempY, W, H, ks2_4, pad2_4);
            //for (int z = 0; z < H; z++) { // z: Height, y축
            //    for (int t = 0; t < W; t++) { // t: Width, x축
            //        _element sr[9];
            //        for (int i = 0; i < ks2_4; i++) { // y축 kernel size
            //            for (int j = 0; j < ks2_4; j++) { // x축 kernel size
            //                sr[i * ks2_4 + j].cols = (t + ((W + pad2_4 + pad2_4) * i) + j + (z * (W + pad2_4 + pad2_4)));
            //                sr[i * ks2_4 + j].w = kernel2_4[i * ks2_4 + j];
            //            }
            //        }

            //        float sumb = 0;
            //        sumb = 0;
            //        for (int j = 0; j < ks2_4; j++) {
            //            for (int k = 0; k < ks2_4; k++) {
            //                sumb = sumb + sr[j * ks2_4 + k].w * float(Padding2_4[sr[j * ks2_4 + k].cols]);
            //            }
            //        }
            //        TempY[z * W + t] = sumb;
            //    }
            //}
            /*
            int y2_4 = 0;
            int k = W * ks2_4;
            for (int z = 0; z < H; z++) { // z: Height, y축
                int h_start = z * PW2_4 * 1;
                int h_end = h_start + k;
                for (int t = 0; t < W; t++) { // t: Width, x축
                    int w_start = t * 1;
                    int w_end = w_start + ks2_4;
                    float sumb = 0;
                    int kx = 0;
                    int ky = 0;
                    for (int i = h_start; i < h_end; i += PW2_4) {
                        for (int j = w_start; j < w_end; j++) {
                            sumb = sumb + kernel2_4[ky + kx] * float(Padding2_4[i + j]);
                            //printf("kernel index: %d, padding[%d]: %f\n", kx + ky, i + j, Padding3[i + j]);
                            kx++;

                        }
                        kx = 0;
                        ky += ks2_4;
                    }
                    TempY[y2_4 + t] = sumb;
                }
                y2_4 += W;
            }
            */
            clock_t plus_S = clock();
            for (int index = 0; index < ImgSize; index++) {
                output2_4[on].feature_map[index] = output2_4[on].feature_map[index] + TempY[index];
            }
            end = clock();
            conv2_4 = (float)(end - start) / CLOCKS_PER_SEC;
            //printf("Sparse Matrix 생성 * convolve 소요 시간: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);


            free(Padding2_4);
            free(TempY);
        }
        clock_t relu2_s = clock();

        for (int index = 0; index < ImgSize; index++) {
            if (output2_4[on].feature_map[index] + bias2_4[on] >= 0) { output2_4[on].feature_map[index] = output2_4[on].feature_map[index] + bias2_4[on]; }
            else { output2_4[on].feature_map[index] = PReLU_weight2_4[on] * (output2_4[on].feature_map[index] + bias2_4[on]); }
            //printf("확인좀요3333 %f\n", output2_4[on].feature_map[index]);
        }
        //SaveFeatureMap_f(W, H, output2_4[on].feature_map, on, "mid_part/conv4/");
        prelu2_4 = (float)(clock() - relu2_s) / CLOCKS_PER_SEC;
        printf("ReLU 및 bias 덧셈 소요 시간: %.3f\n", (float)(clock() - relu2_s) / CLOCKS_PER_SEC);

    }

    clock_t conv2_4e = clock();
    printf("MID PART - conv layer 4 소요 시간: %.3f\n", (float)(conv2_4e - conv2_4S) / CLOCKS_PER_SEC);

    // conv5 layer
    printf("\nCONV5 LAYER\N");
    int kn2_5 = 12;
    int ks2_5 = 3;
    float* kernel2_5 = (float*)malloc(sizeof(float) * ks2_5 * ks2_5);
    int stride2_5 = 1;
    int pad2_5 = 1;
    int PW2_5 = W + (pad2_5 * 2);
    int PH2_5 = H + (pad2_5 * 2);

    // bias 불러오기
    float bias2_5[12];
    getbias_("./parameter/bias/mid/bias0_4.txt", bias2_5);
    for (int i = 0; i < 12; i++) {
        printf("bias2_5: %f\n", bias2_5[i]);
    }

    // PReLU weight 불러오기
    float PReLU_weight2_5[12];
    getbias_("./parameter/PReLU_weight/mid/weight0_4.txt", PReLU_weight2_5);

    // conv3 layer: Convolution 연산 수행(conv1: kernel 56개)
    clock_t conv2_5S = clock();
    _Output output2_5[12];

    for (int on = 0; on < kn2_5; on++) {
        output2_5[on].feature_map = (float*)malloc(sizeof(float) * ImgSize);
        memset(output2_5[on].feature_map, 0, sizeof(float) * ImgSize);

        for (int kn = 0; kn < kn2_4; kn++) {

            float* Padding2_5 = (float*)malloc(sizeof(float) * PW2_5 * PH2_5);
            float* TempY = (float*)malloc(sizeof(float) * ImgSize);
            memset(Padding2_5, 0, sizeof(float) * PW2_5 * PH2_5);
            memset(TempY, 0, sizeof(float) * ImgSize);

            char onnum[10];
            char temponnum[10];
            itoa(on, temponnum, 10);
            itoa(on, onnum, 10);
            char kernelfile_dir[255] = "./parameter/kernel/mid/conv4/conv4_";
            strcat(kernelfile_dir, strcat(onnum, "/"));
            char buffer[30];
            char kernel[30] = "kernel";
            itoa(kn, buffer, 10);
            char sign[10] = "_";
            strcat(kernelfile_dir, strcat(kernel, strcat(buffer, strcat(strcat(sign, temponnum), ".txt"))));
            _minmax cur;
            cur=getkernel_(kernelfile_dir, kernel2_5);
            if (maxc < cur.maxcc)
                maxc = cur.maxcc;
            if (minc > cur.mincc)
                minc = cur.mincc;

            // featuremap 불러오기

            padding_initialize(Padding2_5, output2_4[kn].feature_map, PW2_5, PH2_5, pad2_5);


            clock_t start;
            clock_t end;

            start = clock();
            // 여기 conv 3x3
            //matmul_3(Padding2_5, kernel2_5, TempY, H, W, ks2_5, pad2_5);
            bufferWrite(Padding2_5, kernel2_5, TempY);
            runKernel(TempY, W, H, ks2_5, pad2_5);
            //for (int z = 0; z < H; z++) { // z: Height, y축
            //    for (int t = 0; t < W; t++) { // t: Width, x축
            //        _element sr[9];
            //        for (int i = 0; i < ks2_5; i++) { // y축 kernel size
            //            for (int j = 0; j < ks2_5; j++) { // x축 kernel size
            //                sr[i * ks2_5 + j].cols = (t + ((W + pad2_5 + pad2_5) * i) + j + (z * (W + pad2_5 + pad2_5)));
            //                sr[i * ks2_5 + j].w = kernel2_5[i * ks2_5 + j];
            //            }
            //        }

            //        float sumb = 0;
            //        sumb = 0;
            //        for (int j = 0; j < ks2_5; j++) {
            //            for (int k = 0; k < ks2_5; k++) {
            //                sumb = sumb + sr[j * ks2_5 + k].w * float(Padding2_5[sr[j * ks2_5 + k].cols]);
            //            }
            //        }
            //        TempY[z * W + t] = sumb;
            //    }
            //}
            /*
            int y2_5 = 0;
            int k = W * ks2_5;
            for (int z = 0; z < H; z++) { // z: Height, y축
                int h_start = z * PW2_5 * 1;
                int h_end = h_start + k;
                for (int t = 0; t < W; t++) { // t: Width, x축
                    int w_start = t * 1;
                    int w_end = w_start + ks2_5;
                    float sumb = 0;
                    int kx = 0;
                    int ky = 0;
                    for (int i = h_start; i < h_end; i += PW2_5) {
                        for (int j = w_start; j < w_end; j++) {
                            sumb = sumb + kernel2_5[ky + kx] * float(Padding2_5[i + j]);
                            //printf("kernel index: %d, padding[%d]: %f\n", kx + ky, i + j, Padding3[i + j]);
                            kx++;

                        }
                        kx = 0;
                        ky += ks2_5;
                    }
                    TempY[y2_5 + t] = sumb;
                }
                y2_5 += W;
            }
            */
            clock_t plus_S = clock();
            for (int index = 0; index < ImgSize; index++) {
                output2_5[on].feature_map[index] = output2_5[on].feature_map[index] + TempY[index];
            }
            end = clock();
            conv2_5 = (float)(end - start) / CLOCKS_PER_SEC;
            printf("Sparse Matrix 생성 * convolve 소요 시간: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);

            free(Padding2_5);
            free(TempY);
        }
        clock_t relu2_s = clock();

        for (int index = 0; index < ImgSize; index++) {
            if (output2_5[on].feature_map[index] + bias2_5[on] >= 0) { output2_5[on].feature_map[index] = output2_5[on].feature_map[index] + bias2_5[on]; }
            else { output2_5[on].feature_map[index] = PReLU_weight2_5[on] * (output2_5[on].feature_map[index] + bias2_5[on]); }
            //printf("확인좀요2222 %f\n", output2_5[on].feature_map[index]);
        }
        // SaveFeatureMap_f(W, H, output2_5[on].feature_map, on, "mid_part/conv5/");
        prelu2_5 = (float)(clock() - relu2_s) / CLOCKS_PER_SEC;
        printf("ReLU 및 bias 덧셈 소요 시간: %.3f\n", (float)(clock() - relu2_s) / CLOCKS_PER_SEC);

    }

    clock_t conv2_5e = clock();
    printf("MID PART - conv layer 5 소요 시간: %.3f\n", (float)(conv2_5e - conv2_5S) / CLOCKS_PER_SEC);

    // conv6 layer
    printf("\nCONV6 LAYER\n");
    int kn2_6 = 56;
    int ks2_6 = 1;
    float* kernel2_6 = (float*)malloc(sizeof(float) * ks2_6 * ks2_6);

    // bias 불러오기
    float bias2_6[56];
    getbias_("./parameter/bias/mid/bias0_5.txt", bias2_6);

    // PReLU weight 불러오기
    float PReLU_weight2_6[56];
    getbias_("./parameter/PReLU_weight/mid/weight0_5.txt", PReLU_weight2_6);

    clock_t conv2_6S = clock();
    _Output output2_6[56];


    for (int on = 0; on < kn2_6; on++) {
        output2_6[on].feature_map = (float*)malloc(sizeof(float) * ImgSize);
        memset(output2_6[on].feature_map, 0, sizeof(float) * ImgSize);

        for (int kn = 0; kn < kn2_3; kn++) {
            float* TempY = (float*)malloc(sizeof(float) * ImgSize);
            memset(TempY, 0, sizeof(float) * ImgSize);

            char onnum[10];
            char temponnum[10];
            itoa(on, temponnum, 10);
            itoa(on, onnum, 10);
            char kernelfile_dir[255] = "./parameter/kernel/mid/conv5/conv5_";
            strcat(kernelfile_dir, strcat(onnum, "/"));
            char buffer[30];
            char kernel[30] = "kernel";
            itoa(kn, buffer, 10);
            printf("%d\n", kn);
            char sign[10] = "_";
            strcat(kernelfile_dir, strcat(kernel, strcat(buffer, strcat(strcat(sign, temponnum), ".txt"))));
            printf("kernel file: %s\n", kernelfile_dir);
            _minmax cur;
            cur=getkernel_(kernelfile_dir, kernel2_6);
            if (maxc < cur.maxcc)
                maxc = cur.maxcc;
            if (minc > cur.mincc)
                minc = cur.mincc;

            clock_t start;
            clock_t end;

            start = clock();
            // int cnt = 0;
            //여기 conv 1x1
            //matmul_1(output2_5[kn].feature_map, kernel2_6, TempY, H, W);
            bufferWrite_1(output2_5[kn].feature_map, kernel2_6, TempY);
            runKernel_1(TempY, W, H);
            //for (int z = 0; z < H; z++) { // z: Height, y축
            //    for (int t = 0; t < W; t++) { // t: Width, x축
            //        TempY[z * W + t] = kernel2_6[0] * float(output2_5[kn].feature_map[z * W + t]);
            //    }
            //}

            clock_t plus_S = clock();
            for (int index = 0; index < ImgSize; index++) {
                output2_6[on].feature_map[index] = output2_6[on].feature_map[index] + TempY[index];
                //printf("output2_6: %f\n", output2_6[on].feature_map[index]);
            }
            end = clock();
            conv2_6 = (float)(end - start) / CLOCKS_PER_SEC;
            printf("Sparse Matrix 생성 * convolve 소요 시간: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);

            free(TempY);
        }
        clock_t relu2_s = clock();

        for (int index = 0; index < ImgSize; index++) {

            if (output2_6[on].feature_map[index] + bias2_6[on] >= 0) { output2_6[on].feature_map[index] = output2_6[on].feature_map[index] + bias2_6[on]; }
            else { output2_6[on].feature_map[index] = PReLU_weight2_6[on] * (output2_6[on].feature_map[index] + bias2_6[on]); }
            //printf("확인좀요2222 %f\n", output2_6[on].feature_map[index]);
        }
        //SaveFeatureMap_f(W, H, output2_6[on].feature_map, on, "mid_part/conv6/");

        printf("ReLU 및 bias 덧셈 소요 시간: %.3f\n", (float)(clock() - relu2_s) / CLOCKS_PER_SEC);
        prelu2_6 = (float)(clock() - relu2_s) / CLOCKS_PER_SEC;
    }

    clock_t conv2_6e = clock();
    printf("MID PART - conv layer 6 소요 시간: %.3f\n", (float)(conv2_6e - conv2_6S) / CLOCKS_PER_SEC);


    // last part
    printf("\n===== LAST PART =====\n");
    int kn3 = 1;
    int ks3 = 9;
    int stride3 = 2;
    int pad3 = 4;
    int out_pad3 = 1;
    //int OH = H * scale;
    //int OW = W * scale;
    //int outsize = OH * OW;

    // padding 전용 변수
    int _pad3 = 1 * (ks3 - 1) - pad3;
    int _stride3 = stride3 - 1;
    int PW3 = 2 * W - 1 + (_pad3 * 2) + out_pad3;
    int PH3 = 2 * H - 1 + (_pad3 * 2) + out_pad3;

    // bias 불러오기
    float bias3[1];
    getbias_("./parameter/bias/last/bias0_0.txt", bias3);

    float* kernel3 = (float*)malloc(sizeof(float) * ks3 * ks3);
    _Output output3[1];
    output3[0].feature_map = (float*)malloc(sizeof(float) * W * H * scale * scale);
    memset(output3[0].feature_map, 0, sizeof(float) * W * H * scale * scale);


    for (int on = 0; on < kn3; on++) {
        for (int kn = 0; kn < 56; kn++) {
            // 1) padding 추가

            float* TempY = (float*)malloc(sizeof(float) * outsize);
            memset(TempY, 0, sizeof(float) * outsize);
            float* Padding3 = (float*)malloc(sizeof(float) * PW3 * PH3);
            memset(Padding3, 0, sizeof(float) * PW3 * PH3);

            transpose_padding_initialize(Padding3, output2_6[kn].feature_map, W, H, PW3, PH3, _pad3, stride3);

            // 2) kernel 불러오기 + 상하좌우 반전 시키기
            char buffer[10];
            char kernelfile_dir[255] = "./parameter/kernel/last/kernel";
            strcat(kernelfile_dir, itoa(kn, buffer, 10));
            strcat(kernelfile_dir, "_");
            strcat(kernelfile_dir, "0"); 
            strcat(kernelfile_dir, ".txt");
            printf("kernel file: %s\n", kernelfile_dir);
            _minmax cur;
            cur=getkernel_(kernelfile_dir, kernel3);
            if (maxc < cur.maxcc)
                maxc = cur.maxcc;
            if (minc > cur.mincc)
                minc = cur.mincc;

            /*
            // 상하좌우 반전
            float* kernelT = (float*)malloc(sizeof(float) * ks3 * ks3);
            memset(kernelT, 0, sizeof(float) * ks3 * ks3);
            for (int i = 0; i < ks3; i++) {
                for (int j = 0; j < ks3; j++) {
                    kernelT[i * ks3 + j] = kernel3[(ks3 - 1 - i) * ks3 + (ks3 - 1 - j)];
                }
            }
            */

            // 3) convoluiton
            clock_t start = clock();
            //여기 transposed conv
            int y3 = 0;
            int k = PW3 * ks3;
            //trans_matmul(Padding3, kernel3, TempY, OH, OW, PW3, k, ks3, y3);
            bufferWrite_trans(Padding3, kernel3, TempY);
            runKernel_trans(TempY, OH, OW, PW3, k, ks3, y3);
            //for (int z = 0; z < OH; z++) { // z: Height, y축
            //    int h_start = z * PW3 * 1;
            //    int h_end = h_start + k;
            //    for (int t = 0; t < OW; t++) { // t: Width, x축
            //        int w_start = t * 1;
            //        int w_end = w_start + ks3;
            //        float sumb = 0;
            //        int kx = 0;
            //        int ky = 0;
            //        for (int i = h_start; i < h_end; i += PW3) {
            //            for (int j = w_start; j < w_end; j++) {
            //                sumb = sumb + kernel3[ky + kx] * float(Padding3[i + j]);
            //                //printf("kernel index: %d, padding[%d]: %f\n", kx + ky, i + j, Padding3[i + j]);
            //                kx++;

            //            }
            //            kx = 0;
            //            ky += ks3;
            //        }
            //        TempY[y3 + t] = sumb;
            //        //printf("%d\n", y3 + t);
            //    }
            //    y3 += OW;
            //}

            for (int i = 0; i < outsize; i++) {
                output3[on].feature_map[i] = output3[on].feature_map[i] + TempY[i];
            }
            conv_l = (float)(clock() - start) / CLOCKS_PER_SEC;
            printf("convoltuion 연산 소요시간: %.3f\n", (float)(clock() - start) / CLOCKS_PER_SEC);

            free(Padding3);
            free(TempY);
        }
        clock_t bias_s = clock();
        for (int i = 0; i < W * H * scale * scale; i++) {
            output3[0].feature_map[i] = output3[0].feature_map[i] + bias3[0];
        }
        biast = (float)(clock() - bias_s) / CLOCKS_PER_SEC;
        printf("bias 덧셈 소요 시간 : %.3f\n", (float)(clock() - bias_s) / CLOCKS_PER_SEC);
    }
    //SaveFeatureMap_f(W * scale, H * scale, output3[0].feature_map, 0, "last_part/");

    int OutSize = W * H * scale * scale;

    float* R = (float*)malloc(sizeof(float) * OutSize);
    float* G = (float*)malloc(sizeof(float) * OutSize);
    float* B = (float*)malloc(sizeof(float) * OutSize);

    float* Output = (float*)malloc(sizeof(float) * OutSize * 3);

    clock_t torgb_S = clock();
    for (int i = 0; i < OutSize; i++) {
        R[i] = 298.082 * 255.0 * output3[0].feature_map[i] / 256. + 408.583 * Cr[i] / 256. - 222.921;
        G[i] = 298.082 * 255.0 * output3[0].feature_map[i] / 256. - 100.291 * Cb[i] / 256. - 208.120 * Cr[i] / 256. + 135.576;
        B[i] = 298.082 * 255.0 * output3[0].feature_map[i] / 256. + 516.412 * Cb[i] / 256. - 276.836;
    }

    int t = 0;
    for (int i = 0; i < OutSize * 3; i += 3) {
        Output[i] = B[t];
        Output[i + 1] = G[t];
        Output[i + 2] = R[t];
        t++;
    }
    printf("YCbCr --> RGB 소요 시간: %.3f\n", (float)(clock() - torgb_S) / CLOCKS_PER_SEC);

    clock_t clip_S = clock();
    for (int i = 0; i < OutSize * 3; i++) {
        if (Output[i] > 255.0) {
            Output[i] = 255.0;
        }
        else if (Output[i] < 0.0) {
            Output[i] = 0.0;
        }
    }
    printf("0 미만, 255 초과 값 clip: %.3f\n", (float)(clock() - clip_S) / CLOCKS_PER_SEC);

    clock_t realign_S = clock();
    BYTE* SOutput = (BYTE*)malloc(sizeof(BYTE) * OutSize * 3);

    realign_F(SOutput, Output, OW, OH);
    printf("상하반전 소요 시간: %.3f\n", clock() - realign_S);

    SaveBMPFile(hf, hInfo, hRGB, SOutput, OW, OH, "./test0810.bmp", 1);
    printf("TOTAL TIME: %f\n", (float)(clock() - totalS) / CLOCKS_PER_SEC);
    printf("First conv : %f\n", first_conv);
    printf("First relu : %f\n", first_prelu);
    printf("conv2_1 : %f\n", conv2_1);
    printf("relu2_1 : %f\n", prelu2_1);
    printf("conv2_2 : %f\n", conv2_2);
    printf("relu2_2 : %f\n", prelu2_2);
    printf("conv2_3 : %f\n", conv2_3);
    printf("relu2_3 : %f\n", prelu2_3);
    printf("conv2_4 : %f\n", conv2_4);
    printf("relu2_4 : %f\n", prelu2_4);
    printf("conv2_5 : %f\n", conv2_5);
    printf("relu2_5 : %f\n", prelu2_5);
    printf("conv2_6 : %f\n", conv2_6);
    printf("relu2_6 : %f\n", prelu2_6);
    printf("conv_l : %f\n", conv_l);
    printf("bias : %f\n", biast);
    printf("최소값은 : %f\n", minc);
    printf("최대값은 : %f\n", maxc);
    free(G);
    free(B);
    free(SOutput);
    CLInit();
    return 0;
}