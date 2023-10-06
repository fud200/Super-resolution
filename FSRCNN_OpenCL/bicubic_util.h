#pragma warning(disable:4996)
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <math.h>
#include <io.h>
#include <time.h>


// """ Sparse Matrix ver 1 """

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
    printf("희소행렬 생성 완료: %.3f\n", (float)(end - start) / CLOCKS_PER_SEC);

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
    printf("%.3f\n", (float)(end - start) / CLOCKS_PER_SEC);
}
