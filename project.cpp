
#include "project.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#define MAX_PATH 2048


using namespace std;
using namespace cv;

bool inside_img(int i, int j, int n, int m){
    if ((i >= 0 && i < n) && (j >= 0 && j < m))
        return true;
    return false;
}

void L06_border_tracing(){
    char fname[MAX_PATH];
    Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
    Mat_<uchar> contour = Mat(src.rows, src.cols, CV_8UC1, Scalar(255));

    Point P0, P1, Pn, Pm; // Pm is Pn-1
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            if(src(i, j) == 0){
                P0 = Point(j, i);
                i = src.rows;
                break;
            }
        }
    }
    int dy[] = {0, -1, -1, -1, 0, 1, 1, 1};
    int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    list<int> AC;
    list<int> DC;
    int dir = 7;
    int n = 0;
    Point pixel = P0;
    do{
        n++;
        int old_dir = dir;
        if(dir % 2 == 0)
            dir = (dir + 7) % 8;
        else
            dir = (dir + 6) % 8;
        Point new_pixel = Point(pixel.x + dx[dir], pixel.y + dy[dir]);
        while(src(new_pixel) != 0){
            dir = (dir + 1) % 8;
            new_pixel = Point(pixel.x + dx[dir], pixel.y + dy[dir]);
        }
        contour(pixel) = 0;
        if(n == 1) P1 = new_pixel;
        Pn = new_pixel;
        Pm = pixel;
        pixel = new_pixel;
        AC.emplace_back(dir);
        DC.emplace_back((dir - old_dir + 8) % 8);
    }while(not(Pn == P1 && Pm == P0 && n >= 2));

    printf("AC: \n");
    for(int ac : AC){
        printf("%d ", ac);
    }
    printf("\n");

    printf("AC: \n");
    for(int dc : DC){
        printf("%d ", dc);
    }
    printf("\n");



    //show the image
    imshow("img", src);
    imshow("contour", contour);

    // Wait until user press some key
    waitKey(0);

}

Mat_<uchar> hue, saturation, value;
void convert_to_HSV(Mat_<Vec3b> src){
    int height = src.rows;
    int width = src.cols;
    hue = Mat(height, width, CV_8UC1);
    saturation = Mat(height, width, CV_8UC1);
    value = Mat(height, width, CV_8UC1);

    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++) {
            Vec3b src_pixel = src(i, j);
            uchar B = src_pixel[0];
            uchar G = src_pixel[1];
            uchar R = src_pixel[2];
            float r = (float)R / 255;
            float g = (float)G / 255;
            float b = (float)B / 255;
            float M = max(r, max(g, b));
            float m = min(r, min(g, b));
            float C = M - m;
            float H, S, V = M;

            if (V != 0)
                S = C / V;
            else S = 0;

            if (C != 0){
                if (M == r) H = 60 * (g - b) / C;
                if (M == g) H = 120 + 60 * (b - r) / C;
                if (M == b) H = 240 + 60 * (r - g) / C;
            }
            else H = 0;
            if (H < 0) H = H + 360;

            float H_norm = H*255/360;
            float S_norm = S*255;
            float V_norm = V*255;

            hue(i, j) = H_norm;
            saturation(i, j) = S_norm;
            value(i, j) = V_norm;

        }
//    imshow("original img", src);
//    imshow("hue img", hue);
//    imshow("saturation img", saturation);
//    imshow("value img", value);
}

uchar get_hue_value(){
    Mat_<Vec3b> src = imread("ProjectImages/stop_sign.bmp", IMREAD_COLOR);
    convert_to_HSV(src);
    Mat_<uchar> hue_src = hue.clone();
    return hue_src(src.rows/4, src.cols/2);
}

int main() {
    Mat_<Vec3b> stop_sign = imread("ProjectImages/stop_sign_road.bmp", IMREAD_COLOR);
    convert_to_HSV(stop_sign);
    Mat_<uchar> hue_stop_sign = hue.clone();

    uchar real_hue_value = hue_stop_sign(stop_sign.rows/4, stop_sign.cols/2);
    printf("real hue value = %d\n", real_hue_value);

    uchar hue_value = get_hue_value();
    printf("virtual hue value = %d\n", hue_value);


}