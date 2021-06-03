//Florin Petrean
//Pop Tudor
//group 30432
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include <array>
#include <cmath>

#define MAX_PATH 2048

using namespace std;
using namespace cv;

bool inside_img(int i, int j, int n, int m){
    if ((i >= 0 && i < n) && (j >= 0 && j < m))
        return true;
    return false;
}

template<typename T2, typename T1>
T2 convolution(Mat src, int r, int c, vector<vector<T1>> &kernel, int n, int m){
    T2 sum = 0;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            sum += (src.at<T2>(r - n/2 + i, c - m/2 + j) * kernel[i][j]);
        }
    }
    return sum;
}

list<int> border_tracing(const Mat_<uchar>& src){
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
        while(inside_img(new_pixel.y, new_pixel.x, src.rows, src.cols) && src(new_pixel) != 0){
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
//        DC.emplace_back(dir - old_dir + 8);
    }while(!(Pn == P1 && Pm == P0 && n >= 2));

    printf("AC: \n");
    for(int ac : AC){
        printf("%d ", ac);
    }
    printf("\n");

//    printf("DC: \n");
//    for(int dc : DC){
//        printf("%d ", dc);
//    }
//    printf("\n");

    //show the image
//    imshow(image_name, contour);
    return AC;

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

            hue(i, j) = (uchar)H_norm;
            saturation(i, j) = (uchar)S_norm;
            value(i, j) = (uchar)V_norm;

        }
//    imshow("original img", src);
//    imshow("hue img", hue);
//    imshow("saturation img", saturation);
//    imshow("value img", value);
}

Mat_<uchar> median_filter(const Mat_<uchar>& src, int w = 5){
    Mat_<uchar> dst;
    dst = src.clone();

    for (int i = w / 2; i < src.rows - w / 2; i++) {
        for (int j = w / 2; j < src.cols - w / 2; j++) {
            vector<uchar> kernel;
            for (int ik = 0; ik < w; ik++) {
                for (int jk = 0; jk < w; jk++) {
                    uchar pixel = src.at<uchar>(i - w / 2 + ik, j - w / 2 + jk);
                    kernel.push_back(pixel);
                }
            }
            sort(kernel.begin(), kernel.end());
            dst(i, j) = kernel[w * w / 2];

        }
    }
    return dst;
}

Mat_<uchar> gaussian_filter(const Mat_<uchar>& src, float sigma = 0.5){
    Mat_<uchar> dst, src_filtered;
    src_filtered = Mat(src.rows, src.cols, CV_8UC1);
    //Gaussian filtering
    int w;
    w = ceil(6 * sigma);
    if(w % 2 == 0) w++;
    vector<vector<double>> convolution_kernel(w, vector<double>(w, 0.0));
    //calculate convolution kernel
    int x0 = w/2;
    int y0 = w/2;
    double kernel_sum = 0.0;
    for(int y = 0; y < w; y++) {
        for(int x = 0; x < w; x++) {
            double num = (x - x0) * (x - x0) + (y - y0) * (y - y0);
            convolution_kernel[y][x] = 1/(2*CV_PI*sigma*sigma) * exp(-(num/(2*sigma*sigma)));
            kernel_sum += convolution_kernel[y][x];
        }
    }

    for(int i = w/2; i < src.rows - w/2; i++){
        for(int j = w/2; j < src.cols - w/2; j++){
            double val = convolution<uchar, double>(src, i, j, convolution_kernel, w, w);
            src_filtered(i, j) = (uchar)(val / kernel_sum);
        }
    }
//    imshow("src gaussian filtering", src_filtered);
    return src_filtered;
}

auto structure_points = vector<Point>{
        Point(-1, 0), Point(0, -1), Point(1, 0),
        Point(0, 1), Point(-1, 1), Point(1, 1),
        Point(1, -1), Point(-1, -1), Point(-2, -1),
        Point(-2, 0), Point(-2, 1), Point(-1, 2),
        Point(0, 2), Point(1, 2), Point(2, 1),
        Point(2, 0), Point(2, -1), Point(-1, -2),
        Point(0, -2), Point(1, -2), Point(0, -3),
        Point(-3, 0), Point(0, 3), Point(3, 0)};

Mat_<uchar> dilation(const Mat_<uchar>& src){
    Mat_<uchar> dst = src.clone();
    for(int i = 0; i < dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){
            if(src(i, j) == 0){
                for(auto & structure_point : structure_points) {
                    Point p = Point(j + structure_point.x, i + structure_point.y);
                    if(inside_img(p.y, p.x, src.rows, src.cols))
                        dst(p) = 0;
                }
            }

        }
    }
    return dst;
}

Mat_<uchar> errosion(const Mat_<uchar>& src){
    Mat_<uchar> dst = src.clone();
    for(int i = 0; i < dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){
            bool can_delete = false;
            for(auto & structure_point : structure_points) {
                Point p = Point(j + structure_point.x, i + structure_point.y);
                if(inside_img(p.y, p.x, src.rows, src.cols) && src(p) == 255){
                    can_delete = true;
                }
            }
            if(can_delete)
                dst(i, j) = 255;
        }
    }
    return dst;
}

Mat_<uchar> opening(const Mat_<uchar>& src) {
    Mat_<uchar> aux = errosion(src);
    Mat_<uchar> dst = dilation(aux);
    return dst;
}


vector<Mat_<uchar>> detectObjectsBFS(const Mat_<uchar>& src){
    Mat_<Vec3b> dst = Mat(src.rows, src.cols, CV_8UC3, Scalar(255, 255, 255));
    Mat_<int> labels = Mat(src.rows, src.cols, CV_32SC1, Scalar(0));

    int label = 0;
    int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    for(int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++){
            if(src(y, x) == 0 && labels(y, x) == 0){
                label++;
                queue<Point> Q;
                Q.push(Point(x, y));
                while(!Q.empty()){
                    Point q = Q.front();
                    Q.pop();
                    for(int k = 0; k < 8; k++){
                        Point n = Point(q.x + dx[k], q.y + dy[k]);
                        if (inside_img(n.y, n.x, src.rows, src.cols) && inside_img(q.y, q.x, labels.rows, labels.cols) && src(n) == 0 && labels(n) == 0){
                            labels(n) = label;
                            Q.push(n);
                        }
                    }
                }
            }
        }
    }

    vector<Mat_<uchar>> objects;
    for (int i = 0; i < label; i++){
        objects.emplace_back(Mat(src.rows, src.cols, CV_8UC1, Scalar(255)));
    }

    for(int i = 0; i < labels.rows; i++){
        for(int j = 0; j < labels.cols; j++){
            int label_obj = labels(i, j);
            if(label_obj != 0)
                objects[label_obj - 1](i + 1, j + 1) = 0;

        }
    }

    return objects;
}



bool check_edges(const vector<int>& AC, int edges){
    long N = AC.size();
    if(N < 100)
        return false;

    vector<int> edge_max;
    vector<int> edge_index;
    vector<int> edge_mmax;
    vector<int> edge_mindex;

    for(int fr = 0; fr < edges; fr++) {
        int dir[8] = {0};
        for (long i = fr * N/edges; i < (fr+1) * N/edges; i++) {
            dir[AC[i]]++;
        }

        float mean = 0.0f, std_deviation = 0.0f;
        for(int nr: dir)
            mean += nr;
        mean = mean / edge_max.size();
        for(int nr: dir)
            std_deviation += (nr - mean) * (nr - mean);
        std_deviation = sqrt(std_deviation);
        if(std_deviation < 10)
            return false;

        printf("\ndir edge %d vector:\n", fr);
        int max = 0;
        int mmax = 0;
        int i_max = 0;
        int i_mmax = 0;
        for (int i = 0; i < 8; i++) {
            if (max < dir[i]) {
                i_max = i;
                max = dir[i];
            }



            printf("%d ", dir[i]);
        }
        int prev = ((i_max - 1) + 8) % 8;
        int next = ((i_max + 1) + 8) % 8;
        for(int i = 0; i < 8; i++){
            if(mmax < dir[i] && dir[i] < max && (i == prev || i == next)){
                i_mmax = i;
                mmax = dir[i];
            }
        }
        int diff = abs(i_max - i_mmax);
        if(i_max == 0 && i_mmax == 7)
            diff = 1;
        if(i_mmax == 0 && i_max == 7)
            diff = 1;
        if(diff != 1 && (max - mmax < 10))
            return false;


        edge_max.emplace_back(max);
        edge_index.emplace_back(i_max);
        edge_mmax.emplace_back(mmax);
        edge_mindex.emplace_back(i_mmax);

    }
    float mean = 0.0f, std_deviation = 0.0f;
    for(int max: edge_max)
        mean += max;
    mean = mean / edge_max.size();

    for(int max: edge_max)
        std_deviation += (max - mean) * (max - mean);
    std_deviation = sqrt(std_deviation);

    for(int i = 0; i < edges - 1; i++){
        if((edge_index[i+1] - edge_index[i] + 8) % 8 > 3 || (edge_index[i+1] - edge_index[i] + 8) % 8 == 0)
            if((edge_mindex[i+1] - edge_mindex[i] + 8) % 8 > 3 || (edge_mindex[i+1] - edge_mindex[i] + 8) % 8 == 0)
            return false;
        if(abs(edge_max[i] - edge_max[i+1]) > std_deviation*2)
            return false;
    }
    return true;

}

int main() {
    Mat_<Vec3b> image = imread("ProjectImages/cedeazaTrecerea.bmp", IMREAD_COLOR);
    image = gaussian_filter(image);
    convert_to_HSV(image);
    Mat_<uchar> hue_image = hue.clone();

    uchar hue_value = 0; // for red
    printf("virtual hue value = %d\n", hue_value);


    Mat_<uchar> hue_mask = Mat(image.rows, image.cols, CV_8UC1, Scalar(255));

    int threshold = 10;
    uchar lower_bound = hue_value - threshold;
    uchar upper_bound = hue_value + threshold;

    for (int i = 0; i < hue_image.rows; i++) {
        for (int j = 0; j < hue_image.cols; j++) {
            uchar val = hue_image(i, j);

            if (!(val > upper_bound  && val < lower_bound)) {
                hue_mask(i, j) = 0;

            }
        }
    }

    hue_mask = opening(hue_mask);
    hue_mask = median_filter(hue_mask, 5);
    vector<Mat_<uchar>> objects = detectObjectsBFS(hue_mask);

    for(const Mat_<uchar>& object: objects){
        if(object.rows * object.cols - countNonZero(object) > 500) {
            list<int> AC = border_tracing(object);
            vector<int> vector_AC{begin(AC), end(AC)};
            if (check_edges(vector_AC, 8)) {
//                printf("\nobject %d is a stop sign\n", i);
                imshow("stop sign", object);
            } else if (check_edges(vector_AC, 3)) {
//                printf("object %d is a cedeaza trecerea sign", i);
                imshow("cedeaza trecerea", object);
            }
        }



    }




    imshow("original image", image);
//    imshow("hue image", hue_image);
    imshow("hue mask", hue_mask);
    waitKey();

}

