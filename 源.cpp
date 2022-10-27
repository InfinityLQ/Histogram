#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int bins = 256;
Mat src;

//const char* winTitle = "input image";
//void showHistogram();
//void drawHistogram(Mat& image);

void drawHistogram(Mat& image) {
    // �����������
    const int channels[1] = { 0 };
    const int bins[1] = { 256 };
    float hranges[2] = { 0,255 };
    const float* ranges[1] = { hranges };
    int dims = image.channels();
    if (dims == 3) {
        vector<Mat> bgr_plane;
        split(src, bgr_plane);
        Mat b_hist;
        Mat g_hist;
        Mat r_hist;
        // ����Blue, Green, Redͨ����ֱ��ͼ
        calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
        calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
        calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
        // ��ʾֱ��ͼ
        int hist_w = 512;
        int hist_h = 400;
        int bin_w = cvRound((double)hist_w / bins[0]);
        Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
        // ��һ��ֱ��ͼ����
        normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        // ����ֱ��ͼ����
        for (int i = 1; i < bins[0]; i++) {
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);

        }
        // ��ʾֱ��ͼ
        namedWindow("ֱ��ͼ", WINDOW_AUTOSIZE);
        imshow("ֱ��ͼ", histImage);
    }
    else {
        Mat hist;
        // ����Blue, Green, Redͨ����ֱ��ͼ
        calcHist(&image, 1, 0, Mat(), hist, 1, bins, ranges);
        // ��ʾֱ��ͼ
        int hist_w = 512;
        int hist_h = 400;
        int bin_w = cvRound((double)hist_w / bins[0]);
        Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
        // ��һ��ֱ��ͼ����
        normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        // ����ֱ��ͼ����
        for (int i = 1; i < bins[0]; i++) {
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
                Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
        }
        // ��ʾֱ��ͼ
        namedWindow("ֱ��ͼ", WINDOW_AUTOSIZE);
        imshow("ֱ��ͼ", histImage);
    }
}

void showHistogram() {
    // ��ͨ������
    vector<Mat> bgr_plane;
    split(src, bgr_plane);
    // �����������
    const int channels[1] = { 0 };
    const int bins[1] = { 256 };
    float hranges[2] = { 0,255 };
    const float* ranges[1] = { hranges };
    Mat b_hist;
    Mat g_hist;
    Mat r_hist;
    // ����Blue, Green, Redͨ����ֱ��ͼ
    calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
    calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
    calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
    // ��ʾֱ��ͼ
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / bins[0]);
    Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
    // ��һ��ֱ��ͼ����
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    // ����ֱ��ͼ����
    for (int i = 1; i < bins[0]; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);

        // ��ʾֱ��ͼ
        namedWindow("ֱ��ͼ", WINDOW_AUTOSIZE);
        imshow("ֱ��ͼ", histImage);
    }
}

    int main()
    {
        Mat srcImage, dstImage;
        srcImage = imread("C:/Users/��Ǭ/Pictures/Saved Pictures/pro.jpg");

        //�ж�ͼ���Ƿ���سɹ�
        if (!srcImage.data)
        {
            cout << "ͼ�����ʧ��!" << endl;
            return -1;
        }
        else
            cout << "ͼ����سɹ�!" << endl << endl;

        cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);       //��ԭͼ��ת��Ϊ�Ҷ�ͼ

        equalizeHist(srcImage, dstImage);           //ֱ��ͼ���⻯

        //��������
        String windowNameSrc = "�ڧ���էߧ�� �ڧ٧�ҧ�ѧا֧ߧڧ�ԭͼ";
        String windowNameHist = "���⻯��ͼ��";
        namedWindow(windowNameSrc, WINDOW_AUTOSIZE);
        namedWindow(windowNameHist, WINDOW_AUTOSIZE);

        //��ʾͼ��
        imshow(windowNameSrc, srcImage);
        imshow(windowNameHist, dstImage);

        //waitKey(0);


        Mat src = imread("C:/Users/��Ǭ/Pictures/Saved Pictures/pro.jpg", IMREAD_GRAYSCALE);
        //���봦��Դͼ��ɾ�������IMREAD_GRAYSCALE

       // namedWindow(winTitle, WINDOW_AUTOSIZE);
     //namedWindow(win, WINDOW_AUTOSIZE);

        drawHistogram(src);

        //drawHistogram(dstImage);

        waitKey(0);


        return 0;
    }