#ifndef YOLO_H
#define YOLO_H

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <iterator>

#include <QObject>
#include <QImage>
#include <QThread>

using namespace std;
using namespace cv;
using namespace cv::dnn;


class Yolo: public QObject
{
    Q_OBJECT

private:
    const int m_inpWidth {416};        // Width of network's input image
    const int m_inpHeight {416};       // Height of network's input image

    vector<string> m_classes;
    Net m_net;

    Mat m_frame;

    Mat m_blob; // variable for the network input

    vector<String> m_outNames;

    vector<Rect> m_outBox;

    String m_modelConfiguration;
    String m_modelWeights;

    vector<Mat> m_outs;


public:
    //Yolo();
    explicit Yolo(QObject *parent = 0);
    ~Yolo();

    void setup(void);
    void feedForward(Mat &frame);

private slots:
    void receiveNewFrame(Mat frame);

signals:
    void sendNewParameter(vector<Mat> m_outBoxes);

};

#endif // YOLO_H
