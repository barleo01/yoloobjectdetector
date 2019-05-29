#ifndef OPENCVWORKER_H
#define OPENCVWORKER_H

#include <QObject>
#include <QImage>
#include <QtCore>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "yolo.h"

class OpenCvWorker : public QObject
{
    Q_OBJECT

private:
    // Initialize the for YOLO
    const float m_confThreshold {0.5}; // Confidence threshold
    const float m_nmsThreshold {0.4};  // Non-maximum suppression threshold
    const int m_inpWidth {416};        // Width of network's input image
    const int m_inpHeight {416};       // Height of network's input image

    vector<Mat> m_outs;

    string m_classesFile;
    vector <string> m_classes;

    cv::Mat m_currentFrame;

    cv::VideoCapture *m_cap;

    Yolo *m_yolo;

    bool m_status;
    bool m_toggleStream;

    void checkIfDeviceAlreadyOpened(const int device);

public:
    explicit OpenCvWorker(QObject *parent = 0);//, Yolo *yolo_=new Yolo());

    // process image with yolo outputs
    void postProcess(Mat& frame, const vector<Mat>& outs);
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

    ~OpenCvWorker();

signals:
    void sendFrame(QImage frameProcessed);
    void sendNewFrameYolo(Mat frame);

private slots:
    void activateYOLO();

public slots:
    void captureFrame();
    void receiveSetup(const int device);
    void receiveToggleStream();


    void receiveNewParameters(vector<Mat> outs);

};

#endif // OPENCVWORKER_H
