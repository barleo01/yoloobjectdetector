#include <iostream>
#include "yolo.h"
#include <QtCore>

using namespace std;
using namespace cv;
using namespace cv::dnn;

//---- constructor
Yolo::Yolo(QObject *parent):QObject(parent){
    //qDebug() << "yolo constructor" << endl;
    setup();
    //qDebug() << "yolo construction done" << endl;
}

//---- destructor
Yolo::~Yolo(){
    //qDebug() << "Yolo is destructed" << endl;
}

//---- Setup network
void Yolo::setup(void){

    // Configure Network
    // Give the configuration and weight files for the model
    m_modelConfiguration = "data/yolov3.cfg";
    m_modelWeights = "data/yolov3.weights";

    // Load the network
    m_net = readNetFromDarknet(m_modelConfiguration, m_modelWeights);
    m_net.setPreferableBackend(DNN_BACKEND_OPENCV);
    m_net.setPreferableTarget(DNN_TARGET_CPU);
}


//---- compute network with frame
void Yolo::feedForward(Mat &frame_){
    // NOTE  : Initialization of dnn is in this->setup()
    // create dnn input variable
    blobFromImage(frame_, m_blob, 1/255.0, cvSize(m_inpWidth, m_inpHeight), Scalar(0,0,0), true, false);
    //Sets the input toclassId the network
    m_net.setInput(m_blob, "data");
    //Runs the forward pass to get output of the output layers
    m_outNames = m_net.getLayerNames();
    m_net.forward(m_outs, m_outNames);
}

//---- receive a new frame to compute
void Yolo::receiveNewFrame(Mat frame){
    //qDebug() << "yolo, receiveNewFrame, thread id  " << QThread::currentThreadId();
    m_frame = frame.clone();
    this->feedForward(m_frame); // process new frame through network
    emit sendNewParameter(m_outs); // send new detected vector<Mat> for next boxes drawing
}
