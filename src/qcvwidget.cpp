#include "qcvwidget.h"
#include "ui_qcvwidget.h"
#include "opencvworker.h"

#include <QTimer>

Q_DECLARE_METATYPE(vector<Mat>) //define vector <Mat> to be accepted by Qt

//---- Constructor
QCvWidget::QCvWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::QCvWidget)
{
    ui->setupUi(this);
    ui->labelView->setScaledContents(true);
    qRegisterMetaType< Mat >("Mat"); //Register Mat for data exchange OpenCvWorker ->  Yolo
    qRegisterMetaType<vector<Mat>>("Vect_Mat"); //Register vector<Mat> for data exchange Yolo -> OpenCvWorker
    setup();
    //qDebug() << "QCvWidget construction done" << endl;
}

//---- Destructor
QCvWidget::~QCvWidget()
{
    //qDebug() << "Destroy QCvWidget" << endl;

    //quit thread 1
    m_thread1->quit();
    //quit thread 2
    m_thread2->quit();

    // delete thread 1
    while(!m_thread1->isFinished());
    delete m_thread1;
    // delete thread 2
    m_thread2->quit();
    while(!m_thread2->isFinished());
    delete m_thread2;
    delete ui;
    //qDebug() << "QCvWidget is destructed" << endl;
}

//---- Setup
void QCvWidget::setup()
{
    m_thread1 = new QThread(); //thread used for opencv worker
    m_thread2 = new QThread(); //thread used for yolo worker

    // Worker for yolo
    Yolo *yolo = new Yolo();
    // Worker for opencv
    OpenCvWorker *worker = new OpenCvWorker();
    // Timer Setting for accessing frames
    QTimer *workerTrigger = new QTimer();
    workerTrigger->setInterval(1);

    //Timer
    connect(m_thread1, SIGNAL(finished()), workerTrigger, SLOT(deleteLater()));
    connect(workerTrigger, SIGNAL(timeout()), worker, SLOT(captureFrame()));

    //Worker
    connect(m_thread1, SIGNAL(finished()), worker, SLOT(deleteLater()));
    connect(this, SIGNAL(sendSetup(int)), worker, SLOT(receiveSetup(int)));
    connect(this, SIGNAL(sendToggleStream()), worker, SLOT(receiveToggleStream()));
    connect(ui->pushButtonPlay, SIGNAL(clicked(bool)), this, SLOT(receiveToggleStream()));
    connect(worker, SIGNAL(sendFrame(QImage)), this, SLOT(receiveFrame(QImage)));
    connect(ui->activateYOLO, SIGNAL(clicked(bool)), worker, SLOT(activateYOLO()));

    //YOLO
    connect(m_thread2, SIGNAL(finished()), yolo, SLOT(deleteLater()));

    //YOLO to worker send update
    connect(yolo, SIGNAL(sendNewParameter(vector<Mat>)),
            worker, SLOT(receiveNewParameters(vector<Mat>)));
    connect(worker, SIGNAL(sendNewFrameYolo(Mat)), yolo, SLOT(receiveNewFrame(Mat)));

    workerTrigger->start();

    worker->moveToThread(m_thread1);
    workerTrigger->moveToThread(m_thread1);
    yolo->moveToThread(m_thread2);

    m_thread1->start();
    m_thread2->start();

    this->receiveToggleStream(); // play
    emit sendSetup(0);
}

//---- Receive a new frame to be display
void QCvWidget::receiveFrame(QImage frame)
{
    ui->labelView->setPixmap(QPixmap::fromImage(frame));
}

//---- Start/Stop Video Stream
void QCvWidget::receiveToggleStream()
{
    if(!ui->pushButtonPlay->text().compare("||")){
        ui->pushButtonPlay->setText(">");
    }
    else {
        ui->pushButtonPlay->setText("||");
    }

    emit sendToggleStream();
}
