// FaceRecongnition.cpp : 定义控制台应用程序的入口点。
//


#include "stdafx.h"


//Detect.cpp
//Preprocessing - Detect, Cut and Save
//@Author : TysonTangSong-liangchunjiang

#include "opencv2\opencv.hpp"
#include <iostream>
#include <iterator>
#include <stdio.h>


using namespace std;
using namespace cv;
#define CAM 2
#define PHO 1
#define K 5

/** Function Headers */
void detectAndDisplay(Mat frame);
/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

int person_judge = -1;

void readImage();
void faceReconginition(Mat image_face);
//Ptr<FaceRecognizer> model;// = createFisherFaceRecognizer();  
//opencv的FaceRecogizer目前有三个类实现了他，特征脸和fisherface方法最少训练图像为两张，而LBP可以单张图像训练  
//cv::Ptr<cv::FaceRecognizer> model = cv::createEigenFaceRecognizer();  
//cv::Ptr<cv::FaceRecognizer> model = cv::createFisherFaceRecognizer();  
//cv::Ptr<cv::FaceRecognizer> model = cv::createLBPHFaceRecognizer();//LBP的这个方法在单个人脸验证方面效果最好  

// 如果你需要一个阈值，并且使用默认参数:
cv::Ptr<cv::FaceRecognizer> model = cv::createLBPHFaceRecognizer(1,8,8,8,80);
int main()
{
	VideoCapture cap;
	cap.open(0);
	//model = createFisherFaceRecognizer();
	//readImage();
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	//-- 2. Read the video stream
	if (!cap.isOpened())
	{
		cout << "open camera failed!" << endl;
		return -1;
	}

	int key = -1;
	Mat frame;
	while (key != 27)
	{
		cap >> frame;
		if (!frame.empty())
		{
			//Mat img2 = imread("p1.jpg");
			detectAndDisplay(frame);
			//imshow("src", frame);
			key = waitKey(20);
			if (key == 's')
			{
				imwrite("person.jpg", frame);
			}

		}
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	/*
	1.image表示的是要检测的输入图像
	2.objects表示检测到的人脸目标序列
	3.scaleFactor表示每次图像尺寸减小的比例
	4. minNeighbors表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸),
	5.minSize为目标的最小尺寸
	6.minSize为目标的最大尺寸
	*/
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(100, 100));
	//face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, CASCADE_SCALE_IMAGE, Size(70, 70),Size(100,100));
	Mat img_person;
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		//cout << faces[i].x << "," << faces[i].y << endl;
		rectangle(frame, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), Scalar(0, 255, 0), 2);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		//eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));

		//for (size_t j = 0; j < eyes.size(); j++)
		//{
		//	Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
		//	int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
		//	//circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);

		//	////根据eye和face判断更置信的人脸

		//	//根据眼睛是否在脸的矩形框内的判断不可靠，人脸检测的判断率下降很多，且很多是否是检测不到眼睛的！

		//	//int eye_center_x = faces[i].x + eyes[j].x + eyes[j].width / 2;
		//	//int eye_center_y = faces[i].y + eyes[j].y + eyes[j].height / 2;

		//	////判断眼睛是否在脸的矩形框内，否则删除脸
		//	//if (eye_center_x > faces[i].x && eye_center_x < faces[i].x + faces[i].width 
		//	//	&& eye_center_y > faces[i].y && eye_center_y < faces[i].y + faces[i].height)
		//	//{

		//	//}else
		//	//{
		//	//	//删除指定的元素
		//	//	std::vector<Rect>::iterator it = faces.begin()+i;
		//	//	faces.erase(it);
		//	//}

		//}
		//if (eyes.size() > 1)
		//{
		//	//删除指定的元素
		//	/*std::vector<Rect>::iterator it = faces.begin()+i;
		//	faces.erase(it);*/
		//	//rectangle(frame, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), Scalar(0, 255, 0), 2);
		//	
		//	
		//}
		img_person = frame(faces[i]);
		//faceReconginition(img_person);
	}
	//-- Show what you got
	imshow(window_name, frame);
	char c = waitKey(2);
	if (c == 's')
	{
		/*Mat img = frame();*/
		imwrite("person.jpg",img_person);
	}
}

void readImage()
{
	//两个容器images,labels来存放图像数据和对应的标签 
	vector<Mat> images;  
	vector<int> labels;  
	// images for first person 
	Mat img = imread("person00.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);  
	labels.push_back(0);  

	img = imread("person01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);  
	labels.push_back(0);  

	// images for second person  
	img = imread("person10.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);   
	labels.push_back(1);  
	img = imread("person11.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);   
	labels.push_back(1);
	img = imread("person12.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);   
	labels.push_back(1);
	img = imread("person13.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);    
	labels.push_back(1);
	img = imread("person14.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);    
	labels.push_back(1);
	img = imread("person15.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);   
	labels.push_back(1);
	img = imread("person16.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);    
	labels.push_back(1);

	//// images for third person  
	img = imread("person20.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);    
	labels.push_back(2);  
	img = imread("person21.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);   
	labels.push_back(2);  

	//// images for fourth person  
	img = imread("person30.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);  
	labels.push_back(3);  
	img = imread("person31.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img,img,Size(240,320),CV_8UC1);
	images.push_back(img);   
	labels.push_back(3);  

	model->train(images, labels);  
}

void faceReconginition(Mat image_face)
{


	//Mat img = imread("person01.jpg", CV_LOAD_IMAGE_GRAYSCALE);  
	Mat img;
	cvtColor(image_face,img,CV_BGR2GRAY);
	resize(img,img,Size(240,320),CV_8UC1);
	int predicted = model->predict(img); 

	//连续三次判断是同一个人，则输出对应的人
	/*static int judge_count = 0;
	static int same_person = 0;
	if (person_judge == predicted)
	{
		same_person ++;
	}
	judge_count ++;
	person_judge = predicted;
	if (judge_count < 3)
	{
		return;
	}else
	{
		judge_count = 0;
	}
	if (same_person < 2)
	{
		return;
	}else{
		same_person = 0;
	}*/


	/*cout<<predicted<<endl;*/
	switch(predicted)
	{
	case 0:
		cout<<"梁春疆"<<endl;
		break;
	case 1:
		cout<<"奥巴马"<<endl;
		break;
	case 2:
		cout<<"习近平"<<endl;
		break;
	case 3:
		cout<<"孙中山"<<endl;
		break;
	default: 
		cout<<"..."<<endl;
		break;
	}
}

