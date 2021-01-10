#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <mutex>
#include <thread>
#include "opencv2/opencv.hpp"
#include "MojeCV.hpp"
#include "json.hpp"
#include "recog.hpp"




using namespace std;
using namespace cv;
using json = nlohmann::json;

//globalni promenne
Mutex mtx;
Mat gui = imread("bg.png");
//Mat gui(Size(940,560),CV_8UC3,Scalar(50,50,50));

Mat bgr[3];
Mat vykres;
Rect ROIs[10];
int thresholds[10];
int presents[10];
bool stats[7];


#define PARAMS_FILE_NAME "settings.json"

double getOrientation(const vector<Point> &pts, Mat &img)
{

    //Vytvoření bufferu pro použití pca analýzou
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    ///Provedení PCA analýzy
    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
    //uložení středu objektů
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                      static_cast<int>(pca_analysis.mean.at<double>(0, 1)));



    //Uložení vlastních čísel a vlastních vektorů
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }
    // nakreselní středu a orientace objektů
    circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
    return angle;
}

void gui_refresh()
{
        putText(gui,"Svar 1",Point(10,30),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
        line(gui,Point(0,40),Point(gui.cols,40),Scalar(255,255,255),1,LINE_8);
        line(gui,Point(230,0),Point(230,gui.rows),Scalar(255,255,255),1,LINE_8);

        putText(gui,"Svar 2",Point(10,70),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
        line(gui,Point(0,80),Point(gui.cols,80),Scalar(255,255,255),1,LINE_8);

        putText(gui,"Svar 3",Point(10,110),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
        line(gui,Point(0,120),Point(gui.cols,120),Scalar(255,255,255),1,LINE_8);

        putText(gui,"Svar 4",Point(10,150),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
        line(gui,Point(0,160),Point(gui.cols,160),Scalar(255,255,255),1,LINE_8);

        putText(gui,"Svar 5",Point(10,190),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
        line(gui,Point(0,200),Point(gui.cols,200),Scalar(255,255,255),1,LINE_8);

        putText(gui,"Svar 6",Point(10,230),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
        line(gui,Point(0,240),Point(gui.cols,240),Scalar(255,255,255),1,LINE_8);

        putText(gui,"Prolis",Point(10,270),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
        line(gui,Point(0,280),Point(gui.cols,280),Scalar(255,255,255),1,LINE_8);

        line(gui,Point(350,0),Point(350,gui.rows),Scalar(255,255,255),1,LINE_8);
}

void detect_0()
{
    // # roi 0 detect
        Mat m1 = recog::masked_frame(bgr[0],ROIs[0]);
        Mat dm1 = m1(ROIs[0]);
        Mat edges;
        Canny(dm1,edges,thresholds[0],255);

        int value = (int)recog::average_bgr(edges);
        mtx.lock();
        if(value > presents[0])
        {
            putText(gui,to_string(value),Point(240,30),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"OK",Point(360,30),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            rectangle(vykres,ROIs[0],Scalar(0,255,0),6);
            stats[0] = true;
        }else
        {
            putText(gui,to_string(value),Point(240,30),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"FAULT",Point(360,30),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            rectangle(vykres,ROIs[0],Scalar(0,0,255),6);
            stats[0] = false;
            //cout << '\a' << endl;
        }
        mtx.unlock();

}

void detect_1()
{
    // # roi 1 detect
        Mat m2 = recog::masked_frame(bgr[0],ROIs[1]);
        Mat dm2 = m2(ROIs[1]);
        Mat edges2;
        Canny(dm2,edges2,thresholds[1],255);
        //imshow("e",edges2);

        int value = (int)recog::average_bgr(edges2);
        mtx.lock();
        if(value > presents[1])
        {
            putText(gui,to_string(value),Point(240,70),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"OK",Point(360,70),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            rectangle(vykres,ROIs[1],Scalar(0,255,0),6);
            stats[1] = true;
        }else
        {
            putText(gui,to_string(value),Point(240,70),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"FAULT",Point(360,70),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            rectangle(vykres,ROIs[1],Scalar(0,0,255),6);
            stats[1] = false;
        }
        mtx.unlock();
        // # 1 END
}

void detect_2()
{
    // # roi 2 detect
        Mat m3 = recog::masked_frame(bgr[0],ROIs[2]);
        Mat dm3 = m3(ROIs[2]);
        Mat edges3;
        Canny(dm3,edges3,thresholds[2],255);
        //imshow("e",edges3);

        int value = (int)recog::average_bgr(edges3);
        mtx.lock();
        if(value > presents[2])
        {
            putText(gui,to_string(value),Point(240,110),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"OK",Point(360,110),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            rectangle(vykres,ROIs[2],Scalar(0,255,0),6);
            stats[2] = true;
        }else
        {
            putText(gui,to_string(value),Point(240,110),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"FAULT",Point(360,110),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            rectangle(vykres,ROIs[2],Scalar(0,0,255),6);
            stats[2] = false;
        }
        mtx.unlock();
        // # 2 END
}

void detect_3()
{
    // # roi 3 detect
        Mat m4 = recog::masked_frame(bgr[0],ROIs[3]);
        Mat dm4 = m4(ROIs[3]);
        Mat edges4;
        Canny(dm4,edges4,thresholds[3],100);
        //imshow("e",edges4);
        //imshow("f",dm4);

        int value = (int)recog::average_bgr(edges4);
        mtx.lock();
        if(value > presents[3])
        {
            putText(gui,to_string(value),Point(240,150),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"OK",Point(360,150),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
            rectangle(vykres,ROIs[3],Scalar(0,255,0),6);
            stats[3] = true;
        }else
        {
            putText(gui,to_string(value),Point(240,150),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"FAULT",Point(360,150),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            rectangle(vykres,ROIs[3],Scalar(0,0,255),6);
            stats[3] = false;
        }
        mtx.unlock();
        // # 3 END
}

void detect_4()
{
    // # roi 4 detect
        Mat m5 = recog::masked_frame(bgr[0],ROIs[4]);
        Mat dm5 = m5(ROIs[4]);
        Mat edges5;
        Canny(dm5,edges5,thresholds[4],100);
        //imshow("e",edges5);
        //imshow("f",dm5);

        int value = (int)recog::average_bgr(edges5);
        mtx.lock();
        if (value > presents[4])
        {
            putText(gui,to_string(value),Point(240,190),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"OK",Point(360,190),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
            rectangle(vykres,ROIs[4],Scalar(0,255,0),6);
            stats[4] = true;
        }else
        {
            putText(gui,to_string(value),Point(240,190),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"FAULT",Point(360,190),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            rectangle(vykres,ROIs[4],Scalar(0,0,255),6);
            stats[4] = false;
        }
        mtx.unlock();
        // # 4 END
}

void detect_5()
{
    // # roi 5 detect
        Mat m6 = recog::masked_frame(bgr[0],ROIs[5]);
        Mat dm6 = m6(ROIs[5]);
        Mat edges6;
        Canny(dm6,edges6,thresholds[5],100);
        //imshow("e",edges6);
        //imshow("f",dm6);

        int value = (int)recog::average_bgr(edges6);
        mtx.lock();
        if(value > presents[5])
        {
            putText(gui,to_string(value),Point(240,230),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"OK",Point(360,230),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
            rectangle(vykres,ROIs[5],Scalar(0,255,0),6);
            stats[5] = true;
        }else
        {
            putText(gui,to_string(value),Point(240,230),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"FAULT",Point(360,230),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            rectangle(vykres,ROIs[5],Scalar(0,0,255),6);
            stats[5] = false;
        }
        mtx.unlock();
        // # 5 END
}

void dira_detect()
{


        Mat m6;

        threshold(bgr[0],m6,thresholds[6],255,THRESH_BINARY_INV);
        Mat edges6 = recog::masked_frame(m6,ROIs[6]);


        vector<vector<Point> > contours;
        findContours(edges6, contours, RETR_LIST, CHAIN_APPROX_NONE);
        for (size_t i = 0; i < contours.size(); i++)
        {
            // Calculate the area of each contour
            double area = contourArea(contours[i]);
            // ignorování přílilš malých, nebo příliš velkých kontur
            if (area < 1e3 || 1e4 < area)
            {
                //putText(gui,"FAULT",Point(360,270),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1);
                stats[6] = false;
                continue;  // 1e2 || 1e5
            }
            int a = (int)area;
            putText(gui,to_string(a),Point(240,270),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255),1);
            //putText(gui,"OK",Point(360,270),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
            stats[6] = true;
            // Draw each contour only for visualisation purposes
            drawContours(vykres, contours, static_cast<int>(i), Scalar(0, 255, 0), -1);
            // Find the orientation of each shape
            getOrientation(contours[i], vykres);
        }
}

void vysledek()
{
    if(stats[6])
    {
        putText(gui,"OK",Point(360,270),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
    }else
    {
        putText(gui,"FAULT",Point(360,270),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1);
    }

    if(stats[5])
    {
        putText(gui,"OK",Point(360,230),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
    }else
    {
        putText(gui,"FAULT",Point(360,230),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1);
    }

    if(stats[4])
    {
        putText(gui,"OK",Point(360,190),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
    }else
    {
        putText(gui,"FAULT",Point(360,190),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1);
    }

    if(stats[3])
    {
        putText(gui,"OK",Point(360,150),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
    }else
    {
        putText(gui,"FAULT",Point(360,150),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1);
    }

    if(stats[2])
    {
        putText(gui,"OK",Point(360,110),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
    }else
    {
        putText(gui,"FAULT",Point(360,110),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1);
    }

    if(stats[1])
    {
        putText(gui,"OK",Point(360,70),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
    }else
    {
        putText(gui,"FAULT",Point(360,70),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1);
    }

    if(stats[0])
    {
        putText(gui,"OK",Point(360,30),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1);
    }else
    {
        putText(gui,"FAULT",Point(360,30),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1);
    }
}
int main(int, char**)
{
    recog::alarm a;
    gui_refresh();
    json jsonConf;
    // Open file
	std::ifstream in(PARAMS_FILE_NAME);
	if (!in.is_open()) {
		std::cout << "ERROR: Params file not openned " << PARAMS_FILE_NAME << std::endl;
		return false;
	}//if...

	// Read file
	try {
		jsonConf = json::parse(in);
	}
	catch (json::parse_error& e) {
		std::cout << "ERROR: JSON " << e.what() << std::endl;
		return false;
	}

	try
	{
        json channelJson = jsonConf.at("ROI_0");
        ROIs[0].x = channelJson.at("POS_X").get<int>();
        ROIs[0].y = channelJson.at("POS_Y").get<int>();
        ROIs[0].width = channelJson.at("WIDTH").get<int>();
        ROIs[0].height = channelJson.at("HEIGHT").get<int>();
        presents[0] = channelJson.at("IS_PRESENT").get<int>();
        thresholds[0] = channelJson.at("THRESHOLD").get<int>();

        channelJson = jsonConf.at("ROI_1");
        ROIs[1].x = channelJson.at("POS_X").get<int>();
        ROIs[1].y = channelJson.at("POS_Y").get<int>();
        ROIs[1].width = channelJson.at("WIDTH").get<int>();
        ROIs[1].height = channelJson.at("HEIGHT").get<int>();
        presents[1] = channelJson.at("IS_PRESENT").get<int>();
        thresholds[1] = channelJson.at("THRESHOLD").get<int>();

        channelJson = jsonConf.at("ROI_2");
        ROIs[2].x = channelJson.at("POS_X").get<int>();
        ROIs[2].y = channelJson.at("POS_Y").get<int>();
        ROIs[2].width = channelJson.at("WIDTH").get<int>();
        ROIs[2].height = channelJson.at("HEIGHT").get<int>();
        presents[2] = channelJson.at("IS_PRESENT").get<int>();
        thresholds[2] = channelJson.at("THRESHOLD").get<int>();

        channelJson = jsonConf.at("ROI_3");
        ROIs[3].x = channelJson.at("POS_X").get<int>();
        ROIs[3].y = channelJson.at("POS_Y").get<int>();
        ROIs[3].width = channelJson.at("WIDTH").get<int>();
        ROIs[3].height = channelJson.at("HEIGHT").get<int>();
        presents[3] = channelJson.at("IS_PRESENT").get<int>();
        thresholds[3] = channelJson.at("THRESHOLD").get<int>();

        channelJson = jsonConf.at("ROI_4");
        ROIs[4].x = channelJson.at("POS_X").get<int>();
        ROIs[4].y = channelJson.at("POS_Y").get<int>();
        ROIs[4].width = channelJson.at("WIDTH").get<int>();
        ROIs[4].height = channelJson.at("HEIGHT").get<int>();
        presents[4] = channelJson.at("IS_PRESENT").get<int>();
        thresholds[4] = channelJson.at("THRESHOLD").get<int>();

        channelJson = jsonConf.at("ROI_5");
        ROIs[5].x = channelJson.at("POS_X").get<int>();
        ROIs[5].y = channelJson.at("POS_Y").get<int>();
        ROIs[5].width = channelJson.at("WIDTH").get<int>();
        ROIs[5].height = channelJson.at("HEIGHT").get<int>();
        presents[5] = channelJson.at("IS_PRESENT").get<int>();
        thresholds[5] = channelJson.at("THRESHOLD").get<int>();

        // parametry diry
        channelJson = jsonConf.at("DIRA");
        ROIs[6].x = channelJson.at("POS_X").get<int>();
        ROIs[6].y = channelJson.at("POS_Y").get<int>();
        ROIs[6].width = channelJson.at("WIDTH").get<int>();
        ROIs[6].height = channelJson.at("HEIGHT").get<int>();
        presents[6] = channelJson.at("IS_PRESENT").get<int>();
        thresholds[6] = channelJson.at("THRESHOLD").get<int>();


	}
	catch (...) {
		std::cout << "ERROR: JSON " << std::endl;
		return false;
	}//try...







    char key;
    // otevření streamu ...... int 0 => lokalni webová kamera
    VideoCapture cap;
    cap.open("stabilized.avi");
    if(!cap.isOpened())
        return -1;


    int frame_count = cap.get(CAP_PROP_FRAME_COUNT);
    Size resolution;
    resolution.width = cap.get(CAP_PROP_FRAME_WIDTH);
    resolution.height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    VideoWriter video("zaznam.avi",VideoWriter::fourcc('M','J','P','G'),fps,Size(940,560));

    int iterace = 0;
    for(;;)
    {
        Mat frame;
        cap >> frame; // obnova framebufferu
        if(frame.empty()) break;


        split(frame,bgr);
        vykres = bgr[0].clone();
        cvtColor(vykres,vykres,COLOR_GRAY2BGR);

        dira_detect();
        thread t1(detect_0);
        thread t2(detect_1);
        thread t3(detect_2);
        thread t4(detect_3);
        thread t5(detect_4);
        thread t6(detect_5);
        t1.join();
        t2.join();
        t3.join();
        t4.join();
        t5.join();
        t6.join();
        vysledek();

        a.update(stats);
        if(a.vystup())
        {
            putText(gui,"ALARM",Point(30,310),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1);
            //system("aplay alarm.wav");
        }












        /////////////////// VYKRES /////////////////////////////

        resize(vykres,vykres,Size(497,278));
        vykres.copyTo(gui(Rect(940-497,560-278,497,278)));

        ////////////////////////////////////////////////////////



        if(iterace == frame_count -1)
        {
            cap.set(CAP_PROP_POS_FRAMES,1);
            iterace = 0;
        }
        imshow("TERMINAL",gui);
        //video.write(gui);
        key = (char)waitKeyEx(10);
        if( key == 27 || key == 'q' || key == 'Q' )
        {
            break;
        }
        iterace++;
        gui = imread("bg.png");
        gui_refresh();


    }
    return 0;
}
