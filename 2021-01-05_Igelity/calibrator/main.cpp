#include <iostream>
#include "opencv2/opencv.hpp"
#include "../MojeCV.hpp"

using namespace std;
using namespace cv;


int main(int, char**)
{
    char key;
    // otevření streamu ...... int 0 => lokalni webová kamera
    VideoCapture cap("../stabilized.avi");
    if(!cap.isOpened())
        return -1;


    for(;;)
    {
        Mat frame;
        cap >> frame; // obnova framebufferu

        Rect r = selectROI(frame);
        cout << r << endl;
        break;


        imshow("frame", frame);



        key = (char)waitKeyEx(20);
        if( key == 27 || key == 'q' || key == 'Q' )
        {
            break;
        }


    }
    return 0;
}
