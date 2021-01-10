// nadstavba pro OpenCV verzi 4.4.0
// update: 2021-01-06
#ifndef RECOG_INCLUDED
#define RECOG_INCLUDED
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace recog
{

    class alarm
    {
    public:
        int tolerance = 10;
        bool active = false;
        bool vystup(void)
        {
            if(active)
            {
                return (!(states[0] & states[1] & states[2] & states[3] & states[4] & states[5] & states[6]));
            }else
            {
                return false;
            }

        }
        void update(bool stavy[7])
        {
            if((states[0] == stavy[0]) & (states[1] == stavy[1]) & (states[3] == stavy[3]) & (states[4] == stavy[4]) & (states[5] == stavy[5]) & (states[6] == stavy[6]))
            {
                iterace++;
                if(iterace > tolerance) active = true;
            }else
            {
                iterace = 0;
                active = false;
            }
            states[0] = stavy[0];
            states[1] = stavy[1];
            states[2] = stavy[2];
            states[3] = stavy[3];
            states[4] = stavy[4];
            states[5] = stavy[5];
            states[6] = stavy[6];
        }
    private:
        bool states[7] {true};
        int iterace = 0;
    };
    Mat masked_frame(Mat src, Rect mask)
    {
        //100,100,200,200
        Mat dst = src.clone();
        rectangle(dst,Point(0,0),Point(mask.x,src.cols),Scalar(0,0,0),-1);
        rectangle(dst,Point(0,0),Point(src.cols,mask.y),Scalar(0,0,0),-1);
        rectangle(dst,Point(0,mask.height + mask.y),Point(src.cols,src.rows),Scalar(0,0,0),-1);
        rectangle(dst,Point(mask.width + mask.x, 0),Point(src.cols,src.rows),Scalar(0,0,0),-1);
        return dst;
    }
    uchar average_bgr(Mat src)
    {
        int suma = 0;
        for(int y = 0; y < src.rows; y++)
        {
            for(int x = 0; x < src.cols; x++)
            {
                Vec3b intensity = src.at<Vec3b>(y, x);
                uchar blue = intensity.val[0];
                uchar green = intensity.val[1];
                uchar red = intensity.val[2];
                suma = suma + (blue + green + red);
            }
        }
        double tmp;
        tmp = suma / (src.cols * src.rows * 3);
        uchar r = (uchar)round(tmp);
        return r;

    }
}



#endif // RECOG_INCLUDED
