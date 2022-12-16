#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
int main(int, char**)
{
    VideoCapture cap = VideoCapture();
    cap.open("http://10.7.145.236:8000/");
    if(!cap.isOpened()) return -1;
    Mat frame, edges;
    namedWindow("edges", WINDOW_AUTOSIZE);
    for(;;)
    {
        cap >> frame;
        cvtColor(frame, edges, COLOR_BGR2GRAY);
        imshow("edges", edges);
        int   key_code = waitKey(30);
        int ascii_code = key_code & 0xff; 
        if( ascii_code == 'q') break;
    }
    return 0;
}