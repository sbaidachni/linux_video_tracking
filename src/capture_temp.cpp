// Just my initial code to test OpenCV availability and make file.


#include <stdio.h>
#include <opencv2/opencv.hpp>
 
using namespace cv;
 
int main(int argc, char** argv )
{
    // Open the default camera (usually the first camera)
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) 
    {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    // This will contain the image from the webcam
    cv::Mat frame;

    camera >> frame;

    // Check if the frame is empty
    if (frame.empty()) 
    {
        std::cerr << "ERROR: Could not grab a frame" << std::endl;
    }
    else
    {
        std::cout<<"Image has been read";
        // Save the captured frame to a file
        std::string filename = "captured_image.jpg";
        if (!cv::imwrite(filename, frame)) 
        {
            std::cerr << "Error: Could not save image to file" << std::endl;
            return -1;
        }
    }



    camera.release();
 
    return 0;
}