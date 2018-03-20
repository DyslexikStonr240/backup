#include "opencv2/core.hpp"

#include <iostream>
#include <random>                 // for random numbers & 1D normal distribution
#define _USE_MATH_DEFINES
#include <math.h>                 // for M_PI

#include "mnist_dataset_reader.h"
#include"Classification.h"


using namespace cv;
using namespace std;

int main()
{
    string path_to_extracted_mnist_files = "./MNIST_FILES";
    mnist_dataset_reader* my_reader = new mnist_dataset_reader( path_to_extracted_mnist_files );
    Classification nearestNeighbour(my_reader);
    //delete my_reader;
    /*
    Mat* img = my_reader.get_mnist_image_as_cvmat( my_reader.get_train_images(), 1 );
    Mat img_resized;
    resize( *img, img_resized, cv::Size(0,0), 4,4);
    imshow("A sample digit", img_resized);
    */

    //Mat* samples_as_image = my_reader.get_board_of_sample_images( my_reader.get_train_images(), my_reader.get_train_labels(), 60000 );
    //imshow("Some sample MNIST training images", *samples_as_image );
    //delete samples_as_image;

    //waitKey(0);


    return 0;

}
