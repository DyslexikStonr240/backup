#ifndef DATASET_READER
#define DATASET_READER

#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>


//using namespace std;
//using namespace Eigen;

class Dataset_reader
{
    public:

        Dataset_reader(std::string path_to_extracted_mnist_files, Eigen::MatrixXd& train_images, Eigen::MatrixXd& test_images, Eigen::VectorXd& train_labels, Eigen::VectorXd& test_labels); // reads in all training / test images + labels

        int                     get_nr_train_images_read();          // should be 60.000
        int                     get_nr_test_images_read();
        int                     get_size_of_image();

    private:

        void                    read_mnist_images(std::string full_path, Eigen::MatrixXd& image, int& number_of_images, int& image_size); // for reading in images
        void                    read_mnist_labels(std::string full_path, Eigen::VectorXd& label, int& number_of_labels);                  // for reading in ground truth image labels
        std::string             path_to_extracted_mnist_files; // where you have extracted the files


        int                     nr_train_images_read;          // should be 60.000
        int                     nr_test_images_read;           // should be 10.000
        int                     size_of_image;

};

#endif
