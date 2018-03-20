#ifndef DATASET_READER
#define DATASET_READER

#include <string>
#include <fstream>
#include <iostream>

#include "opencv2/core.hpp"    // for cv::Mat
#include "opencv2/highgui.hpp" // for CV_RGB
#include "opencv2/imgproc.hpp" // for cv::putText
#include "Tensor.h"

using namespace std;
using namespace cv;

template <class T, class U>
class DatasetReader
{
    public:


                                DatasetReader(); // reads in all training / test images + labels

                                template <class S, class V>
                                DatasetReader<T,U>& operator=(DatasetReader<S,V>& rhs);

        Tensor<T>&              get_train_images();//2d
        Tensor<T>&              get_test_images();//2d
        Tensor<U>&              get_train_labels();//1d
        Tensor<U>&              get_test_labels();//1d

        int&                    get_nr_train_images_read();          // should be 60.000
        int&                    get_nr_test_images_read();
        int&                    get_size_of_image();

        void                    centreImage();



    private:

        void                    read_mnist_images(Tensor<T>& dst, std::string dst_name, string full_path, int& number_of_images, int& image_size); // for reading in images
        void                    read_mnist_labels(Tensor<U>& dst, std::string dst_name, string full_path, int& number_of_labels);                  // for reading in ground truth image labels
        string                  path_to_extracted_mnist_files; // where you have extracted the files

        Tensor<T>               train_images;                  // all the 60.000 training images of size 28x28
        Tensor<T>               test_images;                   // all the 10.000 test     images of size 28x28
        Tensor<U>               train_labels;                  // all the 60.000 ground truth labels for the training images
        Tensor<U>               test_labels;                   // all the 10.000 ground truth labels for the test     images

        int                     nr_train_images_read;          // should be 60.000
        int                     nr_test_images_read;           // should be 10.000
        int                     size_of_image;

};

template<class T, class U>
DatasetReader<T,U>::DatasetReader(){
    // 1. store path to folder where dataset was extracted by user
    this->path_to_extracted_mnist_files =  "./MNIST_FILES";


    // 2. read in training data
    int nr_labels_read = 0;

    // 2.1 read in training images
 //    cout << endl;
    read_mnist_images(train_images, "train_images", path_to_extracted_mnist_files + "/" + "train-images.idx3-ubyte", nr_train_images_read, size_of_image);
 //    cout << "I have read " << nr_train_images_read << " training images of size " << size_of_image << "!" << endl;


    // 2.2 read in training ground truth labels
 //    cout << endl;
    read_mnist_labels(train_labels, "train_labels", path_to_extracted_mnist_files + "/" + "train-labels.idx1-ubyte", nr_labels_read);
 //    cout << "I have read " << nr_labels_read << " labels for the training images!" << endl;


    // 3. read in test data

    // 3.1 read in test images
 //    cout << endl;
    read_mnist_images(test_images, "test_images", path_to_extracted_mnist_files + "/" + "t10k-images.idx3-ubyte", nr_test_images_read, size_of_image);
 //    cout << "I have read " << nr_test_images_read << " test images of size " << size_of_image << "!" << endl;

    // 3.2 read in testing ground truth labels
 //    cout << endl;
    read_mnist_labels(test_labels, "test_labels", path_to_extracted_mnist_files + "/" + "t10k-labels.idx1-ubyte", nr_labels_read);
 //    cout << "I have read " << nr_labels_read << " labels for the testing images!" << endl;


}

template <class T, class U>
template <class S, class V>
DatasetReader<T,U>& DatasetReader<T,U>::operator=(DatasetReader<S,V>& rhs){


    train_images = rhs.get_train_images();
    test_images  = rhs.get_test_images();
    train_labels = rhs.get_train_labels();
    test_labels  = rhs.get_test_labels();

    nr_train_images_read = rhs.get_nr_train_images_read();
    nr_test_images_read  = rhs.get_nr_test_images_read();
    size_of_image        = rhs.get_size_of_image();



    //std::cout << "DatasetReader<T,U> Equal created" << std::endl;
    //std::cout << "<T,U> Equal " << (long int)_matrix << " Copied from" <<  (long int)rhs.getTensor() << std::endl << std::endl;
    return *this;
}



template<class T, class U>
void                            DatasetReader<T,U>::read_mnist_images(Tensor<T>& dst, std::string dst_name, string full_path, int& number_of_images, int& image_size){

    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

 //    cout << "reading MNIST file " << full_path << " ... " << endl;

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

 //        cout << "nr of rows x nr of cols = " << n_rows << " x " << n_cols << endl;

        image_size = n_rows * n_cols;


        //const signed char* tmpArray = new signed char[number_of_images * (image_size)];
        Tensor<char>tmpArray("tmpArray_" + dst_name, number_of_images, image_size);
        Tensor<T> _dataset("dataset_" + dst_name, number_of_images, (image_size + 1));// +1 for bias'


        int SIZE = (image_size + 1)*number_of_images;
        file.read((char*)tmpArray.getTensor(), number_of_images * image_size);
        //tmpArray.centre();


        int count = 0;
        //int modCount = image_size;

        for(int i = 0; i < SIZE; i++){
            //_dataset.getTensor()[i] = 1;
            if( (i % (image_size)) == 0 && i != 0 ){
                _dataset.getTensor()[i] = 0; count++;
            }else{
                _dataset.getTensor()[i] = static_cast<T>(tmpArray.getTensor()[i - count]); }
        }

        dst = _dataset;
        dst._name = dst_name;

        //delete[] tmpArray;
        //tmpArray = nullptr;

    }
    else {
    throw runtime_error("Cannot open file `" + full_path + "`!");
    }

    file.close();

} // read_mnist_images
template<class T, class U>
void                            DatasetReader<T,U>::read_mnist_labels(Tensor<U>& dst, std::string dst_name, string full_path, int& number_of_labels){
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

 //    cout << "reading MNIST file " << full_path << " ... " << endl;

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        Tensor<U> _dataset("dataset_" + dst_name, 1,number_of_labels);

        char* tmpArray = new char[number_of_labels];
        file.read((char*)tmpArray, number_of_labels);

        #pragma omp parallel for
        for(int i = 0; i < number_of_labels; i++){
            _dataset.getTensor()[i] = static_cast<U>(tmpArray[i]);
        }





        dst = _dataset;
        dst._name = dst_name;
        delete[] tmpArray;
        tmpArray = nullptr;


        /*unsigned char* _dataset = new unsigned char[number_of_labels];

        for (int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        cout << ".. all labels read!" << endl;
        return _dataset;*/
    }
    else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
    file.close();
} // read_mnist_labels


template<class T, class U>
Tensor<T>&                       DatasetReader<T,U>::get_train_images(){
  return train_images;
}
template<class T, class U>
Tensor<T>&                       DatasetReader<T,U>::get_test_images(){
  return test_images;
}
template<class T, class U>
Tensor<U>&                       DatasetReader<T,U>::get_train_labels(){
  return train_labels;
}
template<class T, class U>
Tensor<U>&                       DatasetReader<T,U>::get_test_labels(){
  return test_labels;
}

template<class T, class U>
int&                            DatasetReader<T,U>::get_nr_train_images_read(){
    return nr_train_images_read;
}
template<class T, class U>
int&                            DatasetReader<T,U>::get_nr_test_images_read(){
    return nr_test_images_read;
}
template<class T, class U>
int&                            DatasetReader<T,U>::get_size_of_image(){
    return size_of_image;
}

template<class T, class U>
void                            DatasetReader<T,U>::centreImage(){
    //std::cout << (int)train_images.mean() <<std::endl;
    train_images.writeAll(-train_images.mean());
    test_images.writeAll(-test_images.mean());
}




#endif
