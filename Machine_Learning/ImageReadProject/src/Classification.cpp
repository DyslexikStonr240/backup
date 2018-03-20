#include"Classification.h"
#include<iostream>
#include<iomanip>
#include<cmath>

Classification::Classification(mnist_dataset_reader *dataset){

    _nr_train_images = dataset->get_nr_train_images_read();
    _nr_test_images = dataset->get_nr_test_images_read();
    _sizeim = dataset->get_size_of_image();

    int _nearestNeighbourArray[_nr_test_images];
    int _sumArray[_nr_test_images];

    unsigned char** testim = dataset->get_test_images();
    unsigned char** trainim = dataset->get_train_images();

    int num_reps = _nr_test_images;
    int sum = 0;
    int tmp = 100000;
    int placeholder = 0;
    for(int i = 0;i < num_reps;i++){
        for(int j = 0;j < _nr_train_images;j++){
            //loop over each picture
            for(int k = 0;k < _sizeim;k++){
                sum += std::abs((int)testim[i][k] - (int)trainim[j][k]);
            }
        if(sum<tmp){
            placeholder = j;
            tmp = sum;
        }
        sum = 0;
        }
    _nearestNeighbourArray[i] = placeholder;
    _sumArray[i] = tmp;
    tmp = 100000;
    sum = 0;
    placeholder = 0;
    }



    for(int i = 0;i < num_reps; i++){
        std::cout << _nearestNeighbourArray[i] << std::endl;
    }

    float _percent = 0.0;
    for(int i = 0; i < num_reps; i++){
        if(dataset->get_test_labels()[i]==dataset->get_train_labels()[_nearestNeighbourArray[i]]){
            _percent++;
        }else{}
    }
    std::cout << "Success rate: " << (_percent/num_reps)*100 << std::endl;

}
