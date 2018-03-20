#ifndef CLASSIFIER
#define CLASSIFIER

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <stdlib.h>
#include "Dataset_reader.h"


class Classifier{
public:

            Classifier  (Eigen::MatrixXd& train_images, Eigen::MatrixXd& test_images, Eigen::VectorXd& train_labels, Eigen::VectorXd& test_labels);

    void    ShuffleData (Eigen::MatrixXd& Data, Eigen::VectorXd& Labels);
    void    NeuralNet   (int BATCH_SIZE, int NUMBER_OF_EPOCHS, double STEP_SIZE);
    void    HyperparameterTuning(int BATCH_SIZE, double STEP_SIZE);


private:

    Eigen::MatrixXd _trainImages;
    Eigen::MatrixXd _trainImagesTest;
    Eigen::MatrixXd _testImages;
    Eigen::VectorXd _trainLabels;
    Eigen::VectorXd _trainLabelsTest;
    Eigen::VectorXd _testLabels;

};


#endif
