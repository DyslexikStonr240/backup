#include "Classifier.h"

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <stdlib.h>
#include <functional>
#include <cmath>
#include <tuple>

#include "Neurons.h"

Classifier::Classifier      (Eigen::MatrixXd& train_images, Eigen::MatrixXd& test_images, Eigen::VectorXd& train_labels, Eigen::VectorXd& test_labels){


    _trainImages    = train_images;
    _trainLabels    = train_labels;
    _testImages     = test_images;
    _testLabels     = test_labels;

    _trainImages       /= (255);
	_trainImagesTest   /= (255);
    _testImages        /= (255);
}

void Classifier::ShuffleData(Eigen::MatrixXd& Data, Eigen::VectorXd& Labels){

    if(Data.rows() != Labels.rows()){

        std::cerr << "Size of Data " << Data.rows() << " is not equal to size of labels " << Labels.rows() << std::endl;
        exit(EXIT_FAILURE);
    }

    Eigen::MatrixXd tmpMat = Data;
    Eigen::VectorXd tmpVec = Labels;

    std::vector<int>   position(Data.rows());
    for(int i = 0; i < position.size(); i++){

        position[i] = i;
    }
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(position), std::end(position), rng);

    for(int i = 0; i < Data.rows(); i++){

        Data.row(i) = tmpMat.row(position[i]);
        Labels(i)   = tmpVec(position[i]);
    }
}

void Classifier::NeuralNet  (int BATCH_SIZE, int NUMBER_OF_EPOCHS, double STEP_SIZE){

    // Defining our constants, Neurons, activation function and gradient of the activation function to be passed into the
    // neural nets. We also check if the total number of images is divisible by
    // the batch size so as to not cause core dumps.
    double  Delta       = 1.0f;
    double  lambda      = 0.0001f;
    auto    sigFun      = [](double x){

                        return 1.0 / (1.0 + std::exp(-x));
    };
    auto    invSigFun   = [](double x){

                        return (x * (1.0 - x));
    };
    Neurons HiddenLayer(0, 100, sigFun, invSigFun);
    Neurons OutputLayer(1, 10, sigFun, invSigFun);

    if(!(_trainImages.rows() % BATCH_SIZE == 0)){

        std::cerr << "BATCH_SIZE must be a factor of " << _trainImages.rows() << std::endl;
        exit(EXIT_FAILURE);
    }





    double trainLoss;
    double trainError;
    double testLoss;
    double testError;
    // Proper training
    for(int i = 0; i < NUMBER_OF_EPOCHS; i++){

        ShuffleData(_trainImages, _trainLabels);

        _trainImagesTest = _trainImages.block(49999, 0, 10000, _trainImages.cols());
        _trainLabelsTest = _trainLabels.segment(49999, 10000);

        // Creating temporary images and labels so as to reshuffle them
        Eigen::MatrixXd tmpMat = _trainImages.block(0, 0, 50000, _trainImages.cols());
        Eigen::VectorXd tmpVec = _trainLabels.segment(0, 50000);

        // Each iteration of stochastic gradient descent
        // on a single batch of images
        float    Error  = 0;
        int ITERATIONS  = static_cast<int>(tmpMat.rows() / BATCH_SIZE);
        for(int j = 0; j < ITERATIONS; j++){

            Eigen::MatrixXd tmpTrainImages = tmpMat.block(j * BATCH_SIZE, 0, BATCH_SIZE, tmpMat.cols());
            Eigen::VectorXd tmpTrainLabels = tmpVec.segment(j * BATCH_SIZE, BATCH_SIZE);

            HiddenLayer.ForwardPass(tmpTrainImages, tmpTrainLabels);
            std::tie(trainLoss, trainError) = OutputLayer.LossFunc(HiddenLayer.getOutputs(), HiddenLayer.getOutputLabels(), Delta, STEP_SIZE, lambda);
            Error += trainError;
            HiddenLayer.Backpass(OutputLayer.getGradientsPassBack(), STEP_SIZE, lambda);
        }

        Error = Error / ITERATIONS;
        std::cout << Error << std::endl;

        HiddenLayer.ForwardPass(_trainImagesTest, _trainLabelsTest);
        std::tie(testLoss, testError) = OutputLayer.ScoreFunc(HiddenLayer.getOutputs(), HiddenLayer.getOutputLabels(), lambda);
        std::cout << testError << std::endl;
    }


    // Final output of test set to see the cost and score
    double outScore;
    double outLoss;
    std::cout << std::endl;
    HiddenLayer.ForwardPass(_testImages, _testLabels);
    std::tie(outLoss, outScore) = OutputLayer.ScoreFunc(HiddenLayer.getOutputs(), HiddenLayer.getOutputLabels(), lambda);
    std::cout << outScore << std::endl;
}

void Classifier::HyperparameterTuning(int BATCH_SIZE, double STEP_SIZE){

    // Defining our constants, Neurons, activation function and gradient of the activation function to be passed into the
    // neural nets. We also check if the total number of images is divisible by
    // the batch size so as to not cause core dumps.
    double  Delta       = 1.0f;
    double  lambda      = 0.01;//0.0001f;
    auto    sigFun      = [](double x){

                        return 1.0 / (1.0 + std::exp(-x));
    };
    auto    invSigFun   = [](double x){

                        return (x * (1.0 - x));
    };
    Neurons HiddenLayer(0, 30, sigFun, invSigFun);
    Neurons OutputLayer(1, 10, sigFun, invSigFun);

    if(!(_trainImages.rows() % BATCH_SIZE == 0)){

        std::cerr << "BATCH_SIZE must be a factor of " << _trainImages.rows() << std::endl;
        exit(EXIT_FAILURE);
    }

    double trainLoss;
    double trainError;
    double testLoss;
    double testError;

    //std::vector<std::vector<double>> vecOfRandomNums[1];
    std::vector<double> Lambda(100);

    std::vector<double> TESTING(100);
    auto _random = [](){

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> d(0.0, 5.0);
        return std::pow(10, -d(gen));
    };
    std::generate(Lambda.begin(), Lambda.end(), _random);

    #pragma omp parallel for
    // Hyperparameter tuning stage with smaller training set of 1000
    for(int i = 0; i < TESTING.size(); i++){

        Neurons HiddenLayer(0, 30, sigFun, invSigFun);
        Neurons OutputLayer(1, 10, sigFun, invSigFun);
        std::cout << "Hyper it " << i << "  - ";
        Eigen::MatrixXd __trainImages = _trainImages.block(0, 0, 2000, _trainImages.cols());
        Eigen::VectorXd __trainLabels = _trainLabels.segment(0, 2000);
        ShuffleData(__trainImages, __trainLabels);

        // Creating temporary images and labels so as to reshuffle them
        Eigen::MatrixXd tmpMat = __trainImages.block(0, 0, 1000, __trainImages.cols());
        Eigen::VectorXd tmpVec = __trainLabels.segment(0, 1000);

        _trainImagesTest = __trainImages.block(1000, 0, 1000, __trainImages.cols());
        _trainLabelsTest = __trainLabels.segment(1000, 1000);



        // Each iteration of stochastic gradient descent
        // on a single batch of images
        float    Error  = 0;
        int epoch = 60;
        int ITERATIONS  = static_cast<int>(tmpMat.rows() / BATCH_SIZE);

        for(int s = 0; s< epoch; s++){
        for(int j = 0; j < ITERATIONS; j++){

            Eigen::MatrixXd tmpTrainImages = tmpMat.block(j * BATCH_SIZE, 0, BATCH_SIZE, tmpMat.cols());
            Eigen::VectorXd tmpTrainLabels = tmpVec.segment(j * BATCH_SIZE, BATCH_SIZE);

            HiddenLayer.ForwardPass(tmpTrainImages, tmpTrainLabels);
            std::tie(trainLoss, trainError) = OutputLayer.LossFunc(HiddenLayer.getOutputs(), HiddenLayer.getOutputLabels(), Delta, STEP_SIZE, lambda, HiddenLayer.getWeightsSum());
            Error += trainError;
            HiddenLayer.Backpass(OutputLayer.getGradientsPassBack(), STEP_SIZE, Lambda[i]);
        }
    }

        Error = Error / ITERATIONS;
        std::cout << Error/epoch << "   ";

        HiddenLayer.ForwardPass(_trainImagesTest, _trainLabelsTest);
        std::tie(testLoss, testError) = OutputLayer.ScoreFunc(HiddenLayer.getOutputs(), HiddenLayer.getOutputLabels(), lambda);
        TESTING[i] = testError;
        std::cout << testError << std::endl;
        std::cout << "Weights sum = " << OutputLayer.getWeightsSum() << std::endl;
    }


    int position = std::distance(TESTING.begin(),std::max_element(TESTING.begin(), TESTING.end()));
    lambda = Lambda.at(position - 1);
    std::cout << "Lambda = " << lambda <<std::endl;
}
