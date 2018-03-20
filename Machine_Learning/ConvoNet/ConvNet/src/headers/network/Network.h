#ifndef NETWORK
#define NETWORK

#include <Eigen/Dense>
#include <vector>

#include "../node/Node.h"

class Network{
private:

                int     _numberOfLayers;
    std::vector<int>    _numberOfNeurons;
                float   _lambda;
                float   _STEP_SIZE;

    Eigen::MatrixXf  _trainImages;
    Eigen::MatrixXf  _trainLabels;

    Eigen::MatrixXf  _validationImages;
    Eigen::MatrixXf  _validationLabels;

    Eigen::MatrixXf  _testImages;
    Eigen::MatrixXf  _testLabels;

    std::vector<baseNode> _node;

public:

    Network    (int numberOfLayers, std::vector<int> numberOfNeurons, float _lambda,
                float _STEP_SIZE, Eigen::MatrixXf& trainImages, Eigen::MatrixXf& trainLabels,
                Eigen::MatrixXf& testImages,Eigen::MatrixXf& testLabels);
};


#endif
