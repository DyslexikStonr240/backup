#ifndef ENDNODE
#define ENDNODE

#include <Eigen/Dense>
#include "BaseNode.h"

class endNode: public baseNode{
private:
        baseNode& _previousNode;

public:

            endNode     (baseNode& previousNode, int numberOfNeurons);
    float   lossFunc    (Eigen::MatrixXf& inputLabels, float lambda, float STEP_SIZE);
    void    forwardPass ();
};

        endNode::endNode(baseNode& previousNode, int numberOfNeurons) : baseNode(), _previousNode(previousNode){

    _numberOfNeurons    = numberOfNeurons;
    _layer              = _previousNode._layer + 1;
     _previousNode      = previousNode;
};

void    endNode::forwardPass(){

    // We pass in the inputs and add an extra col on the end to fill it with 1's.
    // We then make the weights matrix with the same number of rows as the inputs
    // have columns. This means that we can dot the inputs with the weights and
    // add in the biases in one operation. We also find the derivative of the ouputs
    // with respect to the inputs*weights. This is so we can use it later to chain rule
    // in backpropagation.
    int numberOfInputNeurons = _previousNode._activations.rows();

    if(!(_initialisationCheckFlag)){

        auto random = [&](float x){

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> d(0, 1 / std::sqrt(numberOfInputNeurons));
            return d(gen);
        };

        _weights                    = Eigen::MatrixXf::Zero(_previousNode._activations.cols(), _numberOfNeurons).unaryExpr(random);
        _biases                     = Eigen::MatrixXf::Zero(_previousNode._activations.rows(), _numberOfNeurons).unaryExpr(random);
        _initialisationCheckFlag    = true;
    }

    auto    sigFun      = [](float x){

                        return static_cast<float>(1.0 / (1.0 + std::exp(-x)));
    };

    _activations = ((_previousNode._activations * _weights) + _biases).unaryExpr(sigFun);
}

float   endNode::lossFunc(Eigen::MatrixXf& inputLabels, float lambda, float STEP_SIZE){

    int             correct_class;
    std::ptrdiff_t  index;
    float           count               = 0;

    Eigen::MatrixXf gradientsInternal   = Eigen::MatrixXf::Constant(_activations.rows(), _activations.cols(), 1);
    Eigen::MatrixXf LossGradients       = Eigen::MatrixXf::Zero(_activations.rows(), _activations.cols());

    // Performing the loss function
    for(int i = 0; i < _activations.rows(); i++){

        // We assign the correct classification using the ith entry of the output labels vector.
        // We then find the maximum value in the ith row of the output matrix. This is the class
        // that our neural net thinks is correct. Each time the correct class and the neural nets
        // Classification match we increment the counter. This allows us to keep track of the error
        // during each forward pass.
        _activations.row(i).maxCoeff(&index);
        correct_class       = static_cast<int>(inputLabels(i));
        if(correct_class   == index){

            count += 1 / _activations.rows();
        }

        Eigen::MatrixXf y               = Eigen::MatrixXf::Zero(1, _activations.cols());
        y(correct_class)                = static_cast<float>(1);
        LossGradients.row(i)            = (_activations.row(i) - y);
    }

        LossGradients      /= LossGradients.rows();
        LossGradients       = LossGradients.cwiseProduct(gradientsInternal);
        //_gradients  = LossGradients * _weights.transpose();
        _previousNode._gradients = LossGradients * _weights.transpose();

        _weights           *= (1 - lambda * STEP_SIZE);
        _weights           -= STEP_SIZE * (_previousNode._activations.transpose() * LossGradients);
        _biases             = STEP_SIZE * LossGradients;

        return count;
}

#endif
