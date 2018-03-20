#ifndef INTERNALNODE
#define INTERNALNODE

#include <Eigen/Dense>
#include "BaseNode.h"

class internalNode: public baseNode{
private:
    baseNode& _previousNode;

public:

            internalNode    (baseNode& previousNode, int numberOfNeurons);
    void    backwardPass    (float STEP_SIZE, float lambda);
    void    forwardPass     ();
};

internalNode::internalNode(baseNode& previousNode, int numberOfNeurons) : baseNode(), _previousNode(previousNode){

    _numberOfNeurons    = numberOfNeurons;
    _layer              = _previousNode._layer + 1;
    _previousNode       = previousNode;
};

void    internalNode::forwardPass(){

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

void    internalNode::backwardPass   (float STEP_SIZE, float lambda){
    auto    invSigFunc   = [](float x){

                        return static_cast<float>((x * (1.0 - x)));
    };
    // For each pass back through the neuron we need to chain rule the incoming gradients
    // through the activation function. This is achieved by taking the derivative of
    // the activation function with respect to the inputs and weights; we do this during the forward pass.
    // After chain ruling the gradient is then split between the weights and the input values by multiplying it respectively.
    // We need to be careful that we only multiply by the weights and not the biases (which have been combined with the weights matrix)
    // to avoid pass gradients backwards with the wrong dimensions.
    _gradients           = _gradients.cwiseProduct(_activations.unaryExpr(invSigFunc));
    //_gradients  = gradients * (_weights.transpose());

    _weights           *= (1 - lambda * STEP_SIZE);
    _weights           -= STEP_SIZE * (_previousNode._activations.transpose() * _gradients);
    _biases             = STEP_SIZE * _gradients;

    _previousNode._gradients  = _gradients * (_weights.transpose());
}

#endif
