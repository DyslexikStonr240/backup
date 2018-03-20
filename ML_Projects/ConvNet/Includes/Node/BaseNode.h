#ifndef BASENODE
#define BASENODE

#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>

///////////////////////////
///////////////////////////
//Basenode definition class
class baseNode{
protected:

    bool            _initialisationCheckFlag;
public:

    int             _layer;
    int             _numberOfNeurons;
    Eigen::MatrixXf _weights;
    Eigen::MatrixXf _biases;
    Eigen::MatrixXf _activations;
    Eigen::MatrixXf _gradients;

    virtual void    forwardPass(){};

};


#endif
