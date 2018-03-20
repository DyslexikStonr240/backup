#include "Neurons.h"
#include <Eigen/Dense>
#include <stdlib.h>
#include <random>
#include <cmath>
#include <tuple>
#include <functional>


Neurons::Neurons(int LAYER, int NUMBER_OF_NEURONS, double (*activationFunc)(double), double (*gradientActivationFunc)(double)){

    _layer                      = LAYER;
    _numberOfNeurons            = NUMBER_OF_NEURONS;

    _activationFunc             = activationFunc;
    _gradientActivationFunc     = gradientActivationFunc;

    // Defining our random function to seed our weight matrices later
    _random = [&](double x){

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> d(0, 1 / std::sqrt(_numberOfInputNeurons));
        return d(gen);
    };

    // This flag allows us to only seed the weights matrix once
    _initialisationCheckFlag    = false;
}



//GETTERS

Eigen::MatrixXd Neurons::getOutputs(){
    return _outputs;
}
Eigen::MatrixXd Neurons::getInputs(){
    return _inputs;
}
Eigen::MatrixXd Neurons::getWeights(){
    return _weights;
}
Eigen::VectorXd Neurons::getOutputLabels(){
    return _outputLabels;
}
Eigen::MatrixXd Neurons::getGradientsPassBack(){
    return _gradientsPassBack;
}

double          Neurons::getWeightsSum(){

    return _weightsSum;
}



void    Neurons::ForwardPass(Eigen::MatrixXd inputs, Eigen::VectorXd inputLabels, double weightsSum){

    // We pass in the inputs and add an extra col on the end to fill it with 1's.
    // We then make the weights matrix with the same number of rows as the inputs
    // have columns. This means that we can dot the inputs with the weights and
    // add in the biases in one operation. We also find the derivative of the ouputs
    // with respect to the inputs*weights. This is so we can use it later to chain rule
    // in backpropagation.
    _numberOfInputNeurons = inputs.rows();
    _inputs                                             = Eigen::MatrixXd(inputs.rows(), inputs.cols() + 1);
    _inputs.block(0, 0, inputs.rows(), inputs.cols())   = inputs;
    _inputs.col(_inputs.cols() - 1)                     = Eigen::MatrixXd::Constant(_inputs.rows(), 1, 1);


    _inputLabels    = inputLabels;
    _outputLabels   = _inputLabels;

    if(!(_initialisationCheckFlag)){

        _weights                    = Eigen::MatrixXd::Zero(_inputs.cols(), _numberOfNeurons).unaryExpr(_random);
        _initialisationCheckFlag    = true;
    }


    if(_inputs.cols()  != _weights.rows() ){

        std::cerr << "Inputs for neurons in layer " << _layer << " have the wrong dimensions for performing a dot product with the weights" << std::endl;
        std::cerr << "Input cols = " << _inputs.cols() << " and weights rows = " << _weights.rows() << std::endl;
        exit(EXIT_FAILURE);
    }

    _outputs            = (_inputs * _weights).unaryExpr(_activationFunc);
    _gradientsInternal  = (_outputs).unaryExpr(_gradientActivationFunc);
    _weightsSum = weightsSum + _weights.block(0, 0, _weights.rows() - 1, _weights.cols()).lpNorm<2>();
}

void    Neurons::Backpass   (Eigen::MatrixXd inputGradients, double STEP_SIZE, double lambda){

    // For each pass back through the neuron we need to chain rule the incoming gradients
    // through the activation function. This is achieved by taking the derivative of
    // the activation function with respect to the inputs and weights; we do this during the forward pass.
    // After chain ruling the gradient is then split between the weights and the input values by multiplying it respectively.
    // We need to be careful that we only multiply by the weights and not the biases (which have been combined with the weights matrix)
    // to avoid pass gradients backwards with the wrong dimensions.
    _gradientsPassIn    = inputGradients.cwiseProduct(_gradientsInternal);
    _gradientsPassBack  = _gradientsPassIn * (_weights.block(0, 0, _weights.rows() - 1, _weights.cols())).transpose();
    _weights.block(0, 0, _weights.rows() - 1, _weights.cols())  = (1 - lambda * STEP_SIZE) * _weights.block(0, 0, _weights.rows() - 1, _weights.cols());

    _weights           -= STEP_SIZE * (_inputs.transpose() * _gradientsPassIn);
}

std::tuple<double, double>   Neurons::LossFunc   (Eigen::MatrixXd inputs, Eigen::VectorXd inputLabels, double Delta, double STEP_SIZE, double lambda, double weightsSum){

    // The loss function performs the forward pass before evaluating the
    // loss. This means that during the forward pass we don't need to
    // call ForwardPass on the final neuron layer and doing so will slow
    // the performance of the program. This forward pass will actually
    // be modified as the activations will depend on the cost function
    // being implemented. With cross entropy this means that the internal
    // gradients need to be constant 1's. With quadratic cost we need to
    // use the derivative of the activation function.
    _numberOfInputNeurons = inputs.rows();
    _inputs                                             = Eigen::MatrixXd(inputs.rows(), inputs.cols() + 1);
    _inputs.block(0, 0, inputs.rows(), inputs.cols())   = inputs;
    _inputs.col(_inputs.cols() - 1)                     = Eigen::MatrixXd::Constant(_inputs.rows(), 1, 1);

    _inputLabels    = inputLabels;
    _outputLabels   = _inputLabels;



    if(!(_initialisationCheckFlag)){

        _weights = Eigen::MatrixXd::Zero(_inputs.cols(), _numberOfNeurons).unaryExpr(_random);
        _initialisationCheckFlag = true;
    }
    _weightsSum = weightsSum + _weights.block(0, 0, _weights.rows() - 1, _weights.cols()).lpNorm<2>();
    // Performing the forward pass
    _outputs            = (_inputs * _weights).unaryExpr(_activationFunc);
    //_gradientsInternal  = _outputs.unaryExpr(_gradientActivationFunc);
    _gradientsInternal  = Eigen::MatrixXd::Constant(_outputs.rows(), _outputs.cols(), 1);


    int             correct_class;
    std::ptrdiff_t  index;

    double          count               = 0;
    double          Reg                 = _weights.block(0, 0, _weights.rows() - 1, _weights.cols()).lpNorm<2>();
    Eigen::MatrixXd cost                = Eigen::MatrixXd::Zero(_outputs.rows(), 1);
    Eigen::MatrixXd LossGradients       = Eigen::MatrixXd::Zero(_outputs.rows(), _outputs.cols());
    Eigen::MatrixXd logOutputs          = _outputs.unaryExpr([](double x){ return std::log(x);});
    Eigen::MatrixXd logOneMinusOutputs  = _outputs.unaryExpr([](double x){ return std::log(1 - x);});
    Eigen::MatrixXd ones                = Eigen::MatrixXd::Constant(1, _outputs.cols(), 1);

    // Performing the loss function
    for(int i = 0; i < _outputs.rows(); i++){

        // We assign the correct classification using the ith entry of the output labels vector.
        // We then find the maximum value in the ith row of the output matrix. This is the class
        // that our neural net thinks is correct. Each time the correct class and the neural nets
        // Classification match we increment the counter. This allows us to keep track of the error
        // during each forward pass.
        _outputs.row(i).maxCoeff(&index);
        correct_class       = static_cast<int>(_outputLabels(i));
        if(correct_class   == index){

            count++;
        }

        Eigen::MatrixXd y               = Eigen::MatrixXd::Zero(1, _outputs.cols());
        y(correct_class)                = static_cast<double>(1);
        LossGradients.row(i)            = (_outputs.row(i) - y);

        // Working out the cost
        Eigen::MatrixXd y_1 = ones - y;
        cost.row(i) = y * (logOutputs.row(i)).transpose() + y_1 * ((logOneMinusOutputs.row(i)).transpose());
    }

    // During each batch we are finding the average loss over the whole batch and as such we
    // need to divide the total loss by the number of elements in each batch. We then perform the first step in
    // backpropogation. We also return the percentage of correct classifications achieved by the neural net to allow for
    // easier debugging.
    LossGradients                                              /= LossGradients.rows();
    LossGradients                                               = LossGradients.cwiseProduct(_gradientsInternal);
    _gradientsPassBack                                          = LossGradients * (_weights.block(0, 0, _weights.rows() - 1, _weights.cols())).transpose();
    _weights.block(0, 0, _weights.rows() - 1, _weights.cols())  = (1 - lambda * STEP_SIZE) * _weights.block(0, 0, _weights.rows() - 1, _weights.cols());
    _weights                                                   -= STEP_SIZE * (_inputs.transpose() * LossGradients);

    cost  =  (cost.array() + 0.1 * lambda * Reg).matrix();
    cost /= -cost.rows();

    return  std::make_tuple(cost.sum(), count / _outputs.rows());
}

std::tuple<double, double>    Neurons::ScoreFunc  (Eigen::MatrixXd inputs, Eigen::VectorXd inputLabels, double lambda){

    // The score function is used to assess the quality of the neural net on a validation
    // or on a test set. Hence the resetting of inputs and input labels.
    _numberOfInputNeurons = inputs.rows();
    _inputs                                             = Eigen::MatrixXd(inputs.rows(), inputs.cols() + 1);
    _inputs.block(0, 0, inputs.rows(), inputs.cols())   = inputs;
    _inputs.col(_inputs.cols() - 1)                     = Eigen::MatrixXd::Constant(_inputs.rows(), 1, 1);
    _inputLabels                                        = inputLabels;

    if(_inputs.cols() != _weights.rows() ){

        std::cerr << "Inputs for neurons in layer " << _layer << " have the wrong dimensions for performing a dot product with the weights" << std::endl;
        std::cerr << "Input cols = " << _inputs.cols() << " and weights rows = " << _weights.rows() << std::endl;
        exit(EXIT_FAILURE);
    }

    float           count   = 0;
    Eigen::MatrixXd Scores  = (_inputs * _weights);


    double          Reg                 = _weights.block(0, 0, _weights.rows() - 1, _weights.cols()).lpNorm<2>();
    Eigen::MatrixXd cost                = Eigen::MatrixXd::Zero(Scores.rows(), 1);
    Eigen::MatrixXd logOutputs          = Scores.unaryExpr([](double x){ return std::log(x);});
    Eigen::MatrixXd logOneMinusOutputs  = Scores.unaryExpr([](double x){ return std::log(1 - x);});
    Eigen::MatrixXd ones                = Eigen::MatrixXd::Constant(1, Scores.cols(), 1);

    for(int i = 0; i < Scores.rows(); i++){

        std::ptrdiff_t  index;
        Eigen::VectorXd tmpVec = Scores.row(i);

        tmpVec.maxCoeff(&index);
        if(_inputLabels(i) == index){

            count++;
        }

        Eigen::MatrixXd y               = Eigen::MatrixXd::Zero(1, _outputs.cols());
        y(_inputLabels(i))              = static_cast<double>(1);

        // Working out the cost
        Eigen::MatrixXd y_1 = ones - y;
        cost.row(i) = y * (logOutputs.row(i)).transpose() + y_1 * ((logOneMinusOutputs.row(i)).transpose());
    }

    cost  =  (cost.array() + 0.1 * lambda * Reg).matrix();
    cost /= -cost.rows();

    return  std::make_tuple(cost.sum(), count / Scores.rows());

}
