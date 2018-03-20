#ifndef NEURONS
#define NEURONS

#include <Eigen/Dense>
#include <vector>
#include <iostream>



class Neurons{
public:

    //Input images must be one image per row
    Neurons(int LAYER, int NUMBER_OF_NEURONS, double (*activationFunc)(double), double (*gradientActivationFunc)(double));

    //GETTERS
    Eigen::MatrixXd getOutputs();
    Eigen::MatrixXd getInputs();
    Eigen::MatrixXd getWeights();
    Eigen::VectorXd getOutputLabels();
    Eigen::MatrixXd getGradientsPassBack();

    double          getWeightsSum();

    void                                ForwardPass (Eigen::MatrixXd inputs, Eigen::VectorXd inputLabels, double weightsSum = 0);
    void                                Backpass    (Eigen::MatrixXd inputGradients, double STEP_SIZE, double lambda);
    std::tuple<double, double>          LossFunc    (Eigen::MatrixXd inputs, Eigen::VectorXd inputLabels, double Delta, double STEP_SIZE, double lambda, double weightsSum = 0);
    std::tuple<double, double>          ScoreFunc   (Eigen::MatrixXd inputs, Eigen::VectorXd inputLabels, double lambda);

private:

    int _layer;
    int _numberOfNeurons;
    double _numberOfInputNeurons;
    double _weightsSum;
    bool _initialisationCheckFlag;

    //inputs and outputs with their respective labels
    Eigen::MatrixXd _inputs;
    Eigen::MatrixXd _outputs;
    Eigen::VectorXd _inputLabels;
    Eigen::VectorXd _outputLabels;

    //Weights and biases for Neuron
    Eigen::MatrixXd _weights;
    Eigen::RowVectorXd _biases;

    //Activation Function
    double (*_activationFunc)(double);
    double (*_gradientActivationFunc)(double);

    //Random number generating function
    std::function<double(double)>  (_random);

    Eigen::MatrixXd _gradientsPassIn;
    Eigen::MatrixXd _gradientsInternal;
    Eigen::MatrixXd _gradientsPassBack;
};

#endif
