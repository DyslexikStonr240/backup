#include "Network.h"

Network::Network   (int numberOfLayers, std::vector<int> numberOfNeurons, float lambda,
                    float STEP_SIZE, Eigen::MatrixXf& trainImages, Eigen::MatrixXf& trainLabels,
                    Eigen::MatrixXf& testImages,Eigen::MatrixXf& testLabels){
    if(numberOfLayers != numberOfNeurons.size()){

        std::cerr << "Number of layers must equal the number of elements in the vector 'numberOfNeurons'" << std::endl;
        exit(EXIT_FAILURE);
    }
    _numberOfLayers     = numberOfLayers;
    _numberOfNeurons    = numberOfNeurons;
    _lambda             = lambda;
    _STEP_SIZE          = STEP_SIZE;
    _trainImages        = trainImages.block(0, 0, 50000, trainImages.cols());
    _trainLabels        = trainLabels.block(0, 0, 50000, trainImages.cols());

    _validationImages   = trainImages.block(49999, 0, 10000, trainImages.cols());
    _validationLabels   = trainLabels.block(49999, 0, 10000, trainImages.cols());

    _testImages         = testImages;
    _testLabels         = testLabels;

    _trainImages       /= (255);
	_validationImages  /= (255);
    _testImages        /= (255);

    _node.push_back(firstNode(numberOfNeurons.front()));
    for(int i = 1; i < numberOfLayers - 1; i++){

        _node.push_back(internalNode(_node.at(i - 1), numberOfNeurons.at(i)));
    }
    _node.push_back(endNode(_node.back(), numberOfNeurons.back()));
}

void Network::trainNetwork(int BATCH_SIZE, int NUMBER_OF_EPOCHS, double STEP_SIZE){


}

Eigen::MatrixXf test = Eigen::MatrixXf::Zero(4,4);
Eigen::MatrixXf testLabels = Eigen::MatrixXf::Zero(1,4);
float STEP_SIZE = 0.1;
float lambda = 0.1;
FirstNode first(10);
InternalNode internal(first, 5);
EndNode end(internal, 4);

first.forwardPass(test);
internal.forwardPass();
end.forwardPass();

end.lossFunc(testLabels, lambda, STEP_SIZE);
internal.backwardPass(lambda, STEP_SIZE);
first.backwardPass(lambda, STEP_SIZE);
