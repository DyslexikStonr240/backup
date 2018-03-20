#include <iostream>
#include <Eigen/Dense>
#include "Dataset_reader.h"
#include "Classifier.h"



int main(){

	std::srand(time(NULL));
	Eigen::MatrixXd train_images;//image_size x number_of_images = 784 x 60000
	Eigen::MatrixXd test_images;
	Eigen::VectorXd train_labels;
	Eigen::VectorXd test_labels;
	Dataset_reader reader("./MNIST_FILES", train_images, test_images, train_labels, test_labels);

	Classifier classifier(train_images, test_images, train_labels, test_labels);
	classifier.NeuralNet(10, 60, 0.5);
	//classifier.HyperparameterTuning(10, 0.1);

	return 0;


}
