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
	std::uniform_real_distribution<double> d(0.0, 1.0);
	return d(gen);
};
std::generate(Lambda.begin(), Lambda.end(), _random);

// Hyperparameter tuning stage with smaller training set of 1000
for(int i = 0; i < TESTING.size(); i++){

	Neurons HiddenLayer(0, 30, sigFun, invSigFun);
	Neurons OutputLayer(1, 10, sigFun, invSigFun);
	std::cout << "Hyper it " << i << "  - ";
	//ShuffleData(_trainImages, _trainLabels);

	// Creating temporary images and labels so as to reshuffle them
	Eigen::MatrixXd tmpMat = _trainImages.block(0, 0, 1000, _trainImages.cols());
	Eigen::VectorXd tmpVec = _trainLabels.segment(0, 1000);

	_trainImagesTest = _trainImages.block(1000, 0, 1000, _trainImages.cols());
	_trainLabelsTest = _trainLabels.segment(1000, 1000);



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
		std::tie(trainLoss, trainError) = OutputLayer.LossFunc(HiddenLayer.getOutputs(), HiddenLayer.getOutputLabels(), Delta, STEP_SIZE, Lambda[i]);
		Error += trainError;
		HiddenLayer.Backpass(OutputLayer.getGradientsPassBack(), STEP_SIZE, Lambda[i]);
	}
}

	Error = Error / ITERATIONS;
	std::cout << Error/epoch << "   ";

	HiddenLayer.ForwardPass(_trainImagesTest, _trainLabelsTest);
	std::tie(testLoss, testError) = OutputLayer.ScoreFunc(HiddenLayer.getOutputs(), HiddenLayer.getOutputLabels(), Lambda[i]);
	TESTING[i] = testError;
	std::cout << testError << std::endl;
}


int position = std::distance(TESTING.begin(),std::max_element(TESTING.begin(), TESTING.end()));
lambda = Lambda.at(position - 1);
std::cout << "Lambda = " << lambda <<std::endl;
}
