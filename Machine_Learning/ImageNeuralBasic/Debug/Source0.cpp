#include"Tensor.h"
#include"DatasetReader.h"
#include"Classifier.h"
#include<iostream>
#include<string>


//int countCreate = 0;
//int countDestroy = 0;
//float range = 1.0f;

int main(int argc,char** argv){

	std::srand(time(NULL));


	DatasetReader<signed char, unsigned char> reader;
	Classifier<long double, int> classifier(reader);
	//classifier._trainImagesTranspose.out(1);
	//classifier._trainImages.out(1);
	//classifier._weights.out();

	//classifier.scoreFunc(classifier._score, classifier._weights, classifier._trainImages, 0);
	//classifier._score.transpose2D();
	//classifier._score.out(10);
	//std::cout << classifier.LossFunc(classifier._weights, 0, 100) << std::endl;
	//classifier._trainImages.out(1);
	//int count = 0;
	Tensor<long double> Wtry("Wtry",1,1);
	Wtry = classifier.getWeights();
	classifier.getWeights().fillRand();
	long double Loss = classifier.LossFunc(classifier.getWeights(), classifier.getScore(), classifier.getLoss(), 0, 100);
	long double tmpLoss = 0;

	int MIN = 10000;
	int MAX = 20000;
	long double STEP_SIZE = 1.0/(MAX-MIN);

	//for(int i = 0; i < MIN; i++){//reader.get_nr_train_images_read()/100

		//#pragma omp parallel for
		for(int j = 0; j < 1000; j++){

			Wtry.fillRand();
			//Wtry * STEP_SIZE;
			//Wtry.writeAll(1);
			Wtry + classifier.getWeights();
			tmpLoss = classifier.LossFunc(Wtry, classifier.getScore(), classifier.getLoss(), 0,1000);
			//std::cout << tmpLoss << " " << Loss << " " << i << " " << j << " ";
			//std::cout << j << "_" << tmpLoss << "     ";
			if(tmpLoss < Loss){

				Loss = tmpLoss;
				classifier.getWeights() = Wtry;
				//std::cout << j << "_" << Loss << "     ";
			}
		}
		//std::cout << std::endl << i << "_" << Loss << "     ";
	//}

	//classifier.getWeights().fillRand();
	float count = 0.0f;
	int tmp = 0;

	for(int i = MIN; i < MAX; i++){

		tmp = -1000000000;
		classifier.getWeights().dot2D( classifier.getScore(), "TEST", classifier.getTrainImages(), 0, i, i+1);
		classifier.getScore().transpose2D();
		//classifier.getScore().out(1);
		//std::cout << classifier.getTrainLabels().getTensor()[i] << std::endl;
		//std::cout << classifier.getScore().getDim1() << " " << classifier.getScore().getDim2() << std::endl;
		int realJ = 0;
		for(int j = 0; j < classifier.getScore().getDim2(); j++){
			//std::cout << classifier.getScore().getTensor()[j] << "    ";

			if(classifier.getScore().getTensor()[j] > tmp){
				tmp = classifier.getScore().getTensor()[j];
				realJ = j;


			}

		}

		//std::cout << std::endl;
		if(classifier.getTrainLabels().getTensor()[i] == realJ){
			count++;
		}
	}
	std::cout << count/(MAX-MIN) << std::endl;

	return 0;
}
