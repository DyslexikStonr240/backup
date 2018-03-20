#ifndef CLASSIFIER
#define CLASSIFIER

#include<iostream>
#include<string>
#include<algorithm>
#include"Tensor.h"
#include"DatasetReader.h"

template <class T, class U>//T = long double, U = Unsigned char??
class Classifier{
public:

    template <class S, class V>
    Classifier(DatasetReader<S,V>& reader);

    DatasetReader<T,U> _reader;

    Tensor<T>& getWeights();
    Tensor<T>& getTrainImages();
    Tensor<T>& getTrainImagesTranspose();
    Tensor<T>& getTrainLabels();
    Tensor<T>& getScore();
    Tensor<T>& getLoss();




    //template<typename T>
    void                    scoreFunc(Tensor<T>& dst, std::string dst_name, Tensor<T>& weights, Tensor<T>& pixels, int TRANSFLAG = 1, int colsBstart = 0, int colsBend = 0);
    T                       scoreReturn(Tensor<T>& score, int image, int Class);
    T                       LossFunc(/*Tensor<T>& dst, Tensor<T>& score,*/Tensor<T>& weights, Tensor<T>& score, Tensor<T>& loss, int colsBstart = 0, int colsBend = 0);


private:



    Tensor<T>               _weights;
    Tensor<T>               _trainImages;
    Tensor<T>               _trainImagesTranspose;
    Tensor<T>               _trainLabels;
    Tensor<T>               _score;           //Columns = each class, Rows = each picture
    Tensor<T>               _loss;

};

template <class T, class U>
template <class S, class V>
                            Classifier<T,U>::Classifier(DatasetReader<S,V>& reader){

    _reader = reader;
    Tensor<long double> tmpWeight("tmp_Weights_Classifier", 10, _reader.get_size_of_image() + 1);
    tmpWeight.fillRand();
    //tmpWeight.writeAll(1);
    _weights = tmpWeight;
    _weights._name = "Classifier_weights";

 ///////////TRAIN IMAGES/////////////
    _trainImages = _reader.get_train_images();
    _trainImages._name = "Classifier_Train_Images";
    //_trainImagesTranspose = _trainImages;


    //_weights.fillRand();
    _trainImages.centre();
    _trainImages.transpose2D();

    // including the bias
    //#pragma omp parallel for
    for(int i = 0; i < _trainImagesTranspose.getDim1()*_trainImagesTranspose.getDim2(); i++){

        if( (i % (_reader.get_size_of_image()+1)) == 0 && i != 0 ){
            _trainImages.getTensor()[i] = 1;
        }else{}
    }

    _trainImagesTranspose = _trainImages;
    _trainImages.transpose2D();

 ///////////TRAIN LABELS/////////////

    _trainLabels = _reader.get_train_labels();
    _trainLabels._name = "Classifier_Train_Labels";


}

template <class T, class U>
Tensor<T>& Classifier<T,U>::getWeights(){

    return _weights;
}

template <class T, class U>
Tensor<T>& Classifier<T,U>::getTrainImages(){

        return _trainImages;
}

template <class T, class U>
Tensor<T>& Classifier<T,U>::getTrainImagesTranspose(){

        return _trainImagesTranspose;
}

template <class T, class U>
Tensor<T>& Classifier<T,U>::getTrainLabels(){

        return _trainLabels;
}

template <class T, class U>
Tensor<T>& Classifier<T,U>::getScore(){

        return _score;
}

template <class T, class U>
Tensor<T>& Classifier<T,U>::getLoss(){

        return _loss;
}



template <class T, class U>
void                     Classifier<T,U>::scoreFunc(Tensor<T>& dst, std::string dst_name, Tensor<T>& weights, Tensor<T>& pixels, int TRANSFLAG,  int colsBstart, int colsBend){

    weights.dot2D(dst, dst_name, pixels, TRANSFLAG, colsBstart, colsBend);
    dst.transpose2D();

}

template <class T, class U>
T                        Classifier<T,U>::scoreReturn(Tensor<T>& score, int image, int Class){

    if(image > score.getDim1() || Class > score.getDim2()){
        std::cout << "ERROR:    Dimensions of matrices are out of bounds" << __FILE__ << ": line" << __LINE__ << std::endl << std::endl << std::endl;
        exit(EXIT_FAILURE);
    }else{
        return score.getTensor()[image*score.getDim2() + Class];
    }

}

template <class T, class U>
T                        Classifier<T,U>::LossFunc(/*Tensor<T>& dst, Tensor<T>& score,*/Tensor<T>& weights, Tensor<T>& score, Tensor<T>& loss, int colsBstart, int colsBend){

    //Image Class i             Li = sum[ max( 0, Sj - Syi + DELTA) ] // j =/= yi
    float Delta = 1.0f;
    float Lambda = 1.0f;
    float Regularisation = 0.0f;
    float totalLoss = 0.0f;
    Tensor<long double> tmpTensor("LossFunc_tmp_tensor", colsBend - colsBstart/*_reader.get_nr_train_images_read()*/, 1);
    //std::cout << colsBend - colsBstart << std::endl << std::endl;
    //weights.writeAll(1);
    long double tmp = 0.0f;

    // std::cout << "DOT" << std::endl;
    weights.dot2D(score, "Classifier_Scores", _trainImages, 0, colsBstart, colsBend);
    // std::cout << "DOT DONE" << std::endl;

    score.transpose2D();
    //score.out(1);



    //std::cout << score.getDim1() << " " << score.getDim2() << std::endl;

    //one Li for each picture
    for(int i = 0; i < score.getDim1(); i++){

        tmp = 0;
        //sum of max's to get Li
        int correct_class = static_cast<int>(_trainLabels.getTensor()[i]);
        for(int j = 0; j < score.getDim2(); j++){

            if(j != correct_class){

                tmp += static_cast<long double>(std::max(static_cast<long double>(0), scoreReturn(score, i, j) - scoreReturn(score, i, correct_class) + Delta));
            }else{}
        }
        //std::cout << tmpTensor.getTensor()[i] << " ";// tmp;
        tmpTensor.getTensor()[i] = tmp;
        //std::cout << tmp << "    ";
    }

    //std::cout << std::endl << tmp << " ";
    _loss = tmpTensor;
    _loss._name = "Classifier_loss_tensor";
    //std::cout << std::endl << "LOSS I DONE" << std::endl;

    for(int i = 0; i < weights.getDim1(); i++){
        for(int j = 0; j < weights.getDim2(); j++){

            Regularisation += ( weights.getTensor()[i * weights.getDim2() + j] ) * ( weights.getTensor()[i * weights.getDim2() + j] );
        }
    }

    //std::cout << std::endl << "REGULARISATION I DONE" << std::endl;

    for(int i = 0; i < loss.getDim1(); i++){

        totalLoss += loss.getTensor()[i];
    }
    //totalLoss = loss.sum();

    //std::cout << std::endl << "TOTAL_LOSS I DONE" << std::endl;

    totalLoss = totalLoss / colsBend - colsBstart/*_reader.get_nr_train_images_read()*/;
    //totalLoss += Lambda * Regularisation;

    return totalLoss;
}


#endif
