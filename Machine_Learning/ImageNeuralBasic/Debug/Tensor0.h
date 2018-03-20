#ifndef TENSOR_H
#define TENSOR_H


#include<iostream>
#include<stdexcept>
#include<typeinfo>
#include<stdlib.h>
#include<ctime>
#include<omp.h>
#include<string>

//extern int countCreate;
//extern int countDestroy;
//extern float range;

template <class T>

class Tensor{
public:
    Tensor( std::string name, int dim1, int dim2);
    Tensor();

    //template <class U>
    //Tensor(Tensor<U> &tensorCopy);

    template <class U>
    Tensor<T> & operator=(Tensor<U>& rhs);

    template <class U>
    Tensor<T>& operator+(Tensor<U>& rhs);

    Tensor<T>& operator*(T rhs);

    ~Tensor(){
        //std::cout << "matrix " << _name << " pointer deleted at " << &*_matrix << std::endl << std::endl;
        delete[] _matrix;
        _matrix = nullptr;
    }


    void            writeAll(T information);
    void            out(int rows = 0, int cols = 0);

    void            fillRand();
    T               mean();
    void            transpose2D();

    void            dot2D(Tensor<T>& dot, std::string dot_name, Tensor<T>& tensor, int TRANSFLAG = 1, int colsBstart = 0, int colsBend = 0, int rowsA = 0, int colsB = 0);

    void            centre();
    long double     sum();




    T*              getTensor();
    int             getDim1();
    int             getDim2();
    void            setDim1(int arg);
    void            setDim2(int arg);

    std::string     _name;



private:

    T*              _matrix;
    int             _dim1, _dim2;

};

template <class T>
Tensor<T>::Tensor( std::string name, int dim1, int dim2){

    _name = name;
    _dim1 = dim1;
    _dim2 = dim2;

    int size = _dim1*_dim2;
    _matrix = new T[size];
    //std::cout << "OG Tensor " << _name << " Created at " << &*_matrix << std::endl;
}

template <class T>
Tensor<T>::Tensor(){

    _name = "blank";
    _dim1 = 2;
    _dim2 = 2;

    //int size = _dim1*_dim2;
    _matrix = new T[10];
    //std::cout << "Empty Tensor " << _name << " Created at " << &*_matrix << std::endl;
}


/*template <class T>
template <class U>
Tensor<T>::Tensor(Tensor<U> &tensorCopy){

    _name = tensorCopy._name + "_COPY";
    _dim1 = tensorCopy.getDim1();
    _dim2 = tensorCopy.getDim2();
    int size = _dim2*_dim1;
    std::cout << "Deleting Matrix " << _name << " at " << &*_matrix << std::endl;
    delete[] _matrix;
    _matrix = new T[size];
    std::cout << "Creating Copy Matrix" << _name << " at " << &*_matrix << std::endl;
    //delete[] _matrix;

    //#pragma omp parallel for
    for(int i = 0; i < _dim1; i++){
        for(int j = 0; j < _dim2; j++){
            _matrix[_dim2*i + j] = static_cast<T>((tensorCopy.getTensor()[_dim2*i + j]));
        }
    }
    //delete[] &tensorCopy.getTensor();
    //tensorCopy.getTensor() = nullptr;


    //std::cout << "Const Copycreated" << std::endl;
    std::cout << "Copy " << _name << " at " << &*_matrix << " Copied from " << tensorCopy._name << " at " << &*tensorCopy.getTensor() << std::endl << std::endl;
    this->out(1,3);
}
*/

template <class T>
template <class U>
Tensor<T>& Tensor<T>::operator=(Tensor<U>& rhs){

    //omp_lock_t my_lock;
    //omp_init_lock(&my_lock);
    if(_name == "blank"){

        _name = rhs._name + "_equals";
    }

    _dim1 = rhs.getDim1();
    _dim2 = rhs.getDim2();
    //std::cout << "Deleting Matrix " << _name << " at " << &*_matrix << std::endl;
    delete[] _matrix;
    _matrix = new T[_dim1*_dim2];
    //_matrix = rhs._matrix;


    //U tmp = 0;
    #pragma omp parallel for
    for(int i = 0; i < _dim1; i++){
        for(int j = 0; j < _dim2; j++){

            _matrix[_dim2*i + j] = static_cast<T>(rhs.getTensor()[_dim2*i + j]);

            //omp_set_lock(&my_lock);
            //tmp  = rhs.getTensor()[_dim2*i + j];
            //tmp = static_cast<T>(tmp);
            //_matrix[_dim2*i + j] = tmp;
            //omp_unset_lock(&my_lock);
        }
    }
    //rhs._matrix = nullptr;
    //omp_destroy_lock(&my_lock);

    //std::cout << "Tensor<T,U> Equal " << _name << " created at " << &*_matrix << std::endl;
    //std::cout << "Tensor<T,U> Equal " << _name << " at " << &*_matrix << " Copied from " << rhs._name << " at " << &*rhs.getTensor() << std::endl << std::endl;
    return *this;
}

template <class T>
template <class U>
Tensor<T>& Tensor<T>::operator+(Tensor<U>& rhs){

    /*_name = rhs._name + "_equals";
    _dim1 = rhs.getDim1();
    _dim2 = rhs.getDim2();

    delete[] _matrix;
    _matrix = new T[_dim1*_dim2];*/
    if(rhs.getDim1()!= _dim1 || rhs.getDim2()!=_dim2){

                    std::cout << "ERROR:    Dimensions of matrices are wrong for addition in file" << __FILE__ << ": line" << __LINE__ << std::endl << std::endl << std::endl;
                    exit(EXIT_FAILURE);
    }
    else{
        #pragma omp parallel for
        for(int i = 0; i < _dim1; i++){
            for(int j = 0; j < _dim2; j++){

                _matrix[_dim2*i + j] += static_cast<T>(rhs.getTensor()[_dim2*i + j]);
            }
        }

        return *this;
    }
}

template <class T>
Tensor<T>& Tensor<T>::operator*(T rhs){


    #pragma omp parallel for
    for(int i = 0; i < _dim1; i++){
        for(int j = 0; j < _dim2; j++){

            _matrix[_dim2*i + j] *= rhs;
        }
    }

    return *this;
}



template <class T>
void Tensor<T>::writeAll(T information){

    int index = information*0;

    //#pragma omp parallel for
    for(int i = 0; i < _dim1; i++){
        for(int j = 0; j < _dim2; j++){
                    _matrix[index++] = information;
        }
    }
}

template <class T>
void Tensor<T>::out(int rows, int cols){

    if(rows == 0){
        rows = _dim1;
    }
    if(cols == 0){
        cols = _dim2;
    }
    if(rows > _dim1 || cols > _dim2){
        std::cout << "Output dimensions out of range!" << std::endl;
    }else{
        //int index = 0;
        for(int i = 0; i < rows; i++){
            std::cout << std::endl;
            std::cout << i + 1 << "   ";
            for(int j = 0; j < cols; j++){

                        std::cout << /*static_cast<float>*/(_matrix[i*_dim2 + j]) << "  ";
            } //- (signed char)'0') << "
        std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

template <class T>
void Tensor<T>::fillRand(){

    int index = 0;
    for(int i = 0; i < _dim1; i++){
        for(int j = 0; j < _dim2; j++){
                    _matrix[index] = static_cast <float> (std::rand())/ static_cast <float> (RAND_MAX);// - range/2.0f;
                    //_matrix[index] = 2;
                    index++;
        }
    }
}




template <class T>
void Tensor<T>::transpose2D(){

    long long int size = _dim1*_dim2;
    T* tmpMat = new T[size];
    tmpMat[0] = _matrix[0];
    tmpMat[size-1] = _matrix[size-1];
    #pragma omp parallel for
    for(long long int n = 1; n<size-1; n++) {

        tmpMat[_dim1*n%(size - 1)] = _matrix[n];
    }


    //#pragma omp parallel for
    /*for(int n = 1; n<_dim1*_dim2-1; n++) {
        int i = n/_dim2;                    //lower bound on n/_dim1
        int j = n%_dim2;                    //remainder of n/_dim1
        std::cout << i << " " << j << std::endl;;
        tmpMat[n] = _matrix[_dim2*j + i];   //
    }*/
    int index = 0;
    for(int i = 0; i < _dim2; i++){
        for(int j = 0; j < _dim1; j++){
            _matrix[index] = tmpMat[index];
            index++;
        }
    }

    delete[] tmpMat;
    tmpMat = nullptr;

    int tmp = _dim1;
    _dim1 = _dim2;
    _dim2 = tmp;
}

template <class T>
void Tensor<T>::dot2D(Tensor<T>& dot, std::string dot_name, Tensor<T>& tensor, int TRANSFLAG, int colsBstart, int colsBend, int rowsA, int colsB){

    if(TRANSFLAG == 1){
        if(rowsA == 0){
            rowsA = _dim1;
        }
        if(colsB == 0){
            colsB = tensor.getDim2();
        }
        if(colsBend == 0){
            colsBend = tensor.getDim2();
        }
        Tensor<T> dst(dot._name + "transpose", rowsA, colsBend - colsBstart);
        /*dst.setDim1(_dim1);
        dst.setDim2(tensor.getDim2);
        dst.Tensor(_dim1,tensor.getDim2());*/
        //Tensor<T> dst(_dim1, tensor.getDim2() );

        if( (_dim2 != tensor.getDim1() )){

            std::cout << "ERROR:    Dimensions of matrices are wrong for inner product in file" << __FILE__ << ": line" << __LINE__ << std::endl << std::endl << std::endl;
            exit(EXIT_FAILURE);
        }
        else{

            //std::cout << "" << std::endl;
            tensor.transpose2D();


            T* _tensorPtr = tensor.getTensor();

            #pragma omp parallel for
            for(int i=0; i<rowsA; i++) {//rowsA = _dim1
                for(int j=colsBstart; j<colsBend; j++) {//colsB = tensor.getDim1()

                    float tmp = 0;
                    for(int l=0; l<_dim2; l++) {//lrange = _dim2
                        tmp += _matrix[_dim2*i+l]*_tensorPtr[_dim2*j+l];
                    }
                    //std::cout << tmp << " ";
                    dst.getTensor()[_dim2*i + j] = tmp;

                }
                //std::cout << std::endl;
            }


            //_tensorPtr = nullptr;
            tensor.transpose2D();
        }

        dot = dst;
        dot._name = dot_name;
    }

    else{
        if(rowsA == 0){
            rowsA = _dim1;
        }
        if(colsB == 0){
            colsB = tensor.getDim1();
        }
        if(colsBend == 0){
            colsBend = tensor.getDim1();
        }
        Tensor<T> dst(dot._name + "_no_transpose", rowsA, colsBend - colsBstart);
        /*dst.setDim1(_dim1);
        dst.setDim2(tensor.getDim2);
        dst.Tensor(_dim1,tensor.getDim2());*/
        //Tensor<T> dst(_dim1, tensor.getDim2() );

        if( (_dim2 != tensor.getDim2() )){

            std::cout << "ERROR:    Dimensions of matrices are wrong for inner product in file" << __FILE__ << ": line" << __LINE__ << std::endl << std::endl << std::endl;
            exit(EXIT_FAILURE);
        }
        else{


            T* _tensorPtr = tensor.getTensor();

            #pragma omp parallel for
            for(int i=0; i<rowsA; i++) {//rowsA = _dim1
                int realJ = 0;
                for(int j=colsBstart; j<colsBend; j++) {//colsB = tensor.getDim1()


                    T tmp = 0;
                    for(int l=0; l<_dim2; l++) {//lrange = _dim2
                        tmp += _matrix[_dim2*i+l]*_tensorPtr[_dim2*j+l];
                    }
                    //std::cout << tmp << " ";
                    dst.getTensor()[dst.getDim2()*i + realJ] = tmp;
                    realJ++;

                }
                //std::cout << std::endl;
            }

        }
        //dst.transpose2D();
        //dst.out();
        //std::cout << "DELETING" << std::endl;
        //std::cout << dst.getDim1() << dst.getDim2() << dot.getDim1() << dot.getDim2() << std::endl;

        dot = dst;
        dot._name = dot_name;

    }


}


template <class T>
long double Tensor<T>::sum(){

    int index = 0;
    T sum = 0.0f;
    for(int i = 0; i < _dim1; i++){
        for(int j = 0; j < _dim2; j++){
                    sum += _matrix[index++];
        }
    }
    return sum;
}

template <class T>
T Tensor<T>::mean(){

    //int mean = 0;
    int index = 0;
    float sum = 0.0f;
    for(int i = 0; i < _dim1; i++){
        for(int j = 0; j < _dim2; j++){
                    sum += _matrix[index++];
        }
    }
    sum /= _dim1*_dim2;
    //std::cout << sum << std::endl;
    /*if(sum > 0){
        mean = (sum + 0.5f);
    }else{
        mean = (sum - 0.5f);
    }*/


    return sum;
}

template <class T>
void Tensor<T>::centre(){

    long double mean = 0.0f;
    int index = 0;
    long double sum = 0.0f;
    for(int i = 0; i < _dim1; i++){
        for(int j = 0; j < _dim2; j++){
                    sum += static_cast<long double>(_matrix[index]);
                    index++;
        }
    }
    sum /= static_cast<long double>(_dim1)*static_cast<long double>(_dim2);
    mean = sum;
    //std::cout << mean << std::endl;

    index = 0;
    for(int i = 0; i < _dim1; i++){
        for(int j = 0; j < _dim2; j++){
                    _matrix[index] -= static_cast<T>(mean);
                    index++;
                    //_matrix[index] /= 128;
        }
    }
}




template <class T>
T* Tensor<T>::getTensor(){

    return _matrix;
}


template <class T>
int Tensor<T>::getDim1(){

    return _dim1;
}

template <class T>
int Tensor<T>::getDim2(){

    return _dim2;
}

template <class T>
void Tensor<T>::setDim1(int arg){

    _dim1 = arg;
}

template <class T>
void Tensor<T>::setDim2(int arg){

    _dim2 = arg;
}





#endif
