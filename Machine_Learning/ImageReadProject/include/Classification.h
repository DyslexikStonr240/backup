#include"mnist_dataset_reader.h"

class Classification{
public:
    Classification(mnist_dataset_reader *dataset);



private:
    int     _nr_train_images;
    int     _nr_test_images;
    int     _sizeim;
};
