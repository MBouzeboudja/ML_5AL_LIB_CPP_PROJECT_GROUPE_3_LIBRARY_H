#include "utils.h"
#include <random>
using Eigen::MatrixXd;
using Eigen::VectorXd;

double rand_double() {
    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    return dis(engine);
}

double sign_of_double(double value){
    return (value > 0.0 ? 1.0 : -1.0);
}

MatrixXd fill_X(const double *x_train, int x_train_len, int line_len) {
    int matrix_x = x_train_len / line_len;
    MatrixXd X(line_len + 1, matrix_x);
    X.row(0).setOnes();
    for(int i = 0; i < matrix_x; i ++){
        for(int j = 1; j < line_len + 1; j ++) {
            X(j, i) = x_train[j - 1 + (i * line_len)];
        }
    }
    return X.transpose();
}

VectorXd fill_Y(const double *y_train, int y_train_len) {
    VectorXd Y(y_train_len);
    for (int i = 0; i < y_train_len; i++) {
        Y(i) = y_train[i];
    }
    return Y;
}

VectorXd init_W(const double *model, int w_len) {
    VectorXd W(w_len);
    for(int i = 0; i < w_len; i ++){
        W(i) = model[i];
    }
    return W;
}

void fill_model_with_W(double *model, VectorXd W, int w_len){
    for(int i = 0; i < w_len; i++){
        model[i] = W(i);
    }
}