#include "library.h"
#include <iostream>
#include <random>
#include "Eigen"
#include "utils.h"

INTERFACE_EXPORT double *create_linear_model(int input_dim){
    auto model = new double[input_dim];
    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (int i = 0; i < input_dim + 1; i++) {
        model[i] = distribution(engine);
    }
    return model;
}

INTERFACE_EXPORT int train_regression_linear_model(
        double *model,
        double *x_train,
        double *y_train,
        int x_train_len,
        int y_train_len,
        int input_dim) {

    Eigen::MatrixXd X = fill_X(x_train, x_train_len, input_dim);
    std::cout << "Matrix X\n";
    std::cout << X;
    Eigen::VectorXd Y = fill_Y(y_train, y_train_len);
    std::cout << "\nVector Y\n";
    std::cout << Y;
    // W = ((T(X)-1 * T(X)) * Y
    Eigen::MatrixXd W = ((X.transpose() * X).inverse() * X.transpose()) * Y;
    std::cout << "\nVector Y\n";
    std::cout << W;
    for (int i = 0; i < input_dim + 1; i++) {
        model[i] = W(i, 0);
    }
    return 0;
}

INTERFACE_EXPORT double predict_regression_linear_model(
        const double *model,
        const double *input,
        int input_size) {
    double result = 0.0;

    for (int i = 0; i < input_size + 1; i++) {
        if (i == 0) {
            result += 1 * model[i];
        } else {
            result += input[i - 1] * model[i];
        }
    }

    return result;
}

INTERFACE_EXPORT int linear_model_train_classification(
        double *model,
        double alpha,
        int epoch,
        double *x_train,
        double *y_train,
        int x_train_len,
        int y_train_len,
        int input_size) {
    int w_len = input_size + 1;
    int x_train_size = x_train_len / input_size;
    VectorXd W = init_W(model, w_len);
    std::cout << "\nInitiated Vector W\n";
    std::cout << W;
    MatrixXd X = fill_X(x_train, x_train_len, input_size);
    std::cout << "\nMatrix X\n";
    std::cout << X;
    VectorXd Y = fill_Y(y_train, y_train_len);
    std::cout << "\nVector Y\n";
    std::cout << Y;
    auto *intermediate_input = static_cast<double *>(malloc(sizeof(double) * input_size));

    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_int_distribution<int> distribution(0, x_train_size - 1);

    for (int e = 0; e < epoch; e++) {
        int epoch_train_index = distribution(engine);
        for (int k = 1; k < w_len; k++) {
            intermediate_input[k - 1] = X(epoch_train_index, k);
        }
        double tmp = alpha * (Y(epoch_train_index) - linear_model_predict_classification(model, intermediate_input, input_size));

        for (int j = 0; j < w_len; j++) {
            W(j) = W(j) + tmp * X(epoch_train_index, j);
        }
        std::cout << "\nVector W:\n";
        std::cout << W;
        fill_model_with_W(model, W, w_len);
    }
    free(intermediate_input);
    return 0;
}

INTERFACE_EXPORT double linear_model_predict_classification(
        double *model,
        double *input,
        int input_size) {
    return sign_of_double(predict_regression_linear_model(model, input, input_size));
}