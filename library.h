#pragma once

#include <iostream>
#include <random>
#include "utils.h"
#include "Eigen/Eigen"

#define INTERFACE_EXPORT __declspec(dllexport)


extern "C" {
typedef struct NeuralNet {
    int nb_layers;
    int *layers_struct;
    double **inputsCache;
    double **delta;
    double ***weights;
} NeuralNet;

INTERFACE_EXPORT int hello(int x);

void calculate_values(NeuralNet *model, double *inputs, bool regression);

INTERFACE_EXPORT NeuralNet *create_neural_net(int *structure, int nb_layers);
INTERFACE_EXPORT double *neural_net_predict_regression(NeuralNet *model, double *inputs);
INTERFACE_EXPORT double *neural_net_predict_classification(NeuralNet *model, double *inputs);
void train_neural_net(
        NeuralNet *model,
        double *dataset_in,
        int dataset_cnt,
        int params_cnt,
        double *dataset_expect,
        int output_cnt,
        int it_cnt,
        double a,
        bool regression
);
INTERFACE_EXPORT void neural_net_model_train_classification(
        NeuralNet *model,
        double *dataset_in,
        int dataset_cnt,
        int params_cnt,
        double *dataset_expect,
        int output_cnt,
        int it_cnt,
        double a
);

INTERFACE_EXPORT double *create_linear_model(int input_size);

INTERFACE_EXPORT int linear_model_train_regression(
        double *model,
        double *x_train,
        double *y_train,
        int x_train_len,
        int y_train_len,
        int input_size);

INTERFACE_EXPORT int linear_model_train_classification(
        double *model,
        double alpha,
        int epoch,
        double *x_train,
        double *y_train,
        int x_train_len,
        int y_train_len,
        int input_size);

INTERFACE_EXPORT double linear_model_predict_regression(
        const double *model,
        const double *input,
        int input_size);

INTERFACE_EXPORT double linear_model_predict_classification(
        double *model,
        double *input,
        int input_size);
}

