#include "library.h"

extern "C" {


INTERFACE_EXPORT int hello(int x) {
    return x;
}

INTERFACE_EXPORT NeuralNet *create_neural_net(int *structure, int nb_layers) {
    auto nr = new NeuralNet();

    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    nr->layers_struct = structure;
    nr->nb_layers = nb_layers;
    nr->inputsCache = new double *[nb_layers];
    nr->delta = new double *[nb_layers];
    nr->weights = new double **[nb_layers];

    for (auto l = 0; l < nb_layers; ++l) {
        nr->inputsCache[l] = new double[structure[l] + 1];
        nr->inputsCache[l][0] = 1.0;
        nr->delta[l] = new double[structure[l] + 1];
    }

    for (auto l = 1; l < nb_layers; ++l) {
        nr->weights[l] = new double *[structure[l - 1] + 1];

        for (auto i = 0; i < structure[l - 1] + 1; ++i) {
            nr->weights[l][i] = new double[structure[l] + 1];

            for (int j = 0; j < structure[l] + 1; ++j) {
                nr->weights[l][i][j] = distribution(engine);
            }
        }
    }

    return nr;
}

void calculate_values(NeuralNet *model, double *inputs, bool regression) {
    for (int j = 1; j < model->layers_struct[0] + 1; ++j) {
        model->inputsCache[0][j] = inputs[j - 1];
    }

    for (auto l = 1; l < model->nb_layers; ++l) {
        for (auto j = 1; j < model->layers_struct[l] + 1; ++j) {
            auto sum = 0.0;
            for (int i = 0; i < model->layers_struct[l - 1] + 1; ++i) {
                sum += model->weights[l][i][j] * model->inputsCache[l - 1][j];
            }
            model->inputsCache[l][j] = (regression && (l == (model->nb_layers - 1))) ? sum : std::tanh(sum);
        }
    }
}

double *export_values(NeuralNet *model) {
    int lastLayerIndex = model->nb_layers - 1;

    auto res = new double[model->layers_struct[lastLayerIndex]];
    for (auto j = 1; j < model->layers_struct[lastLayerIndex] + 1; ++j) {
        res[j - 1] = model->inputsCache[lastLayerIndex][j];
    }

    return res;
}

INTERFACE_EXPORT double *neural_net_predict_regression(NeuralNet *model, double *inputs) {
    calculate_values(model, inputs, true);

    return export_values(model);
}

INTERFACE_EXPORT double *neural_net_predict_classification(NeuralNet *model, double *inputs) {
    calculate_values(model, inputs, false);

    return export_values(model);
}

INTERFACE_EXPORT void neural_net_model_train_regression(
        NeuralNet *model,
        double *dataset_in,
        int dataset_cnt,
        int params_cnt,
        double *dataset_expect,
        int output_cnt,
        int it_cnt,
        double a
) {
    train_neural_net(
            model, dataset_in, dataset_cnt, params_cnt,
            dataset_expect, output_cnt, it_cnt,
            a, true
    );
}

INTERFACE_EXPORT void neural_net_model_train_classification(
        NeuralNet *model,
        double *dataset_in,
        int dataset_cnt,
        int params_cnt,
        double *dataset_expect,
        int output_cnt,
        int it_cnt,
        double a
) {
    train_neural_net(
            model, dataset_in, dataset_cnt, params_cnt,
            dataset_expect, output_cnt, it_cnt,
            a, false
    );
}

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
) {
    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_int_distribution<int> distribution(0, dataset_cnt - 1);
    int lastL = model->nb_layers - 1;

    for (int it = 0; it < it_cnt; ++it) {
        int sample = distribution(engine);
        auto sample_inputs = dataset_in + sample * params_cnt;
        auto sample_expect = dataset_expect + sample * output_cnt;

        calculate_values(model, sample_inputs, regression);

        // last layer deltas
        if (regression) {
            for (auto j = 1; j < model->layers_struct[lastL] + 1; ++j) {
                model->delta[lastL][j] = model->inputsCache[lastL][j] - sample_expect[j];
            }
        } else {
            for (auto j = 1; j < model->layers_struct[lastL] + 1; ++j) {
                model->delta[lastL][j] = (1 - pow(model->inputsCache[lastL][j], 2)) *
                                         (model->inputsCache[lastL][j] - sample_expect[j]);
            }
        }

        //hidden layers deltas
        for (auto l = lastL; l >= 2; --l) {
            for (auto i = 1; i < model->layers_struct[l - 1] + 1; ++i) {
                double cnt = 0.0;
                for (int j = 1; j < model->layers_struct[l] + 1; ++j) {
                    cnt += model->weights[l][i][j] * model->delta[l][j];
                }
                model->delta[l - 1][i] = (1 - pow(model->inputsCache[l - 1][i], 2)) * cnt;
            }
        }

        //weights update
        for (auto l = 1; l <= lastL; ++l) {
            for (auto i = 0; i < model->layers_struct[l - 1] + 1; ++i) {
                for (int j = 1; j < model->layers_struct[l] + 1; ++j) {
                    model->weights[l][i][j] += -1 * a * model->inputsCache[l - 1][j] * model->delta[l][j];
                }
            }
        }
    }
}


//================ LINEAR =================

INTERFACE_EXPORT double *create_linear_model(int input_size) {
    auto model = new double[input_size + 1];

    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (int i = 0; i < input_size + 1; i++) {
        model[i] = distribution(engine);
    }

    return model;
}

INTERFACE_EXPORT int linear_model_train_regression(
        double *model,
        double *x_train,
        double *y_train,
        int x_train_len,
        int y_train_len,
        int input_size
) {
    Eigen::MatrixXd X = fill_X(x_train, x_train_len, input_size);
    Eigen::VectorXd Y = fill_Y(y_train, y_train_len);
    Eigen::MatrixXd W = ((X.transpose() * X).inverse() * X.transpose()) * Y;

    for (int i = 0; i < input_size + 1; i++) {
        model[i] = W(i, 0);
    }

    return 0;
}

INTERFACE_EXPORT double linear_model_predict_regression(
        const double *model,
        const double *input,
        int input_size
) {
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
        int input_size
) {
    int w_len = input_size + 1;
    int x_train_size = x_train_len / input_size;
    MatrixXd X = fill_X(x_train, x_train_len, input_size);
    VectorXd Y = fill_Y(y_train, y_train_len);
    auto *example_inputs = new double(input_size);

    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_int_distribution<int> distribution(0, x_train_size - 1);

    for (int e = 0; e < epoch; e++) {
        int epoch_example = distribution(engine);
        for (int k = 1; k < w_len; k++) {
            example_inputs[k - 1] = X(epoch_example, k);
        }

        double tmp = alpha * (
                Y(epoch_example) -
                linear_model_predict_classification(model, example_inputs, input_size)
        );

        for (int j = 0; j < input_size + 1; j++) {
            model[j] = model[j] + tmp * X(epoch_example, j);
        }
    }
    free(example_inputs);

    return 0;
}

INTERFACE_EXPORT double linear_model_predict_classification(
        double *model,
        double *input,
        int input_size
) {
    return sign_of_double(linear_model_predict_regression(model, input, input_size));
}
}
