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
        int input_size) {
    return sign_of_double(linear_model_predict_regression(model, input, input_size));
}

// ####### K-means ####### ////

double distance_pow_2(double x1, double z1, double x2, double z2){
    double x = x2 - x1;
    double z = z2 - z1;
    return x*x + z*z;
}

INTERFACE_EXPORT double *k_means(
        double *data,
        int data_size,
        int line_size,
        int nbr_cluster,
        int iteration_number){

    int data_set_line_number = (data_size / line_size);

    auto cluster_means = new double [nbr_cluster * line_size];
    auto cluster_points_count = new int [nbr_cluster];
    auto cluster_centroids = new double [nbr_cluster * line_size];

    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_int_distribution<int> index(0, data_set_line_number - 1);

    //Initial centroids are taken from data set.
    for(auto i=0; i < nbr_cluster; i++){
        int random_index = index(engine);
        cluster_centroids[i * line_size] = data[random_index];
        cluster_centroids[i * line_size +1] = data[random_index + 1];
    }

    std::cout<<"Initial centroids: "<<std::endl;
    for (int i = 0; i < nbr_cluster; ++i) {
        std::cout<<"["<<cluster_centroids[i * line_size] <<"; ";
        std::cout<<cluster_centroids[i * line_size + 1] <<"]";
    }
    std::cout<< std::endl;

    auto cluster_of_each_point = new int[data_set_line_number];
    for(int iteration=0; iteration<iteration_number; iteration++){
        for(int i=0; i<data_set_line_number; i++){
            auto dist_min = std::numeric_limits<double>::max();
            for(int j=0; j<nbr_cluster; j++){
                double dist = distance_pow_2(data[i * line_size], data[i * line_size + 1],
                                                cluster_centroids[j * line_size],cluster_centroids[j * line_size + 1]);
                if (dist_min > dist){
                    dist_min = dist;
                    cluster_of_each_point[i] = j;
                }
            }
        }

        for(int i=0; i < data_set_line_number; i++){
            cluster_means[cluster_of_each_point[i] * line_size] += data[line_size * i];
            cluster_means[cluster_of_each_point[i] * line_size + 1] += data[line_size * i + 1];
            cluster_points_count[cluster_of_each_point[i]] += 1;
        }

        for(int i = 0; i < nbr_cluster; i ++){
            const int count = std::max(1, cluster_points_count[i]);
            cluster_centroids[i * line_size] = cluster_means[i* line_size] / count;
            cluster_centroids[i * line_size +1] = cluster_means[i * line_size +1] / count;
        }
    }

    free(cluster_means);
    free(cluster_of_each_point);
    free(cluster_points_count);

    return cluster_centroids;
}

INTERFACE_EXPORT int linear_model_train_classification_RBF(
        double *model,
        double alpha,
        int epoch,
        double *x_train,
        double *y_train,
        int x_train_len,
        int y_train_len,
        int input_size,
        int nbr_cluster,
        int k_means_iterations,
        double gamma){
    auto centroids  = k_means(x_train, x_train_len, input_size, nbr_cluster,k_means_iterations);
    return 0;
}

INTERFACE_EXPORT int linear_model_train_regression_RBF(
        double *model,
        double *x_train,
        double *y_train,
        int x_train_len,
        int y_train_len,
        int input_size,
        int nbr_cluster,
        int k_means_iterations,
        double gamma){
    auto centroids  = k_means(x_train, x_train_len, input_size, nbr_cluster, k_means_iterations);
    int x_train_lines_size =  x_train_len/input_size;
    int phi_size = nbr_cluster * x_train_lines_size;
    auto phi = new double [phi_size];
    for(int i=0; i < x_train_len/input_size; i++){
        for(int j = 0; j < nbr_cluster; j++){
            phi[i * nbr_cluster + j] = exp(-gamma * distance_pow_2(
                    x_train[i * input_size], x_train[i * input_size + 1],centroids[j * input_size], centroids[j * input_size +1]));
        }
    }

    return linear_model_train_regression(model, phi, y_train, phi_size, y_train_len, nbr_cluster);
}

INTERFACE_EXPORT double linear_model_predict_regression_RBF(
        const double *model,
        const double *input,
        int input_size,
        int nbr_cluster) {

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

}
