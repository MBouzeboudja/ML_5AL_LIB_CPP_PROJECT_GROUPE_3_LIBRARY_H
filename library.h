#ifdef __linux__
#define INTERFACE_EXPORT
#elif _WIN32
#define INTERFACE_EXPORT __declspec(dllexport)
#endif
#ifndef ML_5AL_LIB_CPP_PROJECT_GROUPE_3_LIBRARY_H
#define ML_5AL_LIB_CPP_PROJECT_GROUPE_3_LIBRARY_H


extern "C" {
    INTERFACE_EXPORT double *create_linear_model(int input_dim);

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
#endif //ML_5AL_LIB_CPP_PROJECT_GROUPE_3_LIBRARY_H


