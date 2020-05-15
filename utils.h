#ifndef ML_5AL_LIB_CPP_PROJECT_GROUPE_3_UTILS_H
#define ML_5AL_LIB_CPP_PROJECT_GROUPE_3_UTILS_H

#include "Eigen/Eigen"

using Eigen::MatrixXd;
using Eigen::VectorXd;

double rand_double();
double sign_of_double(double value);
MatrixXd fill_X(const double *x_train, int x_train_len, int line_len);
VectorXd fill_Y(const double *y_train, int y_train_len);
VectorXd init_W(const double *model, int w_len);
void fill_model_with_W(double *model, VectorXd W, int w_len);

#endif //ML_5AL_LIB_CPP_PROJECT_GROUPE_3_UTILS_H
