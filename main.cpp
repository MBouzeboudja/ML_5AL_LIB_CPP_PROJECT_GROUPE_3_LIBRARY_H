#include <iostream>
#include "library.h"
#include "utils.h"
#include "Eigen"

int main(){
    double *matrix = create_linear_model(8);

    for (int i=0; i< 8; i++){
        std::cout << matrix[i];
        std::cout << "-";

    }
    //double *matrix1 = create_linear_model(4);
    Eigen::MatrixXd result = fill_X(matrix, 8, 4);
    std::cout << "\nIt work\n";
    std::cout << result;

    Eigen::VectorXd y = fill_Y(matrix, 8);
    std::cout << "\n y -> \n";
    std::cout << y;
}

