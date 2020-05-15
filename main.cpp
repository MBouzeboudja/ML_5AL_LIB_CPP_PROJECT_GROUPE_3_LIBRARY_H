#include <cstdio>
#include <fstream>
#include "library.h"
#include "csv-parser/include/csv.hpp"
#include <ctime>
#include <algorithm>
#include <random>
#include <cstdlib>

using namespace csv;

typedef std::vector<std::pair<std::vector<double>, std::vector<double>>> DataVector;

DataVector get_data() {
    CSVReader reader("D:\\Users\\Arthur\\Travail2020\\ML\\dll_repo\\var\\iris.data");
    auto data = DataVector();

    for (auto &row: reader) {
        auto inputs = std::vector<double>();
        std::vector<double> expected;

        inputs.push_back(row["inputA"].get<double>());
        inputs.push_back(row["inputB"].get<double>());
        inputs.push_back(row["inputC"].get<double>());
        inputs.push_back(row["inputD"].get<double>());

        if (row["expected"].get() == "Iris-setosa") {
            expected = {1, -1, -1};
        } else if (row["expected"].get() == "Iris-versicolor") {
            expected = {-1, 1, -1};
        } else {
            expected = {-1, -1, 1};
        }
        data.emplace_back(inputs, expected);
    }

    std::shuffle(data.begin(), data.end(), std::default_random_engine{});

    return data;
}

template <typename T> double sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

double *normalizeOut(double *in, int len) {
    for (int i = 0; i < len; ++i) {
        in[i] = in[i];//]sgn<double>(in[i]);
    }

    return in;
}

int main() {
    // Get dataset
    auto data = get_data();

    // Split dataset
    auto trainDataset = DataVector(data.begin(), data.end() - data.size() / 2);
    auto testDataset = DataVector(data.end() - data.size() / 2, data.end());

    //Normalise inputs and expecteds test dataset
    auto inputsTrainDataset = new std::vector<double>();
    auto outputsTrainDataset = new std::vector<double>();

    for (auto &e : trainDataset) {
        inputsTrainDataset->insert(inputsTrainDataset->end(), e.first.data(), e.first.data() + e.first.size());
        outputsTrainDataset->insert(outputsTrainDataset->end(), e.second.data(), e.second.data() + e.second.size());
    }

    auto params_cnt = trainDataset.begin()->first.size();
    auto outputs_cnt = trainDataset.begin()->second.size();

    // MLP structure
    auto structure = new std::vector<int>;
    structure->push_back(params_cnt);
    structure->push_back(4);
    structure->push_back(outputs_cnt);

    // Taining
    int iterations = 10000;
    double alpha = 0.03;

    NeuralNet *model = create_neural_net(structure->data(), structure->size());


    printf("Training : ");
    neural_net_model_train_classification(
            model,
            inputsTrainDataset->data(),
            trainDataset.size(),
            params_cnt,
            outputsTrainDataset->data(),
            outputs_cnt,
            iterations,
            alpha
    );

    printf("Travail termine.\n");
    printf("Test des jeux de donnees.\n");
    printf("Test du dataset de training\n");
    // Testing with training
    int trainFailures = 0;
    int testFailures = 0;

    for (auto &test : trainDataset) {
        auto out = neural_net_predict_classification(model, test.first.data());
        for (int i = 0; i < test.second.size(); ++i) {
            if (test.second.data()[i] != out[i]) {
                ++trainFailures;
                break;
            }
        }
    }

    printf("Taux de reussite [TRAINING DATA] : %d%%\n", (trainFailures-trainDataset.size()) / trainDataset.size() * 100);
    printf("Test du dataset de test\n");
    for (auto &test : testDataset) {
        auto out = neural_net_predict_classification(model, test.first.data());
        out = normalizeOut(out, outputs_cnt);
        for (int i = 0; i < test.second.size(); ++i) {
            printf("%f ", out[i]);
            if (test.second.data()[i] != out[i]) {
                ++testFailures;
                //break;
            }
        }
        printf("\n");
        for (int i = 0; i < test.second.size(); ++i) {
            printf("%f ", test.second.data()[i]);
        }

        printf("\n\n");
    }

    printf("Taux de reussite [TESTING DATA] : %d%%\n", (testFailures-testDataset.size()) * testDataset.size() * 100);
}
