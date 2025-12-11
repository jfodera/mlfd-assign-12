// Final Test Error: 0.0138577

#include <random>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <limits>
#include "neuralnet.h"

void load_data(const std::string& filename, Eigen::MatrixXd& X, Eigen::MatrixXd& y) {
    std::ifstream fin(filename);
    if (!fin.is_open()) throw std::runtime_error("File not found: " + filename);

    std::vector<Eigen::Vector2d> Xv;
    std::vector<double> yv;
    std::string line;
    double label, f1, f2;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        ss >> label >> f1 >> f2;
        Xv.emplace_back(f1, f2);
        yv.push_back(label == 1.0 ? 1.0 : -1.0);
    }
    fin.close();

    size_t N = Xv.size();
    X.resize(N, 2);
    y.resize(N, 1);
    for (size_t i = 0; i < N; ++i) {
        X.row(i) = Xv[i];
        y(i, 0) = yv[i];
    }
}

void initialize_random_weights(Module& m, double scale = 0.01) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(-scale, scale);
    Eigen::MatrixXd& w = const_cast<Eigen::MatrixXd&>(m.weights());
    Eigen::VectorXd& b = const_cast<Eigen::VectorXd&>(m.biases());
    for (int i = 0; i < w.rows(); ++i) {
        b(i) = dis(gen);
        for (int j = 0; j < w.cols(); ++j) w(i, j) = dis(gen);
    }
}

Eigen::MatrixXd identity(const Eigen::MatrixXd& z) { return z; }
Eigen::MatrixXd identity_deriv(const Eigen::MatrixXd& z) { return Eigen::MatrixXd::Ones(z.rows(), z.cols()); }

int main() {
    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    load_data("ZipDigitsRandom.train.txt", X_train, y_train);
    load_data("ZipDigitsRandom.test.txt", X_test, y_test);

    std::cout << "Training samples: " << X_train.rows() << "\nTesting samples: " << X_test.rows() << std::endl;

    Module hidden(2, 10);
    Module output(10, 1);
    initialize_random_weights(hidden);
    initialize_random_weights(output);

    Model net({hidden, output});

    std::vector<double> errors, val_errors;
    net.random_sample_optimization(
        X_train, y_train, errors, val_errors,
        0.0, 0.01, 0.0, 0, 20000000,
        identity, identity_deriv
    );

    std::cout << "Finished training.\n";

    std::ofstream err("errors_sgd.csv");
    for (size_t i = 0; i < errors.size(); ++i) err << i << "," << errors[i] << "\n";
    err.close();
    std::cout << "Training errors saved to: errors_sgd.csv\n";

    const int steps = 200;
    double x1_min = X_train.col(0).minCoeff();
    double x1_max = X_train.col(0).maxCoeff();
    double x2_min = X_train.col(1).minCoeff();
    double x2_max = X_train.col(1).maxCoeff();
    double m1 = (x1_max - x1_min) * 0.05;
    double m2 = (x2_max - x2_min) * 0.05;
    x1_min -= m1; x1_max += m1;
    x2_min -= m2; x2_max += m2;

    Eigen::MatrixXd grid(steps * steps, 2);
    size_t idx = 0;
    for (int i = 0; i < steps; ++i) {
        double x1 = x1_min + i * (x1_max - x1_min) / (steps - 1.0);
        for (int j = 0; j < steps; ++j) {
            double x2 = x2_min + j * (x2_max - x2_min) / (steps - 1.0);
            grid(idx, 0) = x1;
            grid(idx++, 1) = x2;
        }
    }

    Eigen::MatrixXd preds = net.forward(grid, identity);

    std::ofstream grid_out("grid_predictions_SGD.csv");
    idx = 0;
    for (int i = 0; i < steps; ++i)
        for (int j = 0; j < steps; ++j)
            grid_out << grid(idx, 0) << "," << grid(idx, 1) << "," << preds(idx++, 0) << "\n";
    grid_out.close();
    std::cout << "Grid predictions saved to: grid_predictions_SGD.csv\n";

    double test_err = net.appraise_evaluation(X_test, y_test, identity);
    std::cout << "Final Test Error: " << test_err << std::endl;

    return 0;
}