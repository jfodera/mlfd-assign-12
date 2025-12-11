//Final Test Error (with sign): 0.0233333
#include <random>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <limits>
#include "neuralnet.h"

void load_data(const std::string& filename, Eigen::MatrixXd& X, Eigen::MatrixXd& y) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<Eigen::VectorXd> X_vec, y_vec;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        double label, f1, f2;
        ss >> label >> f1 >> f2;

        Eigen::VectorXd x(2);
        x << f1, f2;
        X_vec.push_back(x);

        Eigen::VectorXd t(1);
        t(0) = (label == 1.0) ? 1.0 : -1.0;
        y_vec.push_back(t);
    }
    fin.close();

    size_t N = X_vec.size();
    X.resize(N, 2);
    y.resize(N, 1);
    for (size_t i = 0; i < N; ++i) {
        X.row(i) = X_vec[i];
        y.row(i) = y_vec[i];
    }
}

void initialize_random_weights(Module& module, double min_val = -0.01, double max_val = 0.01) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min_val, max_val);

    Eigen::MatrixXd& w = const_cast<Eigen::MatrixXd&>(module.weights());
    Eigen::VectorXd& b = const_cast<Eigen::VectorXd&>(module.biases());

    for (int i = 0; i < w.rows(); ++i) {
        b(i) = dis(gen);
        for (int j = 0; j < w.cols(); ++j) {
            w(i, j) = dis(gen);
        }
    }
}

Eigen::MatrixXd identity_func(const Eigen::MatrixXd& z) { return z; }
Eigen::MatrixXd identity_derivative(const Eigen::MatrixXd& z) { return Eigen::MatrixXd::Ones(z.rows(), z.cols()); }

Eigen::MatrixXd sign_func(const Eigen::MatrixXd& z) {
    return z.unaryExpr([](double v) { return (v >= 0.0 ? 1.0 : -1.0); });
}

int main() {
    Eigen::MatrixXd X_train, y_train;
    Eigen::MatrixXd X_test, y_test;

    load_data("ZipDigitsRandom.train.txt", X_train, y_train);
    load_data("ZipDigitsRandom.train.txt", X_test, y_test);

    Module hidden(2, 10);
    initialize_random_weights(hidden);

    Module output(10, 1);
    initialize_random_weights(output);

    std::vector<Module> components = {hidden, output};
    Model network(components);

    size_t N = X_train.rows();
    double reg_lambda = 0.01 / N;
    double eta0 = 0.01;
    double alpha = 1.05;
    double beta = 0.7;
    int max_iters = 2000000;
    std::vector<double> errors;
    std::vector<double> val_errors;
    Eigen::MatrixXd X_val;
    Eigen::MatrixXd y_val;

    network.dynamic_rate_optimization(
        X_train, y_train,
        errors, val_errors,
        reg_lambda,
        eta0, alpha, beta,
        max_iters,
        0.0, 0,
        identity_func,
        identity_derivative,
        X_val, y_val
    );

    std::ofstream fout("errors_var_wd.csv");
    for (size_t i = 0; i < errors.size(); ++i) {
        fout << i << "," << errors[i] << "\n";
    }
    fout.close();

    double x1_min = X_train.col(0).minCoeff();
    double x1_max = X_train.col(0).maxCoeff();
    double x2_min = X_train.col(1).minCoeff();
    double x2_max = X_train.col(1).maxCoeff();

    double margin_x1 = (x1_max - x1_min) * 0.05;
    double margin_x2 = (x2_max - x2_min) * 0.05;
    x1_min -= margin_x1; x1_max += margin_x1;
    x2_min -= margin_x2; x2_max += margin_x2;

    const int steps = 200;
    Eigen::MatrixXd grid(steps * steps, 2);
    int idx = 0;
    for (int i = 0; i < steps; ++i) {
        double x1 = x1_min + i * (x1_max - x1_min) / (steps - 1.0);
        for (int j = 0; j < steps; ++j) {
            double x2 = x2_min + j * (x2_max - x2_min) / (steps - 1.0);
            grid(idx, 0) = x1;
            grid(idx, 1) = x2;
            ++idx;
        }
    }

    Eigen::MatrixXd grid_outputs = network.forward(grid, sign_func);

    std::ofstream grid_out("grid_predictions_var_wd.csv");
    idx = 0;
    for (int i = 0; i < steps; ++i) {
        for (int j = 0; j < steps; ++j) {
            grid_out << grid(idx, 0) << "," << grid(idx, 1) << "," << grid_outputs(idx, 0) << "\n";
            ++idx;
        }
    }
    grid_out.close();

    double test_error = network.appraise_evaluation(X_test, y_test, sign_func);
    std::cout << "Final Test Error (with sign): " << test_error << std::endl;

    return 0;
}