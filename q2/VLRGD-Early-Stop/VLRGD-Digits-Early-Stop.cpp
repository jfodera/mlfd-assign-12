//Halting early at cycle 5675. Optimal check discrepancy = 0.0193475
//Final Test Error: 0.0166704


#include <random>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <limits>
#include <numeric>
#include "neuralnet.h"

void load_data(const std::string& filename, Eigen::MatrixXd& X, Eigen::MatrixXd& y) {
    std::ifstream fin(filename);
    if (!fin.is_open()) throw std::runtime_error("Could not open file: " + filename);

    std::vector<Eigen::Vector2d> X_vec;
    std::vector<double> y_vec;
    std::string line;
    double label, f1, f2;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        ss >> label >> f1 >> f2;
        X_vec.emplace_back(f1, f2);
        y_vec.push_back(label == 1.0 ? 1.0 : -1.0);
    }
    fin.close();

    size_t N = X_vec.size();
    X.resize(N, 2);
    y.resize(N, 1);
    for (size_t i = 0; i < N; ++i) {
        X.row(i) = X_vec[i];
        y(i) = y_vec[i];
    }
}

void initialize_random_weights(Module& module) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.01, 0.01);
    Eigen::MatrixXd& w = const_cast<Eigen::MatrixXd&>(module.weights());
    Eigen::VectorXd& b = const_cast<Eigen::VectorXd&>(module.biases());
    for (int i = 0; i < w.rows(); ++i) {
        b(i) = dis(gen);
        for (int j = 0; j < w.cols(); ++j) w(i, j) = dis(gen);
    }
}

Eigen::MatrixXd identity_func(const Eigen::MatrixXd& z) { return z; }
Eigen::MatrixXd identity_derivative(const Eigen::MatrixXd& z) { return Eigen::MatrixXd::Ones(z.rows(), z.cols()); }
Eigen::MatrixXd sign_func(const Eigen::MatrixXd& z) { return z.unaryExpr([](double v){ return v >= 0.0 ? 1.0 : -1.0; }); }

int main() {
    Eigen::MatrixXd X_all, y_all, X_test, y_test;
    load_data("ZipDigitsRandom.train.txt", X_all, y_all);
    load_data("ZipDigitsRandom.test.txt", X_test, y_test);

    std::vector<int> idx(X_all.rows());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device()()));

    const int train_sz = 250;
    const int val_sz = 50;

    Eigen::MatrixXd X_train(train_sz, 2), y_train(train_sz, 1);
    Eigen::MatrixXd X_val(val_sz, 2), y_val(val_sz, 1);
    for (int i = 0; i < train_sz; ++i) {
        int k = idx[i];
        X_train.row(i) = X_all.row(k);
        y_train(i) = y_all(k);
    }
    for (int i = 0; i < val_sz; ++i) {
        int k = idx[train_sz + i];
        X_val.row(i) = X_all.row(k);
        y_val(i) = y_all(k);
    }

    Module hidden(2, 10);
    initialize_random_weights(hidden);
    Module output(10, 1);
    initialize_random_weights(output);

    Model network({hidden, output});

    double eta0 = 0.01;
    double alpha = 1.05;
    double beta = 0.7;
    int max_iters = 2000000;
    double reg_lambda = 0.0;
    double tol = 1e-6;
    int patience = 100;
    std::vector<double> errors, val_errors;

    network.dynamic_rate_optimization(
        X_train, y_train,
        errors, val_errors,
        reg_lambda,
        eta0, alpha, beta,
        max_iters,
        tol, patience,
        identity_func,
        identity_derivative,
        X_val, y_val
    );

    std::ofstream err("errors_var_es.csv");
    for (size_t i = 0; i < errors.size(); ++i) err << i << "," << errors[i] << "\n";
    err.close();

    std::ofstream verr("val_errors_var_es.csv");
    for (size_t i = 0; i < val_errors.size(); ++i) verr << i << "," << val_errors[i] << "\n";
    verr.close();

    double x1_min = X_train.col(0).minCoeff();
    double x1_max = X_train.col(0).maxCoeff();
    double x2_min = X_train.col(1).minCoeff();
    double x2_max = X_train.col(1).maxCoeff();
    double mx1 = (x1_max - x1_min) * 0.05;
    double mx2 = (x2_max - x2_min) * 0.05;
    x1_min -= mx1; x1_max += mx1;
    x2_min -= mx2; x2_max += mx2;

    const int steps = 200;
    Eigen::MatrixXd grid(steps * steps, 2);
    int pos = 0;
    for (int i = 0; i < steps; ++i) {
        double x1 = x1_min + i * (x1_max - x1_min) / (steps - 1.0);
        for (int j = 0; j < steps; ++j) {
            double x2 = x2_min + j * (x2_max - x2_min) / (steps - 1.0);
            grid(pos, 0) = x1;
            grid(pos, 1) = x2;
            ++pos;
        }
    }

    Eigen::MatrixXd preds = network.forward(grid, sign_func);

    std::ofstream gout("grid_predictions_var_es.csv");
    pos = 0;
    for (int i = 0; i < steps; ++i)
        for (int j = 0; j < steps; ++j)
            gout << grid(pos, 0) << "," << grid(pos, 1) << "," << preds(pos++, 0) << "\n";
    gout.close();

    double test_error = network.appraise_evaluation(X_test, y_test, sign_func);
    std::cout << "Final Test Error: " << test_error << std::endl;

    return 0;
}