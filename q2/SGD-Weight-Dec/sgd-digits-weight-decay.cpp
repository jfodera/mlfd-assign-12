#include <random>
#include <vector>
#include <string>
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
    std::vector<Eigen::Vector2d> Xv;
    std::vector<double> yv;
    std::string line;
    double l, f1, f2;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        ss >> l >> f1 >> f2;
        Xv.emplace_back(f1, f2);
        yv.push_back(l == 1.0 ? 1.0 : -1.0);
    }
    fin.close();
    size_t N = Xv.size();
    X.resize(N, 2);
    y.resize(N, 1);
    for (size_t i = 0; i < N; ++i) {
        X.row(i) = Xv[i];
        y(i) = yv[i];
    }
}

void initialize_random_weights(Module& m) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(-0.01, 0.01);
    Eigen::MatrixXd& w = const_cast<Eigen::MatrixXd&>(m.weights());
    Eigen::VectorXd& b = const_cast<Eigen::VectorXd&>(m.biases());
    for (int i = 0; i < w.rows(); ++i) {
        b(i) = dis(gen);
        for (int j = 0; j < w.cols(); ++j) w(i, j) = dis(gen);
    }
}

Eigen::MatrixXd identity_func(const Eigen::MatrixXd& z) { return z; }
Eigen::MatrixXd identity_derivative(const Eigen::MatrixXd& z) { return Eigen::MatrixXd::Ones(z.rows(), z.cols()); }
Eigen::MatrixXd sign_func(const Eigen::MatrixXd& z) { return z.unaryExpr([](double v){ return v >= 0.0 ? 1.0 : -1.0; }); }

int main() {
    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    load_data("ZipDigitsRandom.train.txt", X_train, y_train);
    load_data("ZipDigitsRandom.test.txt", X_test, y_test);

    Module hidden(2, 10);
    initialize_random_weights(hidden);
    Module output(10, 1);
    initialize_random_weights(output);

    Model network({hidden, output});

    int N = X_train.rows();
    double reg_lambda = 0.01 / N;
    double eta = 0.01;
    int max_iters = 20000000;
    std::vector<double> errors, val_errors;

    network.random_sample_optimization(
        X_train, y_train,
        errors, val_errors,
        reg_lambda,
        eta,
        0.0,
        0,
        max_iters,
        identity_func,
        identity_derivative
    );

    std::ofstream err("errors_sgd_wd.csv");
    for (size_t i = 0; i < errors.size(); ++i) err << i << "," << errors[i] << "\n";
    err.close();

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

    std::ofstream gout("grid_predictions_sgd_wd.csv");
    pos = 0;
    for (int i = 0; i < steps; ++i)
        for (int j = 0; j < steps; ++j)
            gout << grid(pos, 0) << "," << grid(pos, 1) << "," << preds(pos++, 0) << "\n";
    gout.close();

    double test_error = network.appraise_evaluation(X_test, y_test, sign_func);
    std::cout << "Final Test Error: " << test_error << std::endl;

    return 0;
}