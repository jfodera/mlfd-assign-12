#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iomanip>
#include <vector>
#include "neuralnet.h"

Eigen::MatrixXd merge_params(const Eigen::MatrixXd& param_set, const Eigen::VectorXd& offset_set) {
    size_t unit_count = param_set.rows();
    size_t entry_count = param_set.cols();
    Eigen::MatrixXd combined(unit_count, entry_count + 1);

    for (size_t unit_idx = 0; unit_idx < unit_count; ++unit_idx) {
        combined(unit_idx, 0) = offset_set(unit_idx);
        combined.row(unit_idx).segment(1, entry_count) = param_set.row(unit_idx);
    }

    return combined.transpose();
}

Eigen::VectorXd hyp_tan_act(const Eigen::VectorXd& z) { return z.array().tanh(); }
Eigen::VectorXd hyp_tan_deriv(const Eigen::VectorXd& z) { return (1.0 - z.array().square()).matrix(); }
Eigen::VectorXd lin_act(const Eigen::VectorXd& z) { return z; }
Eigen::VectorXd lin_deriv(const Eigen::VectorXd& z) { return Eigen::VectorXd::Ones(z.size()); }

void run_test(Model& sys, const std::string& title, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> act, std::function<Eigen::VectorXd(const Eigen::VectorXd&)> deriv) {
    std::cout << title << "\n\n";

    double e_in = 0.0;
    std::vector<Eigen::MatrixXd> w_grads;
    std::vector<Eigen::VectorXd> b_grads;

    sys.execute_optimization({Eigen::VectorXd{{2, 1}}}, {Eigen::VectorXd{{-1}}}, e_in, w_grads, b_grads, 0.0, act, deriv);

    std::cout << "E_in: " << std::fixed << std::setprecision(6) << e_in << "\n\n";

    for (size_t l = 0; l < w_grads.size(); ++l) {
        Eigen::MatrixXd grad = merge_params(w_grads[l], b_grads[l]);
        std::cout << "Layer " << (l + 1) << ":\n";
        std::cout << std::fixed << std::setprecision(6) << grad << "\n\n";
    }
    std::cout << "\n";
}

int main() {
    Module hidden(2, 2);
    Module output(2, 1);
    Model net({hidden, output});

    std::vector<Eigen::VectorXd> x(1), y(1);
    x[0] << 2, 1;
    y[0] << -1;

    run_test(net, "Tanh Activation", hyp_tan_act, hyp_tan_deriv);
    run_test(net, "Identity Activation", lin_act, lin_deriv);

    std::vector<Module> layers = net.retrieve_components();
    for (Module& layer : layers) {
        layer.assign_params(Eigen::MatrixXd::Constant(layer.retrieve_params().rows(), layer.retrieve_params().cols(), 0.1500001));
        layer.assign_offsets(Eigen::VectorXd::Constant(layer.retrieve_offsets().size(), 0.1500001));
    }
    net.configure_components(layers);

    run_test(net, "Tanh Activation (perturbed)", hyp_tan_act, hyp_tan_deriv);
    run_test(net, "Identity Activation (perturbed)", lin_act, lin_deriv);

    return 0;
}