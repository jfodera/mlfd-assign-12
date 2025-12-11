#ifndef NEURALNET_H
#define NEURALNET_H

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <limits>

// ===================================================================
// Module class
// ===================================================================

class Module {
private:
    Eigen::MatrixXd weights_;  // output_dim x input_dim
    Eigen::VectorXd biases_;   // output_dim
    Eigen::MatrixXd inputs_;   // N x input_dim (batch)
    Eigen::MatrixXd outputs_;  // N x output_dim (batch)
    Eigen::MatrixXd deltas_;   // N x output_dim (batch)

public:
    Module(size_t input_dim, size_t output_dim)
        : weights_(Eigen::MatrixXd::Constant(output_dim, input_dim, 0.15)),
          biases_(Eigen::VectorXd::Constant(output_dim, 0.15)) {}

    const Eigen::MatrixXd& weights() const { return weights_; }
    const Eigen::VectorXd& biases() const { return biases_; }
    const Eigen::MatrixXd& outputs() const { return outputs_; }
    const Eigen::MatrixXd& deltas() const { return deltas_; }

    void set_weights(const Eigen::MatrixXd& new_weights) { weights_ = new_weights; }
    void set_biases(const Eigen::VectorXd& new_biases) { biases_ = new_biases; }

    // Batch forward (inputs: N x input_dim)
    Eigen::MatrixXd forward(const Eigen::MatrixXd& inputs,
                            std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> activation = nullptr) {
        inputs_ = inputs;
        size_t N = inputs.rows();
        Eigen::MatrixXd z = inputs * weights_.transpose() + biases_.transpose().replicate(N, 1);
        if (activation) {
            outputs_ = activation(z);
        } else {
            outputs_ = z.array().tanh();
        }
        return outputs_;
    }

    // Batch backward for hidden layers
    Eigen::MatrixXd backward(const Eigen::MatrixXd& next_weights, const Eigen::MatrixXd& next_deltas) {
        Eigen::MatrixXd deriv = (1.0 - outputs_.array().square()).matrix();
        deltas_ = deriv.cwiseProduct(next_deltas * next_weights);
        return deltas_;
    }

    // Batch backward for output layer
    Eigen::MatrixXd last_backward(const Eigen::MatrixXd& outputs, const Eigen::MatrixXd& targets,
                                  std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> deriv_func) {
        Eigen::MatrixXd deriv = deriv_func(outputs);
        deltas_ = (0.5 * (outputs - targets)).cwiseProduct(deriv);
        return deltas_;
    }

    // Compute weight gradient (prev_outputs: N x prev_dim) -> output_dim x prev_dim
    Eigen::MatrixXd compute_grad_weights(const Eigen::MatrixXd& prev_outputs) const {
        return (deltas_.transpose() * prev_outputs) / static_cast<double>(prev_outputs.rows());
    }

    // Compute bias gradient -> output_dim
    Eigen::VectorXd compute_grad_biases() const {
        return deltas_.colwise().mean();
    }

    // Update parameters
    void update(const Eigen::MatrixXd& grad_w, const Eigen::VectorXd& grad_b, double rate) {
        weights_ -= rate * grad_w;
        biases_ -= rate * grad_b;
    }
};

// ===================================================================
// Model class
// ===================================================================

class Model {
private:
    std::vector<Module> layers_;

public:
    Model(const std::vector<Module>& layers) : layers_(layers) {}

    const std::vector<Module>& layers() const { return layers_; }

    // Batch forward (X: N x input_dim) -> N x output_dim
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X,
                            std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> last_activation = nullptr) {
        Eigen::MatrixXd out = X;
        for (size_t l = 0; l < layers_.size() - 1; ++l) {
            out = layers_[l].forward(out);
        }
        return layers_.back().forward(out, last_activation);
    }

    // Batch backward
    void backward(const Eigen::MatrixXd& outputs, const Eigen::MatrixXd& targets,
                  std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> last_deriv) {
        Eigen::MatrixXd next_delta = layers_.back().last_backward(outputs, targets, last_deriv);
        for (int l = static_cast<int>(layers_.size()) - 2; l >= 0; --l) {
            next_delta = layers_[l].backward(layers_[l + 1].weights(), next_delta);
        }
    }

    // Batch error
    double compute_error(const Eigen::MatrixXd& outputs, const Eigen::MatrixXd& targets, double reg_lambda) {
        size_t N = outputs.rows();
        size_t dim = outputs.cols();
        Eigen::MatrixXd diff = outputs - targets;
        double e_in = diff.squaredNorm() / (4.0 * N);
        double reg = 0.0;
        if (reg_lambda > 0.0) {
            for (const auto& layer : layers_) {
                reg += layer.weights().squaredNorm();
            }
            reg *= (reg_lambda / (N * dim));
        }
        return e_in + reg;
    }

    // Batch gradient descent
    void gradient_descent(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y,
                          std::vector<Eigen::MatrixXd>& grads_w, std::vector<Eigen::VectorXd>& grads_b,
                          double& e_in, double reg_lambda,
                          std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> last_activation,
                          std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> last_deriv) {
        size_t num_layers = layers_.size();
        size_t N = X.rows();

        Eigen::MatrixXd outputs = forward(X, last_activation);
        e_in = compute_error(outputs, y, reg_lambda);
        backward(outputs, y, last_deriv);

        grads_w.resize(num_layers);
        grads_b.resize(num_layers);
        for (size_t l = 0; l < num_layers; ++l) {
            const Eigen::MatrixXd& prev_out = (l == 0) ? X : layers_[l - 1].outputs();
            grads_w[l] = layers_[l].compute_grad_weights(prev_out);
            grads_b[l] = layers_[l].compute_grad_biases();
        }

        if (reg_lambda > 0.0) {
            for (size_t l = 0; l < num_layers; ++l) {
                grads_w[l] += (2.0 * reg_lambda / N) * layers_[l].weights();
            }
        }
    }

    // Variable LR GD (same as before, but using batch GD)
    void dynamic_rate_optimization(
        const Eigen::MatrixXd& X, const Eigen::MatrixXd& y,
        std::vector<double>& errors, std::vector<double>& val_errors,
        double penalty_coeff = 0.0,
        double rate0 = 0.01, double grow_rate = 1.05, double shrink_rate = 0.7,
        int max_cycles = 2000000, double precision = 0.0, int endurance = 0,
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> terminal_activation = nullptr,
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> terminal_deactivation = nullptr,
        const Eigen::MatrixXd& check_X = Eigen::MatrixXd(),
        const Eigen::MatrixXd& check_y = Eigen::MatrixXd()
    ) {
        bool employ_check = check_X.rows() > 0;
        size_t comp_count = layers_.size();
        double rate = rate0;

        std::vector<Eigen::MatrixXd> param_grads(comp_count);
        std::vector<Eigen::VectorXd> offset_grads(comp_count);

        std::vector<Eigen::MatrixXd> optimal_params(comp_count);
        std::vector<Eigen::VectorXd> optimal_offsets(comp_count);
        double optimal_check_discrep = std::numeric_limits<double>::max();
        int endurance_tracker = 0;

        for (size_t i = 0; i < comp_count; ++i) {
            optimal_params[i] = layers_[i].weights();
            optimal_offsets[i] = layers_[i].biases();
        }

        int cycle_count = 0;
        while (cycle_count < max_cycles) {
            if (cycle_count % 10000 == 0) {
                std::cout << "Completed cycle " << cycle_count << std::endl;
            }

            std::vector<Eigen::MatrixXd> prior_params(comp_count);
            std::vector<Eigen::VectorXd> prior_offsets(comp_count);
            for (size_t i = 0; i < comp_count; ++i) {
                prior_params[i] = layers_[i].weights();
                prior_offsets[i] = layers_[i].biases();
            }

            double discrep;
            gradient_descent(X, y, param_grads, offset_grads, discrep, penalty_coeff, terminal_activation, terminal_deactivation);
            errors.push_back(discrep);

            for (size_t i = 0; i < comp_count; ++i) {
                layers_[i].update(param_grads[i], offset_grads[i], rate);
            }

            double fresh_discrep;
            gradient_descent(X, y, param_grads, offset_grads, fresh_discrep, penalty_coeff, terminal_activation, terminal_deactivation);

            if (fresh_discrep < discrep) {
                rate *= grow_rate;
            } else {
                for (size_t i = 0; i < comp_count; ++i) {
                    layers_[i].set_weights(prior_params[i]);
                    layers_[i].set_biases(prior_offsets[i]);
                }
                rate *= shrink_rate;
            }

            if (employ_check) {
                Eigen::MatrixXd check_outputs = forward(check_X, terminal_activation);
                double check_discrep = compute_error(check_outputs, check_y, penalty_coeff);
                val_errors.push_back(check_discrep);

                if (check_discrep < optimal_check_discrep - precision) {
                    optimal_check_discrep = check_discrep;
                    endurance_tracker = 0;
                    for (size_t i = 0; i < comp_count; ++i) {
                        optimal_params[i] = layers_[i].weights();
                        optimal_offsets[i] = layers_[i].biases();
                    }
                } else {
                    ++endurance_tracker;
                    if (endurance > 0 && endurance_tracker >= endurance) {
                        std::cout << "Halting early at cycle " << cycle_count
                                  << ". Optimal check discrepancy = " << optimal_check_discrep << std::endl;
                        break;
                    }
                }
            }

            ++cycle_count;
        }

        if (employ_check) {
            for (size_t i = 0; i < comp_count; ++i) {
                layers_[i].set_weights(optimal_params[i]);
                layers_[i].set_biases(optimal_offsets[i]);
            }
        }
    }

    // Batch test error
    double appraise_evaluation(const Eigen::MatrixXd& eval_X, const Eigen::MatrixXd& eval_y,
                               std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> terminal_activation = nullptr) {
        Eigen::MatrixXd outputs = forward(eval_X, terminal_activation);
        double accum = (outputs - eval_y).squaredNorm();
        return accum / (4.0 * eval_X.rows());
    }
};

#endif // NEURALNET_H