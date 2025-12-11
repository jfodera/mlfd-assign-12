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

class Module {
private:
    Eigen::MatrixXd weights_;
    Eigen::VectorXd biases_;
    Eigen::MatrixXd inputs_;
    Eigen::MatrixXd outputs_;
    Eigen::MatrixXd deltas_;

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

    Eigen::MatrixXd backward(const Eigen::MatrixXd& next_weights, const Eigen::MatrixXd& next_deltas) {
        Eigen::MatrixXd deriv = (1.0 - outputs_.array().square()).matrix();
        deltas_ = deriv.cwiseProduct(next_deltas * next_weights);
        return deltas_;
    }

    Eigen::MatrixXd last_backward(const Eigen::MatrixXd& outputs, const Eigen::MatrixXd& targets,
                                  std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> deriv_func) {
        Eigen::MatrixXd deriv = deriv_func(outputs);
        deltas_ = (0.5 * (outputs - targets)).cwiseProduct(deriv);
        return deltas_;
    }

    Eigen::MatrixXd compute_grad_weights(const Eigen::MatrixXd& prev_outputs) const {
        return (deltas_.transpose() * prev_outputs) / static_cast<double>(prev_outputs.rows());
    }

    Eigen::VectorXd compute_grad_biases() const {
        return deltas_.colwise().mean();
    }

    void update(const Eigen::MatrixXd& grad_w, const Eigen::VectorXd& grad_b, double rate) {
        weights_ -= rate * grad_w;
        biases_ -= rate * grad_b;
    }
};

class Model {
private:
    std::vector<Module> layers_;

public:
    Model(const std::vector<Module>& layers) : layers_(layers) {}

    const std::vector<Module>& layers() const { return layers_; }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X,
                            std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> last_activation = nullptr) {
        Eigen::MatrixXd out = X;
        for (size_t l = 0; l < layers_.size() - 1; ++l) {
            out = layers_[l].forward(out);
        }
        return layers_.back().forward(out, last_activation);
    }

    void backward(const Eigen::MatrixXd& outputs, const Eigen::MatrixXd& targets,
                  std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> last_deriv) {
        Eigen::MatrixXd next_delta = layers_.back().last_backward(outputs, targets, last_deriv);
        for (int l = static_cast<int>(layers_.size()) - 2; l >= 0; --l) {
            next_delta = layers_[l].backward(layers_[l + 1].weights(), next_delta);
        }
    }

    double compute_error(const Eigen::MatrixXd& outputs, const Eigen::MatrixXd& targets, double reg_lambda) {
        size_t N = outputs.rows();
        Eigen::MatrixXd diff = outputs - targets;
        double e_in = diff.squaredNorm() / (4.0 * N);
        double reg = 0.0;
        if (reg_lambda > 0.0) {
            for (const auto& layer : layers_) {
                reg += layer.weights().squaredNorm();
            }
            reg *= (reg_lambda / N);
        }
        return e_in + reg;
    }

    void random_sample_optimization(
        const Eigen::MatrixXd& X, const Eigen::MatrixXd& y,
        std::vector<double>& errors, std::vector<double>& val_errors,
        double penalty_coeff = 0.0,
        double learn_rate = 0.01,
        double precision = 1e-6,
        int endurance = 10,
        int max_steps = 1000,
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> terminal_activation = nullptr,
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> terminal_deactivation = nullptr,
        const Eigen::MatrixXd& check_X = Eigen::MatrixXd(),
        const Eigen::MatrixXd& check_y = Eigen::MatrixXd()
    ) {
        bool employ_check = check_X.rows() > 0;
        size_t sample_count = X.rows();

        errors.clear();
        val_errors.clear();

        double best_val = std::numeric_limits<double>::infinity();
        int counter = 0;
        std::vector<Eigen::MatrixXd> best_w(layers_.size());
        std::vector<Eigen::VectorXd> best_b(layers_.size());
        for (size_t l = 0; l < layers_.size(); ++l) {
            best_w[l] = layers_[l].weights();
            best_b[l] = layers_[l].biases();
        }

        int step = 0;
        while (step < max_steps) {
            if (step % 1000000 == 0 && step > 0) {
                std::cout << "Completed step " << step << std::endl;
            }

            size_t idx = std::rand() % sample_count;
            Eigen::MatrixXd x_batch = X.row(idx);
            Eigen::MatrixXd y_batch = y.row(idx);

            Eigen::MatrixXd output = forward(x_batch, terminal_activation);
            backward(output, y_batch, terminal_deactivation);

            for (size_t l = 0; l < layers_.size(); ++l) {
                const Eigen::MatrixXd& prev_out = (l == 0) ? x_batch : layers_[l - 1].outputs();
                Eigen::MatrixXd grad_w = layers_[l].compute_grad_weights(prev_out);
                Eigen::VectorXd grad_b = layers_[l].compute_grad_biases();

                if (penalty_coeff > 0.0) {
                    grad_w += (2.0 * penalty_coeff / sample_count) * layers_[l].weights();
                }

                layers_[l].update(grad_w, grad_b, learn_rate);
            }

            if (step % sample_count == 0) {
                Eigen::MatrixXd full_out = forward(X, terminal_activation);
                double err = compute_error(full_out, y, penalty_coeff);
                errors.push_back(err);

                if (employ_check) {
                    Eigen::MatrixXd val_out = forward(check_X, terminal_activation);
                    double val_err = compute_error(val_out, check_y, penalty_coeff);
                    val_errors.push_back(val_err);

                    if (val_err < best_val - precision) {
                        best_val = val_err;
                        counter = 0;
                        for (size_t l = 0; l < layers_.size(); ++l) {
                            best_w[l] = layers_[l].weights();
                            best_b[l] = layers_[l].biases();
                        }
                    } else {
                        ++counter;
                        if (counter >= endurance) {
                            std::cout << "Early stopping triggered at step "
                                      << step << ". Minimum Validation Error = "
                                      << best_val << std::endl;
                            for (size_t l = 0; l < layers_.size(); ++l) {
                                layers_[l].set_weights(best_w[l]);
                                layers_[l].set_biases(best_b[l]);
                            }
                            return;
                        }
                    }
                }
            }

            ++step;
        }

        // If early stopping wasn't triggered but validation was used, restore best anyway
        if (employ_check) {
            for (size_t l = 0; l < layers_.size(); ++l) {
                layers_[l].set_weights(best_w[l]);
                layers_[l].set_biases(best_b[l]);
            }
        }
    }

    double appraise_evaluation(const Eigen::MatrixXd& eval_X, const Eigen::MatrixXd& eval_y,
                               std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> terminal_activation = nullptr) {
        Eigen::MatrixXd outputs = forward(eval_X, terminal_activation);
        double accum = (outputs - eval_y).squaredNorm();
        return accum / (4.0 * eval_X.rows());
    }
};

#endif // NEURALNET_H