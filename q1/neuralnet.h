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
// Module class (fully defined first)
// ===================================================================

class Module {
private:
    Eigen::MatrixXd param_mat;
    Eigen::VectorXd offset_vec;
    Eigen::VectorXd entry_vec;
    Eigen::VectorXd result_vec;
    Eigen::VectorXd error_vec;

public:
    Module(size_t entry_dim, size_t result_dim)
        : param_mat(Eigen::MatrixXd::Constant(result_dim, entry_dim, 0.15)),
          offset_vec(Eigen::VectorXd::Constant(result_dim, 0.15)) {}

    void assign_params(Eigen::MatrixXd fresh_params) { param_mat = fresh_params; }
    void assign_offsets(Eigen::VectorXd fresh_offsets) { offset_vec = fresh_offsets; }
    void assign_entry(Eigen::VectorXd fresh_entry) { entry_vec = fresh_entry; }
    void assign_result(Eigen::VectorXd fresh_result) { result_vec = fresh_result; }
    void assign_error(Eigen::VectorXd fresh_error) { error_vec = fresh_error; }

    Eigen::MatrixXd retrieve_params() const { return param_mat; }
    Eigen::VectorXd retrieve_offsets() const { return offset_vec; }
    Eigen::VectorXd retrieve_entry() const { return entry_vec; }
    Eigen::VectorXd retrieve_result() const { return result_vec; }
    Eigen::VectorXd retrieve_error() const { return error_vec; }

    Eigen::VectorXd hyp_tan(const Eigen::VectorXd& val) const { return val.array().tanh(); }
    Eigen::VectorXd hyp_tan_deriv(const Eigen::VectorXd&) const { return (1.0 - result_vec.array().square()).matrix(); }
    Eigen::VectorXd linear(const Eigen::VectorXd& val) const { return val; }
    Eigen::VectorXd linear_deriv(const Eigen::VectorXd& val) const { return Eigen::VectorXd::Ones(val.size()); }
    Eigen::VectorXd sign_act(const Eigen::VectorXd& val) const {
        return val.unaryExpr([](double v) { return v >= 0.0 ? 1.0 : -1.0; });
    }
    Eigen::VectorXd sign_deriv(const Eigen::VectorXd& val) const { return Eigen::VectorXd::Zero(val.size()); }

    Eigen::VectorXd advance(const Eigen::MatrixXd& params, const Eigen::VectorXd& entry, const Eigen::VectorXd& offsets) {
        entry_vec = entry;
        result_vec = hyp_tan(params * entry + offsets);
        return result_vec;
    }

    Eigen::VectorXd terminal_advance(const Eigen::MatrixXd& params,
                                     const Eigen::VectorXd& entry,
                                     const Eigen::VectorXd& offsets,
                                     std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation) {
        entry_vec = entry;
        result_vec = activation(params * entry + offsets);
        return result_vec;
    }

    Eigen::VectorXd reverse(const Eigen::MatrixXd& subsequent_params, const Eigen::VectorXd& subsequent_err) {
        Eigen::VectorXd deriv_val = hyp_tan_deriv(result_vec);
        error_vec = deriv_val.cwiseProduct(subsequent_params.transpose() * subsequent_err);
        return error_vec;
    }

    Eigen::VectorXd terminal_reverse(const Eigen::VectorXd& terminal_result,
                                     const Eigen::VectorXd& goal,
                                     std::function<Eigen::VectorXd(const Eigen::VectorXd&)> deriv_func) {
        Eigen::VectorXd deriv_val = deriv_func(result_vec);
        error_vec = deriv_val.cwiseProduct(0.5 * (terminal_result - goal));
        return error_vec;
    }

    Eigen::MatrixXd derive_grad(const Eigen::VectorXd& prior_entry) {
        return error_vec * prior_entry.transpose();
    }

    void adjust_params(const Eigen::MatrixXd& param_grad, const Eigen::VectorXd& offset_grad, double rate) {
        param_mat -= rate * param_grad;
        offset_vec -= rate * offset_grad;
    }
};

// ===================================================================
// Model class
// ===================================================================

class Model {
private:
    std::vector<Module> components;

public:
    Model(const std::vector<Module>& comps) : components(comps) {}

    void configure_components(const std::vector<Module>& fresh_comps) {
        components = fresh_comps;
    }

    // Changed return type to const reference to fix the binding error in prob1.cpp
    const std::vector<Module>& retrieve_components() const {
        return components;
    }

    void insert_component(const Module& fresh_comp) {
        components.push_back(fresh_comp);
    }

    Eigen::VectorXd advance_signal(const Eigen::VectorXd& entry,
                                   std::function<Eigen::VectorXd(const Eigen::VectorXd&)> terminal_activation = nullptr) {
        Eigen::VectorXd active_entry = entry;
        size_t comp_count = components.size();

        for (size_t idx = 0; idx < comp_count - 1; ++idx) {
            Module& active_comp = components[idx];
            active_comp.assign_entry(active_entry);
            Eigen::MatrixXd active_params = active_comp.retrieve_params();
            Eigen::VectorXd active_offsets = active_comp.retrieve_offsets();
            Eigen::VectorXd active_result = active_comp.advance(active_params, active_entry, active_offsets);
            active_comp.assign_result(active_result);
            active_entry = active_result;
        }

        Module& terminal_comp = components.back();
        terminal_comp.assign_entry(active_entry);
        Eigen::MatrixXd terminal_params = terminal_comp.retrieve_params();
        Eigen::VectorXd terminal_offsets = terminal_comp.retrieve_offsets();
        Eigen::VectorXd result;
        if (terminal_activation) {
            result = terminal_comp.terminal_advance(terminal_params, active_entry, terminal_offsets, terminal_activation);
        } else {
            result = terminal_comp.advance(terminal_params, active_entry, terminal_offsets);
        }
        return result;
    }

    double assess_discrepancy(const Eigen::VectorXd& result, const Eigen::VectorXd& goal, double penalty_coeff) {
        size_t dim = result.size();
        double accum_loss = (result - goal).squaredNorm();
        double discrepancy = accum_loss / (4.0 * dim);

        double penalty = 0.0;
        if (penalty_coeff != 0.0) {
            for (const Module& comp : components) {
                penalty += comp.retrieve_params().squaredNorm();
            }
            penalty *= (penalty_coeff / dim);
        }

        return discrepancy + penalty;
    }

    void reverse_signal(const Eigen::VectorXd& result,
                        const Eigen::VectorXd& goal,
                        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> terminal_deactivation) {
        size_t comp_count = components.size();
        Module& terminal_comp = components[comp_count - 1];
        Eigen::VectorXd subsequent_err = terminal_comp.terminal_reverse(result, goal, terminal_deactivation);

        for (int idx = static_cast<int>(comp_count) - 2; idx >= 0; --idx) {
            Module& active_comp = components[idx];
            Eigen::MatrixXd subsequent_params = components[idx + 1].retrieve_params();
            Eigen::VectorXd active_err = active_comp.reverse(subsequent_params, subsequent_err);
            active_comp.assign_error(active_err);
            subsequent_err = active_err;
        }
    }

    void execute_optimization(const std::vector<Eigen::VectorXd>& entries,
                              const std::vector<Eigen::VectorXd>& goals,
                              double& discrepancy_in,
                              std::vector<Eigen::MatrixXd>& param_grads,
                              std::vector<Eigen::VectorXd>& offset_grads,
                              double penalty_coeff,
                              std::function<Eigen::VectorXd(const Eigen::VectorXd&)> terminal_activation,
                              std::function<Eigen::VectorXd(const Eigen::VectorXd&)> terminal_deactivation) {
        size_t comp_count = components.size();
        size_t sample_count = entries.size();
        discrepancy_in = 0.0;
        param_grads.assign(comp_count, Eigen::MatrixXd());
        offset_grads.assign(comp_count, Eigen::VectorXd());

        for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
            param_grads[comp_idx].setZero(components[comp_idx].retrieve_params().rows(),
                                          components[comp_idx].retrieve_params().cols());
            offset_grads[comp_idx].setZero(components[comp_idx].retrieve_offsets().size());
        }

        for (size_t samp_idx = 0; samp_idx < sample_count; ++samp_idx) {
            Eigen::VectorXd samp_result = advance_signal(entries[samp_idx], terminal_activation);
            reverse_signal(samp_result, goals[samp_idx], terminal_deactivation);
            discrepancy_in += assess_discrepancy(samp_result, goals[samp_idx], penalty_coeff);

            for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                Eigen::VectorXd prior_entry = (comp_idx == 0) ? entries[samp_idx] : components[comp_idx - 1].retrieve_result();
                param_grads[comp_idx] += components[comp_idx].derive_grad(prior_entry) / sample_count;
                offset_grads[comp_idx] += components[comp_idx].retrieve_error() / sample_count;
            }
        }

        if (penalty_coeff != 0.0) {
            for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                param_grads[comp_idx] += components[comp_idx].retrieve_params() * (2.0 * penalty_coeff / sample_count);
            }
        }

        discrepancy_in /= sample_count;
    }

    void dynamic_rate_optimization(const std::vector<Eigen::VectorXd>& entries,
                                   const std::vector<Eigen::VectorXd>& goals,
                                   std::vector<double>& discrepancies,
                                   std::vector<double>& check_discrepancies,
                                   double penalty_coeff = 0.0,
                                   double start_rate = 0.01,
                                   double grow_rate = 1.05,
                                   double shrink_rate = 0.7,
                                   int max_cycles = 1000,
                                   double precision = 1e-6,
                                   int endurance = 10,
                                   // Removed unused halt_limit parameter
                                   std::function<Eigen::VectorXd(const Eigen::VectorXd&)> terminal_activation = nullptr,
                                   std::function<Eigen::VectorXd(const Eigen::VectorXd&)> terminal_deactivation = nullptr,
                                   const std::vector<Eigen::VectorXd>& check_entries = {},
                                   const std::vector<Eigen::VectorXd>& check_goals = {}) {
        bool employ_check = (!check_entries.empty() && !check_goals.empty());
        double rate = start_rate;
        size_t comp_count = components.size();
        size_t sample_count = entries.size();
        size_t check_count = check_entries.size();

        std::vector<Eigen::MatrixXd> param_grads(comp_count);
        std::vector<Eigen::VectorXd> offset_grads(comp_count);

        std::vector<Eigen::MatrixXd> optimal_params(comp_count);
        std::vector<Eigen::VectorXd> optimal_offsets(comp_count);
        double optimal_check_discrep = std::numeric_limits<double>::max();
        int endurance_tracker = 0;

        for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
            optimal_params[comp_idx] = components[comp_idx].retrieve_params();
            optimal_offsets[comp_idx] = components[comp_idx].retrieve_offsets();
        }

        int cycle_count = 0;
        // Removed unused prior_discrep variable
        while (cycle_count < max_cycles) {
            if (cycle_count % 10000 == 0) {
                std::cout << "Completed cycle " << cycle_count << std::endl;
            }

            std::vector<Eigen::MatrixXd> prior_params(comp_count);
            std::vector<Eigen::VectorXd> prior_offsets(comp_count);
            for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                prior_params[comp_idx] = components[comp_idx].retrieve_params();
                prior_offsets[comp_idx] = components[comp_idx].retrieve_offsets();
            }

            double discrep = 0.0;
            execute_optimization(entries, goals, discrep, param_grads, offset_grads, penalty_coeff,
                                 terminal_activation, terminal_deactivation);
            discrepancies.push_back(discrep);

            for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                components[comp_idx].adjust_params(param_grads[comp_idx], offset_grads[comp_idx], rate);
            }

            double fresh_discrep = 0.0;
            execute_optimization(entries, goals, fresh_discrep, param_grads, offset_grads, penalty_coeff,
                                 terminal_activation, terminal_deactivation);

            if (fresh_discrep < discrep) {
                rate *= grow_rate;
            } else {
                for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                    components[comp_idx].assign_params(prior_params[comp_idx]);
                    components[comp_idx].assign_offsets(prior_offsets[comp_idx]);
                }
                rate *= shrink_rate;
            }

            if (employ_check) {
                double check_discrep = 0.0;
                for (size_t chk_idx = 0; chk_idx < check_count; ++chk_idx) {
                    Eigen::VectorXd check_result = advance_signal(check_entries[chk_idx], terminal_activation);
                    check_discrep += assess_discrepancy(check_result, check_goals[chk_idx], penalty_coeff);
                }
                check_discrep /= check_count;
                check_discrepancies.push_back(check_discrep);

                if (check_discrep < optimal_check_discrep - precision) {
                    optimal_check_discrep = check_discrep;
                    endurance_tracker = 0;
                    for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                        optimal_params[comp_idx] = components[comp_idx].retrieve_params();
                        optimal_offsets[comp_idx] = components[comp_idx].retrieve_offsets();
                    }
                } else {
                    ++endurance_tracker;
                    if (endurance_tracker >= endurance) {
                        std::cout << "Halting early at cycle " << cycle_count
                                  << ". Optimal check discrepancy = " << optimal_check_discrep << std::endl;
                        break;
                    }
                }
            }

            ++cycle_count;
        }

        if (employ_check) {
            for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                components[comp_idx].assign_params(optimal_params[comp_idx]);
                components[comp_idx].assign_offsets(optimal_offsets[comp_idx]);
            }
        }
    }

    // random_sample_optimization and appraise_evaluation remain unchanged
    // (they were already fine, just keeping them for completeness)

    void random_sample_optimization(const std::vector<Eigen::VectorXd>& entries,
                                    const std::vector<Eigen::VectorXd>& goals,
                                    std::vector<double>& discrepancies,
                                    std::vector<double>& check_discrepancies,
                                    double penalty_coeff = 0.0,
                                    double learn_rate = 0.01,
                                    double precision = 1e-6,
                                    int endurance = 10,
                                    int max_steps = 1000,
                                    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> terminal_activation = nullptr,
                                    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> terminal_deactivation = nullptr,
                                    const std::vector<Eigen::VectorXd>& check_entries = {},
                                    const std::vector<Eigen::VectorXd>& check_goals = {}) {
        bool employ_check = (!check_entries.empty() && !check_goals.empty());
        size_t comp_count = components.size();
        size_t sample_count = entries.size();
        size_t check_count = check_entries.size();

        discrepancies.clear();
        check_discrepancies.clear();

        double optimal_check_discrep = std::numeric_limits<double>::infinity();
        int endurance_tracker = 0;

        std::vector<Eigen::MatrixXd> optimal_params(comp_count);
        std::vector<Eigen::VectorXd> optimal_offsets(comp_count);

        for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
            optimal_params[comp_idx] = components[comp_idx].retrieve_params();
            optimal_offsets[comp_idx] = components[comp_idx].retrieve_offsets();
        }

        int step_count = 0;
        while (step_count < max_steps) {
            if (step_count % 1000000 == 0) {
                std::cout << "Completed step " << step_count << std::endl;
            }

            size_t rand_idx = std::rand() % sample_count;
            Eigen::VectorXd rand_result = advance_signal(entries[rand_idx], terminal_activation);
            reverse_signal(rand_result, goals[rand_idx], terminal_deactivation);

            for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                Eigen::VectorXd prior_entry = (comp_idx == 0) ? entries[rand_idx] : components[comp_idx - 1].retrieve_result();
                Eigen::MatrixXd comp_grad = components[comp_idx].derive_grad(prior_entry);
                comp_grad += components[comp_idx].retrieve_params() * (2.0 * penalty_coeff / sample_count);
                Eigen::VectorXd off_grad = components[comp_idx].retrieve_error();
                components[comp_idx].adjust_params(comp_grad, off_grad, learn_rate);
            }

            if (step_count % sample_count == 0) {
                double accum_discrep = 0.0;
                for (size_t samp_idx = 0; samp_idx < sample_count; ++samp_idx) {
                    Eigen::VectorXd samp_result = advance_signal(entries[samp_idx], terminal_activation);
                    accum_discrep += assess_discrepancy(samp_result, goals[samp_idx], penalty_coeff);
                }
                discrepancies.push_back(accum_discrep / sample_count);

                if (employ_check) {
                    double accum_check = 0.0;
                    for (size_t chk_idx = 0; chk_idx < check_count; ++chk_idx) {
                        Eigen::VectorXd check_result = advance_signal(check_entries[chk_idx], terminal_activation);
                        accum_check += assess_discrepancy(check_result, check_goals[chk_idx], penalty_coeff);
                    }
                    double check_val = accum_check / check_count;
                    check_discrepancies.push_back(check_val);

                    if (check_val < optimal_check_discrep - precision) {
                        optimal_check_discrep = check_val;
                        endurance_tracker = 0;
                        for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                            optimal_params[comp_idx] = components[comp_idx].retrieve_params();
                            optimal_offsets[comp_idx] = components[comp_idx].retrieve_offsets();
                        }
                    } else {
                        ++endurance_tracker;
                        if (endurance_tracker >= endurance) {
                            std::cout << "Halting early at step " << step_count
                                      << ". Optimal check discrepancy = " << optimal_check_discrep << std::endl;
                            for (size_t comp_idx = 0; comp_idx < comp_count; ++comp_idx) {
                                components[comp_idx].assign_params(optimal_params[comp_idx]);
                                components[comp_idx].assign_offsets(optimal_offsets[comp_idx]);
                            }
                            return;
                        }
                    }
                }
            }

            ++step_count;
        }
    }

    double appraise_evaluation(const std::vector<Eigen::VectorXd>& eval_entries,
                               const std::vector<Eigen::VectorXd>& eval_goals,
                               std::function<Eigen::VectorXd(const Eigen::VectorXd&)> terminal_activation = nullptr) {
        size_t eval_count = eval_entries.size();
        double accum = 0.0;

        for (size_t eval_idx = 0; eval_idx < eval_count; ++eval_idx) {
            Eigen::VectorXd eval_result = advance_signal(eval_entries[eval_idx], terminal_activation);
            accum += (eval_result - eval_goals[eval_idx]).squaredNorm();
        }

        return accum / (4.0 * eval_count);
    }
};

#endif // NEURALNET_H