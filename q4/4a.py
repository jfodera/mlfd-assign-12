import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


class PolynomialKernelDegree8:
    @staticmethod
    def compute(A, B):
        return (1 + A @ B.T) ** 8


class DualSVMTrainer:
    def __init__(self, regularizer):
        self.C = regularizer

    def fit(self, X, y):
        n = X.shape[0]
        K = PolynomialKernelDegree8.compute(X, X)

        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n))
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
        A = matrix(y.reshape(1, -1).astype(float))
        b = matrix([0.0])

        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(sol['x'])

        sv = alphas > 1e-7
        bounded_sv = (alphas > 1e-7) & (alphas < self.C * 0.999)

        if bounded_sv.sum() > 0:
            bias = np.mean(y[bounded_sv] - (alphas * y) @ PolynomialKernelDegree8.compute(X, X[bounded_sv]))
        else:
            bias = np.mean(y[sv] - (alphas * y) @ PolynomialKernelDegree8.compute(X, X[sv]))

        self.alphas = alphas
        self.X_sv = X
        self.y_sv = y
        self.bias = bias
        self.support_idx = np.where(sv)[0]

        return alphas, self.support_idx, bias

    def predict(self, X):
        k = PolynomialKernelDegree8.compute(self.X_sv, X)
        return (self.alphas * self.y_sv) @ k + self.bias


def five_fold_cv_optimized(X, y, candidates, folds=5):
    n = len(y)
    indices = np.random.permutation(n)
    fold_size = n // folds

    best_c = None
    best_cv_err = np.inf
    cv_results = []

    for C in candidates:
        errors = np.zeros(folds)

        for f in range(folds):
            val_start = f * fold_size
            val_end = val_start + fold_size if f < folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.setdiff1d(indices, val_idx)

            trainer = DualSVMTrainer(C)
            trainer.fit(X[train_idx], y[train_idx])

            pred_val = np.sign(trainer.predict(X[val_idx]))
            errors[f] = np.mean(pred_val != y[val_idx])

        mean_err = errors.mean()
        cv_results.append(mean_err)

        if mean_err < best_cv_err:
            best_cv_err = mean_err
            best_c = C

    return best_c, cv_results


def render_boundary(trainer, X, y, C_val, fname):
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 600),
                         np.linspace(y_min, y_max, 600))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = trainer.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7,7))
    plt.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#ffcccc', '#cce5ff'], alpha=1.0)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=3)

    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k', s=35, linewidth=0.8)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Polynomial Kernel SVM (degree 8) - C = {C_val:.2f}")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


if __name__ == "__main__":
    train = np.loadtxt("ZipDigitsRandom.train.txt")
    test  = np.loadtxt("ZipDigitsRandom.test.txt")

    X_train = train[:, 1:3]
    y_train = np.where(train[:,0] == 1, +1, -1)
    X_test  = test[:, 1:3]
    y_test  = np.where(test[:,0] == 1, +1, -1)

    C_small = 0.01
    C_large = 20.0

    trainer_small = DualSVMTrainer(C_small)
    trainer_small.fit(X_train, y_train)
    render_boundary(trainer_small, X_train, y_train, C_small, "boundary_smallC.png")

    trainer_large = DualSVMTrainer(C_large)
    trainer_large.fit(X_train, y_train)
    render_boundary(trainer_large, X_train, y_train, C_large, "boundary_largeC.png")

    print(f"C = {C_small:8.4f} SVs: {len(trainer_small.support_idx):3d}")
    print(f"C = {C_large:8.4f} SVs: {len(trainer_large.support_idx):3d}")

    C_grid = np.logspace(-2, 2, 40)
    best_C, cv_scores = five_fold_cv_optimized(X_train, y_train, C_grid, folds=5)

    print("\nCross-validation results (selected best marked):")
    for c, err in zip(C_grid, cv_scores):
        print(f"  C = {c:8.5f}  CV error = {err:.4f}", "  BEST" if abs(c-best_C)<1e-10 else "")

    final_trainer = DualSVMTrainer(best_C)
    final_trainer.fit(X_train, y_train)
    render_boundary(final_trainer, X_train, y_train, best_C, f"boundary_best_C_{best_C:.5f}.png")

    test_preds = np.sign(final_trainer.predict(X_test))
    test_error = np.mean(test_preds != y_test)

    print(f"\nBest C selected by 5-fold CV : {best_C:.6f}")
    print(f"Support vectors              : {len(final_trainer.support_idx)}")
    print(f"Test error (E_test)          : {test_error:.4f} ({test_error*100:5.2f}%)")