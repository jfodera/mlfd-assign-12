import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.3)
sns.set_palette("deep")

grid_file = "grid_predictions_var_es.csv"
train_file = "ZipDigitsRandom.train.txt"
output_file = "decision_boundary_styled.png"
plot_title = "Variable Rate with Early Stopping"

train_data = np.loadtxt(train_file)
X_train = train_data[:, 1:3]
y_train = train_data[:, 0]

grid_data = np.loadtxt(grid_file, delimiter=',')
X_grid = grid_data[:, :2]
Z_grid = grid_data[:, 2]

x1_vals = np.unique(X_grid[:, 0])
x2_vals = np.unique(X_grid[:, 1])
Z = Z_grid.reshape(len(x1_vals), len(x2_vals)).T
X1, X2 = np.meshgrid(x1_vals, x2_vals)

plt.figure(figsize=(10, 8), dpi=150)

plt.contourf(X1, X2, Z, levels=[Z.min(), 0, Z.max()], 
             colors=['#ff9999', '#9999ff'], alpha=0.4, zorder=1)

plt.contour(X1, X2, Z, levels=[0], colors='red', linewidths=4, linestyles='--', zorder=2)

plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            c='blue', marker='o', s=100,
            edgecolors='white', linewidth=1.5, alpha=0.9, zorder=3)

plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1],
            c='green', marker='^', s=100,
            edgecolors='white', linewidth=1.5, alpha=0.9, zorder=3)

plt.xlabel("Symmetry", fontsize=14)
plt.ylabel("Intensity", fontsize=14)
plt.title(f"Decision Boundary of Neural Network\n({plot_title})", fontsize=18, pad=20)

plt.legend(fontsize=12, loc='upper left', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"Styled plot saved to {output_file}")