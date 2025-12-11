import matplotlib.pyplot as plt
import csv
import seaborn as sns
import os

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)
sns.set_palette("muted")

csv_file = "errors_var.csv"
output_file = "errors_plot_sgd.png"
plot_title = "Variable Learning Rate Gradient Descent"

iters = []
e_in = []

# Load training errors
with open(csv_file) as f:
    reader = csv.reader(f)
    for row in reader:
        iters.append(int(row[0]))
        e_in.append(float(row[1]))

# Plot only the training error
plt.figure(figsize=(12, 8), dpi=150)
plt.plot(iters, e_in, linewidth=3, color='#1f77b4', label="Training Error")

plt.xlabel("Iteration", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.title(f"Error vs Iterations ({plot_title})", fontsize=20, pad=20)
plt.legend(fontsize=14, loc="upper right", frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()