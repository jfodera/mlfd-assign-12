import matplotlib.pyplot as plt
import csv
import seaborn as sns
import os

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)
sns.set_palette("muted")

csv_file = "errors_sgd_es.csv"
output_file = "errors_plot_sgd_es.png"
plot_title = "SGD with Early Stopping"
val_csv_file = "val_errors_sgd_es.csv"

iters = []
e_in = []

with open(csv_file) as f:
    reader = csv.reader(f) 
    for row in reader:
        iters.append(int(row[0]))
        e_in.append(float(row[1]))

plt.figure(figsize=(12, 8), dpi=150)

plt.plot(iters, e_in, linewidth=3, color='#1f77b4', label="Training Error")

if os.path.exists(val_csv_file) and os.path.getsize(val_csv_file) > 0:
    e_val = []
    with open(val_csv_file) as f:
        reader = csv.reader(f)
        rows = list(reader)
        if rows:
            for row in rows:
                if row:
                    e_val.append(float(row[1]))
            if e_val:
                val_iters = [int(row[0]) if len(row) > 1 else i for i, row in enumerate(rows)]
                plt.plot(val_iters, e_val, linewidth=3, color='#ff7f0e', label="Validation Error")

plt.xlabel("Iteration", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.title(f"Error vs Iterations ({plot_title})", fontsize=20, pad=20)

plt.legend(fontsize=14, loc="upper right", frameon=True, fancybox=True, shadow=True)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()