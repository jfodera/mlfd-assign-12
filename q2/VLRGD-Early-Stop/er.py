import matplotlib.pyplot as plt
import csv
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)
sns.set_palette("muted")

csv_file = "errors_var_es.csv"
output_file = "errors_plot_var_es.png"
plot_title = "VLRGD with Early Stopping"
val_csv_file = "val_errors_var_es.csv"

iters = []
e_in = []

with open(csv_file) as f:
    reader = csv.reader(f)
    for row in reader:
        iters.append(int(row[0]))
        e_in.append(float(row[1]))

plt.figure(figsize=(12, 8), dpi=150)

plt.plot(iters, e_in, linewidth=3, color='#1f77b4', label="Training Error")

if val_csv_file:
    e_val = []
    try:
        with open(val_csv_file) as f:
            reader = csv.reader(f)
            for row in reader:
                e_val.append(float(row[1]))
        if len(e_val) == len(iters):
            plt.plot(iters, e_val, linewidth=3, color='#ff7f0e', label="Validation Error")
    except:
        pass

plt.xlabel("Iteration", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.title(f"Error vs Iterations ({plot_title})", fontsize=20, pad=20)

plt.legend(fontsize=14, loc="upper right", frameon=True, fancybox=True, shadow=True)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()