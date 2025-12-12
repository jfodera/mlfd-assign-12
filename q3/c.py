import numpy as np
import matplotlib.pyplot as plt

x1_pos = 1.0
x2_pos = -1.0

x1 = np.linspace(-1.6, 1.6, 500)
x2 = np.linspace(-2.2, 2.2, 500)
X1, X2 = np.meshgrid(x1, x2)

Z_decision = X1**3 - X2

plt.figure(figsize=(9, 8), dpi=120)
ax = plt.gca()

plt.contourf(X1, X2, Z_decision, levels=[-10, 0, 10], colors=['#FF9999', '#9999FF'], alpha=0.25)

plt.axvline(0, color='#2E86C1', linewidth=3.5, linestyle='--', label='Linear (part a): $x_1 = 0$')

x1_curve = np.linspace(-1.5, 1.5, 400)
plt.plot(x1_curve, x1_curve**3, color='#8E44AD', linewidth=4.5, label='Z-space boundary: $x_2 = x_1^3$')

plt.scatter([x1_pos], [0], s=600, c='#E74C3C', edgecolors='k', linewidth=2, zorder=10, label='$(1,0)$ to +1')
plt.scatter([x2_pos], [0], s=600, c='#3498DB', edgecolors='k', linewidth=2, zorder=10, label='$(-1,0)$ to -1')

ax.set_xlabel('$x_1$', fontsize=15)
ax.set_ylabel('$x_2$', fontsize=15)
ax.set_title('Decision Boundaries in X-space\nLinear (part a) vs Nonlinear via Z-space Transformation', fontsize=16, pad=20)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='upper left', fontsize=12, fancybox=True, shadow=True)

plt.xlim(-1.6, 1.6)
plt.ylim(-2.2, 2.2)

plt.tight_layout()
plt.show()