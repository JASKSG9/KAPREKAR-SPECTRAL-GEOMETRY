import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import inv

def build_hh(N, g):
    H = np.diag(np.zeros(N))
    H += np.diag((1+g)*np.ones(N-1), k=1)
    H += np.diag((1-g)*np.ones(N-1), k=-1)
    return H

N = 30
g = 0.5
H = build_hh(N, g)

# Elliptical contour
theta = np.linspace(0, 2*np.pi, 200)
center = 0.0
a, b = 2.5, 2.0
z_vals = center + a*np.cos(theta) + 1j*b*np.sin(theta)

# Observables
edge = np.zeros(N, dtype=complex); edge[-1] = 1.0
bulk = np.ones(N, dtype=complex) / np.sqrt(N)

mu_edge = [np.vdot(edge, inv(z*np.eye(N)-H) @ edge) for z in z_vals]
mu_bulk = [np.vdot(bulk, inv(z*np.eye(N)-H) @ bulk) for z in z_vals]
res_norm = [np.linalg.norm(inv(z*np.eye(N)-H), 2) for z in z_vals]

mu_edge = np.array(mu_edge)
mu_bulk = np.array(mu_bulk)
res_norm = np.array(res_norm)

fig, ax = plt.subplots(figsize=(6,5))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)

scat = ax.scatter([], [], c=[], cmap='coolwarm', s=50, vmin=0, vmax=np.max(res_norm))
point, = ax.plot([], [], 'ko', markersize=8)
line, = ax.plot([], [], 'k-', lw=0.5)

def init():
    scat.set_offsets(np.empty((0,2)))
    line.set_data([], [])
    point.set_data([], [])
    return scat, line, point

def animate(i):
    # draw contour up to i
    line.set_data(np.real(z_vals[:i+1]), np.imag(z_vals[:i+1]))
    # current point
    z = z_vals[i]
    point.set_data([np.real(z)], [np.imag(z)])
    # show mu_edge vs mu_bulk as color? We'll just highlight current point
    # instead, use separate annotation
    return scat, line, point

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(z_vals), interval=50, blit=True)

# Save as GIF
ani.save('demo/orsm-demo.gif', writer='pillow', fps=20)
plt.close()
