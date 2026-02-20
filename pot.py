import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm

def sombrero(r, scale=1.0):
    return scale * np.sin(r) / (r + np.finfo(float).eps)

x = np.linspace(-8, 8, 101)
y = np.linspace(-8, 8, 101)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

Z = sombrero(R)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(
    X, Y, Z,
    cmap=cm.coolwarm,
    edgecolor="none",
    antialiased=True,
)

ax.set_title("Mexican-Hat Potential")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

fig.subplots_adjust(bottom=0.2)
slider_ax = fig.add_axes([0.25, 0.07, 0.5, 0.03])
scale_slider = Slider(
    ax=slider_ax,
    label="Scale",
    valmin=0.1,
    valmax=5.0,
    valinit=1.0,
    valstep=0.1,
)

def update(val):
    new_scale = scale_slider.val
    Z_new = sombrero(R, scale=new_scale)
    ax.collections.clear()
    ax.plot_surface(
        X, Y, Z_new,
        cmap=cm.coolwarm,
        edgecolor="none",
        antialiased=True,
    )
    fig.canvas.draw_idle()

scale_slider.on_changed(update)
plt.show()
