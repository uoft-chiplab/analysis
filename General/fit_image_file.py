import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image


def gaussian_2d(xy, amp, xo, yo, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function with rotation."""
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amp * np.exp(-(a*((x - xo)**2) + 2*b*(x - xo)*(y - yo) + c*((y - yo)**2)))
    return g.ravel()

# Camera parameters
px_to_um = 5.2  # um per px

# Load image
image_filename = "E:/Data/2025/09 September2025/09September2025/HF_light_images/no_knife.png"
image = Image.open(image_filename).convert("L")  # Convert to grayscale
data = np.array(image, dtype=float)

# Create x and y indices
x = np.linspace(0, data.shape[1] - 1, data.shape[1])
y = np.linspace(0, data.shape[0] - 1, data.shape[0])
x, y = np.meshgrid(x, y)

# Initial guess for the parameters
initial_guess = (data.max(), data.shape[1]/2, data.shape[0]/2, 20, 20, 0, data.min())

# Fit the Gaussian
popt, pcov = curve_fit(gaussian_2d, (x, y), data.ravel(), p0=initial_guess)
perr = np.sqrt(np.diag(pcov))

# Extract parameters
amp, xo, yo, sigma_x, sigma_y, theta, offset = popt
e_amp, e_xo, e_yo, e_sigma_x, e_sigma_y, e_theta, e_offset = perr
print("Fitted parameters:")
print(f"Amplitude : {amp:.2f}({1e2*e_amp:.0f})")
print(f"Y center  : {yo:.2f}({1e2*e_yo:.0f}) px")
print(f"X center  : {xo:.2f}({1e2*e_xo:.0f}) px")
print(f"Sigma X   : {sigma_x:.2f}({1e2*e_sigma_x:.0f}) px")
print(f"Sigma Y   : {sigma_y:.2f}({1e2*e_sigma_y:.0f}) px")
print(f"Sigma X um  : {sigma_x*px_to_um:.2f}({1e2*e_sigma_x*px_to_um:.0f}) um")
print(f"Sigma Y um  : {sigma_y*px_to_um:.2f}({1e2*e_sigma_y*px_to_um:.0f}) um")
print(f"Theta     : {theta:.2f}({1e2*e_theta:.0f}) radians")
print(f"Offset    : {offset:.2f}({1e2*e_offset:.0f})")

# Generate fitted data for plotting
fit_data = gaussian_2d((x, y), *popt).reshape(data.shape)

# Plot original image and fit
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(data, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Fitted 2D Gaussian")
plt.imshow(data, cmap='gray', alpha=0.5)
plt.contour(x, y, fit_data, colors='r')
plt.colorbar()

plt.tight_layout()
plt.show()
