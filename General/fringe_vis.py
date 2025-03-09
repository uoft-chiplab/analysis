import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from numba import jit
from scipy.io import loadmat as spio_loadmat
from scipy.signal import find_peaks

fittype = 2
plot = True
subtract_cloud = False
fit_sin = False
average = False
scan = False

path = r"E:\Data\2025\02 February2025\06February2025\A_TShots\imgs"

def gauss1d(x:np.array, dx:float, x0:float, A:float, b:float):
    return A * np.exp(-((x - x0)/dx)**2 / 2) + b

@jit(nopython=True)
def gauss2d(XY:np.array, dx:float, dy:float, x0:float, y0:float, A:float, b:float):
    """returns 2d gaussian evaluated at each point specified in XY[i]"""
    x, y = XY[:,0], XY[:,1]

    g2 = A * np.exp(-((x - x0)/dx)**2 / 2)*np.exp(-((y - y0)/dy)**2 / 2) + b

    return g2.ravel()

@jit(nopython=True)
def guessGaussParams1D(x:np.array, y:np.array):
    """returns parameter guesses for 1d gaussian 
    """
    x0_guess = np.mean(x[ y> 0.9*np.max(y)])
    A_guess = np.max(y)-np.min(y)
    b_guess = np.min(y)

    HM_x = x[np.argmin(np.abs(y - np.max(y/2)))] # x at the half max
    FWHM = 2*(x0_guess - HM_x)
    dx_guess = FWHM/2.35

    return dx_guess, x0_guess, A_guess, b_guess

def gauss1dFit(x, y):
    pguess = guessGaussParams1D(x, y)

    # fit
    popt, pcov = curve_fit(gauss1d, x, y, p0 = pguess)
    perr = np.sqrt(np.diag(np.abs(pcov)))

    # output params order: [dx, x0, amp, offs]
    return popt, perr

def gauss2dFit(x, y, z): 
    """
    Fit a 2D Gaussian to data.

    Parameters:
    - x, y: Vectors of x and y coordinates which z spans
    - z: 2D array of size (len(x), len(y)) representing the data.

    Returns:
    - popts: optimal fitted parameters
    - perr: errors on fit params
    """

    dx_g, x0_g, Ax_g, bx_g = guessGaussParams1D(x, np.sum(z, axis=0)/z.shape[0])
    dy_g, y0_g, Ay_g, by_g = guessGaussParams1D(y, np.sum(z, axis=1)/z.shape[1])
    pguess = [abs(dx_g), abs(dy_g), x0_g, y0_g, (Ax_g + Ay_g)/2, (bx_g + by_g) / 2]

    # reorder into 2D list of all points on grid, (x[i], y[i]) to input into gauss2d
    XY = np.stack(np.meshgrid(x, y), axis=1) 

    # assume gauss centre is somewhere in the image
    lower_lims = [0, 0, np.min(x), np.min(y), 0,  -np.inf]
    upper_lims = [np.inf, np.inf, np.max(x), np.max(y), np.inf, np.inf]
    bounds = np.stack((lower_lims, upper_lims), axis=0)

    popts, pcov = curve_fit(gauss2d, XY, z.ravel(), p0=pguess, bounds=bounds)
    perr = np.sqrt(np.diag(np.abs(pcov)))

    return popts, perr

@jit(nopython=True)
def sin(x, A, omega, phi, c):
	return A*np.sin(omega*x - phi) + c

def OD(img):
    # read images and calculate oD
    atom_img = img[f"image2"]
    ref_img = img[f"image3"]
    bg_img = img[f"image4"]

    atom_img -= bg_img
    atom_img[atom_img < 0] = 0

    ref_img -= bg_img
    ref_img[ref_img < 0] = 0

    OD_img = np.real(-np.emath.log(atom_img / ref_img))
    
    return OD_img

def sinFit(x, y, plot=True, height=0, dist = 7):
    # adjust distance/height if necessary
    peaks = find_peaks(y, height = height, distance=dist)[0] # height = 0, distance=6

    # guess for amplitude and bg 
    bg_max, bg_min = y.max(), y.min()
    omega_g = 2*np.pi*0.09 #len(peaks)/len(x)
    A_g, bg_g = (bg_max-bg_min)/2, (bg_max+bg_min)/2
    phi_g = 0 #np.pi/2
    popts, __ = curve_fit(sin, x, y, p0 = [A_g,  omega_g, phi_g, bg_g]) 
                        #   bounds=([0, 2*np.pi*0.05, 0, bg_min], [1, 2*np.pi*1.2, np.pi, bg_max]))

    if plot:
        fig, axs = plt.subplots(1, 1, figsize = (3, 2))
        axs.plot(x, y, color="plum")
        axs.plot(x, sin(x, *popts), color="hotpink")
        axs.plot(peaks+x[0], y[peaks], linestyle="", marker="o", color="cornflowerblue", 
                markersize = 2)

        axs.set_title(f"SNR = {2*np.abs(popts[0])/np.std(y):.2f} \namplitude = {A:.2f}")
        plt.show()
    return popts

img_files = os.listdir(path)
v_list = []
a_list = []
snr_list = []
rms_list = []

for img_path in np.array(img_files):#[[169*2, 177*2, 206*2, 221*2, 234*2, 247*2, 277*2]]:
    if "settings" in img_path:
        continue
    img = spio_loadmat(os.path.join(path, img_path), simplify_cells = True)

    if subtract_cloud:
        # crop it
        OD_img = OD(img)[100:200, 400:650]
        xs, ys = np.arange(OD_img.shape[1]), np.arange(OD_img.shape[0])

        # fits atom cloud to gaussian and subtract from OD
        if fittype == 2:
            # 2D fit
            # I think x and y are flipped in the fit...
            dx, dy, x0, y0, A, bg = gauss2dFit(xs, ys, OD_img)[0] # ignore fit erorrs
            Ax, Ay = A, A # here just for code compatibility between 1D/2D fits
            bgx, bgy = bg, bg

            if plot:
                # plot fit
                fig, axs = plt.subplots(2, 1, figsize = (15, 5))
                axs[0].imshow(OD_img, cmap="RdPu_r")
                XY = np.stack(np.meshgrid(xs, ys), axis=1) 
                axs[1].imshow(np.reshape(gauss2d(XY, dx, dy, x0, y0, A, bg), OD_img.shape), 
                            vmin = OD_img.min(), vmax = OD_img.max(), cmap="RdPu_r")
                
                # plot where 1D slices will be taken from
                axs[0].axvline(x0, color="powderblue", linestyle="--")
                axs[0].axhline(y0, color="powderblue", linestyle="--")

                axs[0].set_title(img_path)

        else:
            # fit to 1D instead
            dx, x0, Ax, bgx = gauss1dFit(xs, OD_x)[0] # ignore fit erorrs
            dy, y0, Ay, bgy = gauss1dFit(ys, OD_y)[0]

        OD_x = OD_img[int(y0), :]
        OD_y = OD_img[:, int(x0)]

        # subtract fit
        x_bg = OD_x - gauss1d(xs, dx, x0, Ax, bgx)
        y_bg = OD_y - gauss1d(ys, dy, y0, Ay, bgy)

        if plot:
            # plot fit to 1D slices and subtracted background
            fig, axs = plt.subplots(2,2, figsize = (7, 3))
            axs = axs.flatten()

            axs[0].plot(ys, OD_y, color="plum")
            axs[0].plot(ys, gauss1d(ys, dy, y0, Ay, bgy), color="hotpink")
            axs[1].plot(xs, OD_x, color="plum")
            axs[1].plot(xs, gauss1d(xs, dx, x0, Ay, bgy), color="hotpink")

            axs[2].plot(y_bg, color="hotpink")
            axs[3].plot(x_bg, color="hotpink")

            
            axs[0].set_ylabel("fit y")
            axs[1].set_ylabel("fit x")
            axs[2].set_ylabel("y, fit subtracted")
            axs[3].set_ylabel("x, fit subtracted")

            plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
            plt.show()
    else: # grab indices near centre of image to fit to

        # crop to smaller region
        OD_img = OD(img)[100:200, 400:600]
        xs, ys = np.arange(OD_img.shape[1]), np.arange(OD_img.shape[0])

        # subtract the  background
        OD_img -= np.mean(OD_img)

        y_i = int(OD_img.shape[0]/2)
        x_i = int(OD_img.shape[1]/4)

        x_bg = OD_img[y_i, :]
        y_bg = OD_img[:, x_i]

    if average:
        # Window size for moving average
        window_size = 3
        # Create the moving average filter
        window = np.ones(window_size) / window_size

        fig, axs = plt.subplots(1, 1, figsize = (3, 2))
        axs.plot(xs, x_bg, label='Raw', color="plum")

        # Apply the moving average using convolution
        x_bg = np.convolve(x_bg, window, mode='same')
        y_bg = np.convolve(y_bg, window, mode='same')
        
        if False:
            axs.plot(xs, x_bg, label='Smoothed', linewidth=2, color="hotpink")
            axs.legend()

    if fit_sin:
        # try fitting to a sinusoid
        i_start = 50
        i_end = 150
        xs_crop, x_bg_crop = xs[i_start:i_end], x_bg[i_start:i_end]

        popts_x = sinFit(xs_crop, x_bg_crop, plot)
        # popts_y = sinFit(ys, y_bg, plot)

        A, __, __, bg = popts_x
        A = np.abs(A)
        fringe_min, fringe_max = bg-A, bg+A

        x_bg_crop_std = np.std(x_bg_crop)
    
    else:
        fringe_min, fringe_max = x_bg.min(), x_bg.max()
        x_bg_crop_std = 1 # placeholder

    visibility = (fringe_max-fringe_min)/(fringe_max+fringe_min)
    amplitude = np.abs((fringe_max-fringe_min)/fringe_max)
    SNR = (fringe_max - fringe_min)/x_bg_crop_std
    RMS = np.sqrt(np.mean(OD_img**2))

    v_list.append(visibility)
    a_list.append(amplitude)
    snr_list.append(SNR)
    rms_list.append(RMS)
        
v_list = np.array(v_list)
a_list = np.array(a_list)
snr_list = np.array(snr_list)
rms_list = np.array(rms_list)

# if scanning
if scan:
    scanlist = np.loadtxt(r"E:\Data\2025\02 February2025\06February2025\E_HFinLF\E_HFinLF.mscan",
                          skiprows=6, usecols=1, delimiter=",")

    # plot variable choice
    y_list = rms_list
    y_name = "RMS"

    plt.figure()
    plt.plot(scanlist, y_list, color="hotpink", marker=".", linestyle="")
    plt.xlabel(r"tshots")
    plt.ylabel("RMS")

    # average over scan times
    y_avg = []
    for t in np.unique(scanlist):
        y_avg.append(np.mean(y_list[scanlist == t]))

    plt.plot(np.unique(scanlist), y_avg, color="cornflowerblue", 
             marker=".", linestyle="", markersize = 15, label="average")
    plt.legend()
else:
    plt.figure()
    plt.plot(np.arange(len(a_list)), rms_list, color="hotpink", marker=".", linestyle="")
    plt.xlabel("shot no.")
    plt.ylabel("abs. rms")