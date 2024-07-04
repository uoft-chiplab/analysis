# -*- coding: utf-8 -*-
"""
2024-03-08
Colin Dale

The goal of this script is to determine whether the heating rate scaling
with amplitude is linear or quadratic, and find which range of 
A or Delta E the scaling is quadratic.

Uses data_class to load data and create pandas array
"""
from data_class import *

def quadratic(x, A, C):
		return A*(x)**2 + C
	
def linear(x, m, b):
		return m*x + b
	
def lin_p_quad(x, m, b, C):
	return m*x+b+C*x**2
	
def calc_chi_sq(xx, yy, yerr, fit_func, popt):
	return 1/(len(xx)-len(popt)) * np.sum((yy - fit_func(xx, *popt))**2/yerr**2)

xlabel = "amplitude"
ylabel = "meanEtot_kHz"

log_ylabel = "log_DeltaE"
log_xlabel = "log_"+xlabel

files = ["2024-03-07_G_e_wiggle time=1.dat", 
		 "2024-03-07_G_e_wiggle time=9.dat"]

wiggle_times = [1, 9]
colors = ["Blue", "Orange"]
amplitude_cutoffs = [0.0, 0.0]

files = ["2024-03-08_B_e_wiggle time=1.dat",
		 "2024-03-08_B_e_wiggle time=3.dat",
		 "2024-03-08_B_e_wiggle time=5.dat",
		 "2024-03-08_B_e_wiggle time=7.dat",
		 "2024-03-08_B_e_wiggle time=10.dat"]

wiggle_times = [1, 3, 5, 7, 10]
E_i = [36.36, 36.95, 37.54, 38.09, 38.84]
colors = ["Blue", "Orange", "Green", "Red", "Purple"]

files = ["2024-03-08_D.dat"]

wiggle_times = [1]
E_i = [36.36]
colors = ["Blue"]

amplitude_cutoff = 2
amplitude_cutoffs = amplitude_cutoff*np.ones(len(colors))

fit_func = linear

results = []

# linear = lin_p_quad

###
### Analyze data runs
###

# loop over data, putting information in data class, then append class to results
for i in range(len(files)):
	run = Data(files[i])
	run.data = run.data.drop(run.data[run.data.amplitude>amplitude_cutoffs[i]].index)
	run.wiggle_time = wiggle_times[i]
	run.color = colors[i]
	
	# fit to linear and quadratic functions
	run.popt, run.pcov = curve_fit(fit_func, run.data[xlabel], run.data[ylabel])
	run.perr = np.sqrt(np.diag(run.pcov))
	run.poptq, run.pcovq = curve_fit(quadratic, run.data[xlabel], run.data[ylabel])
	run.perrq = np.sqrt(np.diag(run.pcovq))
	
	# subtract offset to get change in energy
	run.data["DeltaE"] = run.data[ylabel] - E_i[i]
	run.Ei = E_i[i]
	
	# average data by scan param
	run.group_by_mean(xlabel)
	
	
	run.avg_data[log_xlabel] = run.avg_data[xlabel].apply(np.log)
	run.avg_data[log_ylabel] = run.avg_data["DeltaE"].apply(np.log) ## PROBLEM this produces a lot of NANs
	
	# calculate log error
	run.avg_data["em_"+log_ylabel] = run.avg_data["em_"+ylabel]/run.avg_data["DeltaE"]
	
	# add fits for log data
	filtered_data = run.avg_data[run.avg_data[log_ylabel].notnull()]
	run.popt_log, run.pcov_log = curve_fit(linear, filtered_data[log_xlabel], 
					filtered_data[log_ylabel], sigma = filtered_data["em_"+log_ylabel])
	run.perr_log = np.sqrt(np.diag(run.pcov_log))
	
	# calculate chi^2 values
	run.chisq = calc_chi_sq(run.avg_data[xlabel], run.avg_data[ylabel], 
						 run.avg_data["em_"+ylabel], fit_func, run.popt)
	run.chisq_q = calc_chi_sq(run.avg_data[xlabel], run.avg_data[ylabel], 
						 run.avg_data["em_"+ylabel], quadratic, run.poptq)
	
	# legend labels
	run.label = r"$t_w =$ {}".format(run.wiggle_time)
	run.label_log = r"$t_w =$ {}, slope = {:.2f}$\pm${:.2f}".format(run.wiggle_time, 
												run.popt_log[0], run.perr_log[0])
	run.label_res = r"$t_w =$ {}, $\chi^2 =$ {:.2f}".format(run.wiggle_time, run.chisq)
	run.label_resq = r"$t_w =$ {}, $\chi^2 =$ {:.2f}".format(run.wiggle_time, run.chisq_q)
	
	if fit_func == lin_p_quad:
		print("m={:.2f}+-{:.2f}, b={:.2f}+-{:.2f}, C={:.2f}+-{:.2f}".format(run.popt[0], run.perr[0],
								 run.popt[1], run.perr[1],run.popt[2], run.perr[2]))
	elif fit_func == linear:
		print("m={:.2f}+-{:.2f}, b={:.2f}+-{:.2f}".format(run.popt[0], run.perr[0],
								 run.popt[1], run.perr[1]))
		
	results.append(run)
	

###
### Plotting
###
plt.rcParams['figure.figsize'] = [12, 8]

# four plots: raw data w/ fits, residuals, log of data with fits, residuals
fig, axs = plt.subplots(2,2)
num = 200
xx = np.linspace(0.01, max(run.data.amplitude), num)

# Raw data and fits
ax = axs[0,0]
ax.set(xlabel=xlabel, ylabel=ylabel)
for run in results:
	ax.errorbar(run.avg_data[xlabel], run.avg_data[ylabel], 
			 yerr=run.avg_data["em_"+ylabel], fmt="o", capsize=2, label=run.label, 
			 color=run.color)
	ax.plot(xx, fit_func(xx, *run.popt), "-", color=run.color)
	ax.plot(xx, quadratic(xx, *run.poptq), "--", color=run.color)
	ax.plot(np.array([0]), np.array([run.Ei]), 'x', color=run.color)
ax.legend()

# Residuals - Linear
ax = axs[1,0]
ax.set(xlabel=xlabel, ylabel="linear residuals")
for run in results:
	ax.errorbar(run.avg_data[xlabel], run.avg_data[ylabel] - fit_func(run.avg_data[xlabel], *run.popt), 
			 yerr=run.avg_data["em_"+ylabel], fmt="o", capsize=2, label=run.label_res, 
			 color=run.color)
	ax.plot(run.avg_data[xlabel], np.zeros(len(run.avg_data[xlabel])), "-", color="k")
ax.legend()

# Residuals - Quadratic
ax = axs[1,1]
ax.set(xlabel=xlabel, ylabel="quadratic residuals")
for run in results:
	ax.errorbar(run.avg_data[xlabel], run.avg_data[ylabel] - quadratic(run.avg_data[xlabel], *run.poptq), 
			 yerr=run.avg_data["em_"+ylabel], fmt="o", capsize=2, label=run.label_resq, 
			 color=run.color)
	ax.plot(run.avg_data[xlabel], np.zeros(len(run.avg_data[xlabel])), "-", color="k")
ax.legend()

# Log of data
ax = axs[0,1]
ax.set(xlabel="log "+xlabel, ylabel=log_ylabel)

# need to edit the fit function here, need to make new ones.
for run in results:
 	ax.errorbar(run.avg_data[log_xlabel], run.avg_data[log_ylabel], 
			 yerr=run.avg_data["em_"+log_ylabel], fmt="o", capsize=2, 
			 label=run.label_log, color=run.color)
 	ax.plot(run.avg_data[log_xlabel], linear(run.avg_data[log_xlabel], *run.popt_log), 
		  "-", color=run.color)
ax.legend()

plt.show()
	
