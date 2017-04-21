from matplotlib import pyplot as plt

from ore import load_data, linecolors
files = ["MG", "heart", "tide_new"]
labelsize = 20
params= { 'text.usetex':True, 
	#'font.family':'serif', 'font.serif': ["Times", "Times New Roman"], 
	'legend.fontsize':labelsize, 'axes.labelsize':labelsize, 'axes.titlesize':labelsize, 
	'xtick.labelsize' :labelsize, 'ytick.labelsize' : labelsize
}
plt.rcParams.update(params)

for x in files:
	data = load_data(x)[0][:1000]

	fig, ax = plt.subplots(figsize=(5,5))
	ax.plot(data, color=linecolors[0], linewidth=2)
	fig.tight_layout()
	fig_path = "attractors/{}.pdf".format(x)
	print "saved to", fig_path
	fig.savefig(fig_path)