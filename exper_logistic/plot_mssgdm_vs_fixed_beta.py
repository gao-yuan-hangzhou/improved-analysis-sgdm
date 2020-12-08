import numpy as np
import matplotlib.pyplot as plt

def moving_ave(arr, k=937):
    n = len(arr)
    ma = arr.copy()
    for idx in range(1, n):
        ma[idx] = np.mean(arr[max(0, idx-k): idx])
    return ma

def get_moving_ave(alpha, beta):
    loaded = [np.loadtxt("logs/fixed_alpha-{}-beta-{}-seed-{}/train_losses.txt".format(alpha, beta, sd)) for sd in (1,2,3)]
    return moving_ave(np.mean(loaded, axis=0))


beta = 0.9
# param_list = [(xx, beta) for xx in (0.1, 0.2, 0.5, 1.0, 2.0, 2.5)]
param_list = [(xx, beta) for xx in (0.1, 0.5, 1.0, 2.0, 2.5, 5.0)]

loss_dict = {}

markers = ["o", "v", "s", "p", "P", "*", "+", "x"]
burn_in, end = 0, 500
later_in, later_end = 10000, 25000


loaded = [np.loadtxt("logs/sgdm_c1-2-c2-1-seed-{}/train_losses.txt".format(i)) for i in (1, 2, 3)]
mssgdm_ma = moving_ave(np.mean(loaded, axis=0))
loaded = [np.loadtxt("logs/sgdm_c1-2-c2-1-momentum_val-0.9-seed-{}/train_losses.txt".format(i)) for i in (1, 2, 3)]
fix_beta09_ma = moving_ave(np.mean(loaded, axis=0))
loaded = [np.loadtxt("logs/sgdm_c1-2-c2-1-momentum_val-0.0-seed-{}/train_losses.txt".format(i)) for i in (1, 2, 3)]
fix_beta00_ma = moving_ave(np.mean(loaded, axis=0))
loaded = [np.loadtxt("logs/sgdm_c1-2-c2-1-momentum_val-0.6-seed-{}/train_losses.txt".format(i)) for i in (1, 2, 3)]
fix_beta06_ma = moving_ave(np.mean(loaded, axis=0))

loaded  = [np.loadtxt("logs/fixed_alpha-0.6666666666666666-beta-0.9-seed-{}/train_losses.txt".format(i)) for i in (1, 2, 3)]
fixed_alpha_large_ma = moving_ave(np.mean(loaded, axis=0))

loaded  = [np.loadtxt("logs/fixed_alpha-0.09523809523809523-beta-0.9-seed-{}/train_losses.txt".format(i)) for i in (1, 2, 3)]
fixed_alpha_small_ma = moving_ave(np.mean(loaded, axis=0))

markers = ["o", "v", "s", "p", "P", "*", "+", "x"]
burn_in, end = 0, 500
later_in, later_end = 1500, 25000
fig, (ax_initial, ax_later) = plt.subplots(nrows=1, ncols=2, figsize=(6,3), sharex=False, sharey=False, constrained_layout=True)
idx = 0
# initial iterates
ax_initial.plot(range(burn_in, end), mssgdm_ma[burn_in:end], label="Multistage SGDM", marker="o", markevery=0.2) # both change
# ax_initial.plot(range(burn_in, end), fix_beta00_ma[burn_in:end], label="VS-SGD", marker="*", markevery=0.25) # changing alpha, fixed beta
# # ax_initial.plot(range(burn_in, end), fix_beta06_ma[burn_in:end], label=r"SGDM $\beta=0.6$", marker="p", markevery=0.3)
# ax_initial.plot(range(burn_in, end), fix_beta09_ma[burn_in:end], label=r"VS-SGDM $\beta=0.9$", marker="P", markevery=0.35) # fixed large alpha
ax_initial.plot(range(burn_in, end), fixed_alpha_large_ma[burn_in:end], label=r"$\alpha=0.66, \beta=0.9$", marker="^", markevery=0.25) # fixed large alpha
ax_initial.plot(range(burn_in, end), fixed_alpha_small_ma[burn_in:end], label=r"$\alpha=0.095, \beta=0.9$", marker="X", markevery=0.3) # fixed large alpha
ax_initial.plot()
ax_initial.set_title("Initial Iterations")
# later iterates
ax_later.plot(range(later_in, later_end), mssgdm_ma[later_in:later_end], label="Multistage SGDM", marker="o", markevery=4000)
# ax_later.plot(range(later_in, later_end), fix_beta00_ma[later_in:later_end], label=r"Multistage $\beta=0$", marker="*", markevery=0.25)
# # ax_later.plot(range(later_in, later_end), fix_beta06_ma[later_in:later_end], label=r"SGDM $\beta=0.6$", marker="p", markevery=0.3)
# ax_later.plot(range(later_in, later_end), fix_beta09_ma[later_in:later_end], label=r"Multistage $\beta=0.9$", marker="P", markevery=0.35)
ax_later.plot(range(later_in, later_end), fixed_alpha_large_ma[later_in:later_end], label=r"$\alpha=0.66, \beta=0.9$", marker="^", markevery=5000) # fixed large alpha
ax_later.plot(range(later_in, later_end), fixed_alpha_small_ma[later_in:later_end], label=r"$\alpha=0.095, \beta=0.9$", marker="X", markevery=6000) # fixed large alpha
ax_later.set_title("Later Iterations")

handles, labels = ax_later.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5, 1), labelspacing=0.3, columnspacing=1.0, ncol=1)
fig.text(0.5, -0.04, 'Batch (Iteration)', ha='center')
fig.text(-0.02, 0.45, 'Training Loss', va='center', rotation='vertical')
plt.savefig("logistic_mssgdm_vs_two_static_baselines.pdf", bbox_inches="tight")