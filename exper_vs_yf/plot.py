import numpy as np
import matplotlib.pyplot as plt


def moving_ave(arr, k=390):
    n = len(arr)
    ma = arr.copy()
    for idx in range(1, n):
        ma[idx] = np.mean(arr[max(0, idx-k):idx])
    return ma

c1, c2 = np.loadtxt("logs/sgdm-baseline-beta-0.9-seed-1/c1c2.txt")

# get baseline (ave. and ma)
loaded = [np.loadtxt("logs/sgdm-baseline-beta-0.9-seed-{}/train_losses.txt".format(i)) for i in range(1,4)]
bl = moving_ave(np.mean(loaded, axis=0))
bl_val = np.mean([np.loadtxt("logs/sgdm-baseline-beta-0.9-seed-{}/val_acc.txt".format(i)) for i in range(1,4)], axis=0)

# get sgdm
loaded = [np.loadtxt("logs/sgdm-schedule_id-1-seed-{}/train_losses.txt".format(i)) for i in range(1,4)]
sgdm = moving_ave(np.mean(loaded, axis=0))
sgdm_val = np.mean([np.loadtxt("logs/sgdm-schedule_id-1-seed-{}/val_acc.txt".format(i)) for i in range(1,4)], axis=0)

# get yf
loaded = [np.loadtxt("logs/yf-lr-0.01-seed-{}/train_losses.txt".format(i)) for i in range(1,4)]
yf = moving_ave(np.mean(loaded, axis=0))
yf_val = np.mean([np.loadtxt("logs/yf-lr-0.1-seed-{}/val_acc.txt".format(i)) for i in range(1,4)], axis=0)

# get sgd with fixed alpha (large) and beta (0.9)
loaded = [np.loadtxt("logs/sgdm-alpha-0.4-momentum-0.9-seed-{}/train_losses.txt".format(i)) for i in range(1,4)]
alpha_large = moving_ave(np.mean(loaded, axis=0))
alpha_large_val = np.mean([np.loadtxt("logs/sgdm-alpha-0.4-momentum-0.9-seed-{}/val_acc.txt".format(i)) for i in range(1,4)], axis=0)

# get sgd with fixed alpha (small) and beta (0.9)
loaded = [np.loadtxt("logs/sgdm-alpha-0.05714285714285714-momentum-0.9-seed-{}/train_losses.txt".format(i)) for i in range(1,4)]
alpha_small = moving_ave(np.mean(loaded, axis=0))
alpha_small_val = np.mean([np.loadtxt("logs/sgdm-alpha-0.05714285714285714-momentum-0.9-seed-{}/val_acc.txt".format(i)) for i in range(1,4)], axis=0)

burn_in = 500

fig, (ax_train, ax_val) = plt.subplots(nrows=1, ncols=2, figsize=(6,3), constrained_layout=True)
# plot train
ax_train.plot(range(burn_in, len(sgdm)), sgdm[burn_in:], label = "Multistage SGDM", marker="o", markevery=0.2)
ax_train.plot(range(burn_in, len(yf)), yf[burn_in:], label="YellowFin", marker="*", markevery=0.25)
ax_train.plot(range(burn_in, len(bl)), bl[burn_in:], label = "Baseline", marker="^", markevery=0.3)
# ax_train.plot(range(burn_in, len(bl)), alpha_large[burn_in:], label = r"$\alpha=0.4$, $\beta=0.9$", marker="v", markevery=0.35)
# ax_train.plot(range(burn_in, len(bl)), alpha_small[burn_in:], label = r"$\alpha=0.057$, $\beta=0.9$", marker="X", markevery=0.4)
ax_train.set_title("Training Loss")
ax_train.set_xlabel("Batch (Iteration)")
ax_train.set_ylabel("Loss")
# plot val
ax_val.plot(sgdm_val, label = "Multistage SGDM", marker="o", markevery=0.2)
ax_val.plot(yf_val, label = "YellowFin", marker="*", markevery=0.25)
ax_val.plot(bl_val, label = "Baseline", marker="^", markevery=0.23)
# ax_val.plot(alpha_large_val, label=r"$\alpha=0.4$, $\beta=0.9$", marker="v", markevery=0.35)
# ax_val.plot(alpha_small_val, label=r"$\alpha=0.057$, $\beta=0.9$", marker="X", markevery=0.4)
ax_val.set_title("Validation Accuracy")
ax_val.set_xlabel("Epoch")
ax_val.set_ylabel("Accuracy")
handles, labels = ax_val.get_legend_handles_labels()
fig.legend(handles, labels, loc = "center right")

# save figure
plt.savefig("plot_sgdm_vs_yf_vs_bl_c1_{}_c2_{}.pdf".format(c1, c2), bbox_inches="tight")