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

alpha = 5.0
# param_list = [(xx, beta) for xx in (0.1, 0.2, 0.5, 1.0, 2.0, 2.5)]
# param_list = [(xx, beta) for xx in (0.1, 0.5, 1.0, 2.0, 5.0)]
param_list = [(alpha, xx) for xx in (0.1, 0.2, 0.5, 0.9, 0.99)]

loss_dict = {}

markers = ["o", "v", "s", "p", "P", "*", "+", "x"]
burn_in, end = 0, 500
later_in, later_end = 10000, 25000

fig, (ax_initial, ax_later) = plt.subplots(nrows=1, ncols=2, figsize=(6,3), sharex=False, sharey=False, constrained_layout=True)
idx = 0
for alpha, beta in param_list:
    loss_dict[alpha, beta] = get_moving_ave(alpha, beta)
    label_str = r"$\alpha={}, \beta={}$".format(alpha, beta)
    # plot initial 
    ax_initial.plot(range(burn_in, end), loss_dict[alpha, beta][burn_in:end], label=label_str, marker=markers[idx], markevery=0.3)
    ax_initial.set_title("Initial Iterations")
    # plot subsequent
    ax_later.plot(range(later_in, later_end), loss_dict[alpha, beta][later_in:later_end], label=label_str, marker=markers[idx], markevery=0.3)
    ax_later.set_title("Later Iterations")
    # increment count
    idx += 1
handles, labels = ax_later.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.99), labelspacing=0.1, columnspacing=1.0, ncol=1)
# plt.suptitle("Logistic Regression on MNIST")
fig.text(0.5, -0.04, 'Batch (Iteration)', ha='center')
fig.text(-0.02, 0.45, 'Training Loss', va='center', rotation='vertical')
# save...
plt.savefig("logistic_alpha-{}_compare_beta.pdf".format(str(int(alpha*10))), bbox_inches="tight")