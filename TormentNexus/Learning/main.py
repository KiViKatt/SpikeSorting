import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

file_path = 'C:\\Users\\ivank\\OneDrive\\Documents\\GitHub\\HelloWorld\\data.pkl'  # Change path for new pickles

with open(file_path, 'rb') as file:
    data = pickle.load(file)

data_dict = data[0]
color_map = plt.get_cmap('tab10')
# Helper function to plot the data in subplots and collect unique handles for the legend
def plot_subplots_and_collect_legend(axs, outer_keys, filter_extremes=False):
    handles_dict = {}

    for ax_idx, outer_key in enumerate(outer_keys):
        if outer_key not in data_dict:
            print(f"Condition {outer_key} not found in data.")
            continue

        inner_condition_data = data_dict[outer_key]
        max_data_length = 0

        for idx, (condition, values_list) in enumerate(inner_condition_data.items()):
            condition_color = color_map(idx)

            for dataset_idx, values_array in enumerate(values_list):
                # Filter out extreme values if requested
                if filter_extremes and ((values_array > 30).any() or (values_array < -30).any()):
                    continue

                max_data_length = max(max_data_length, len(values_array))

                line, = axs[ax_idx].plot(values_array, label=f"{condition}", color=condition_color)

                if condition not in handles_dict:
                    handles_dict[condition] = line

        # Add titles, labels, for each subplot (no individual legends)
        axs[ax_idx].set_title(f"{outer_key}")
        axs[ax_idx].set_xlabel("Time (s)")
        axs[ax_idx].set_ylabel("Membrane Voltage (mV)")
        axs[ax_idx].grid(True)
        axs[ax_idx].set_xlim([0, max_data_length])

        current_ticks = axs[ax_idx].get_xticks()
        scaled_ticks = current_ticks / 20
        axs[ax_idx].set_xticks(current_ticks)
        axs[ax_idx].set_xticklabels([f'{tick:.1f}' for tick in scaled_ticks])

    return handles_dict


def plot_data_in_subplots_original(outer_keys, fig_num=1):
    #extra space for the legend
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.3])

    # subplots
    axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    handles_dict = plot_subplots_and_collect_legend(axs, outer_keys)

    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis("off")
    legend_ax.legend(handles_dict.values(), handles_dict.keys(), loc='upper right')
    fig.suptitle(f"Tridesclous 2 Batch 3 Graph {fig_num} (Original Data)", fontsize=16, y=0.98)
    plt.show()

#plot data after filtering datasets
def plot_data_in_subplots_filtered(outer_keys, fig_num=1):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.3])

    # subplots
    axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    handles_dict = plot_subplots_and_collect_legend(axs, outer_keys, filter_extremes=True)

    #legend
    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis("off")
    legend_ax.legend(handles_dict.values(), handles_dict.keys(), loc='upper right')
    fig.suptitle(f"Tridesclous 2 Batch 3 Graph {fig_num} (Limited Data)", fontsize=16, y=0.98)
    plt.show()

outer_keys_list = list(data_dict.keys())

#set of 4 (original)
plot_data_in_subplots_original(outer_keys_list[:4], fig_num=1)

#set of 4 (limited)
plot_data_in_subplots_filtered(outer_keys_list[:4], fig_num=2)
