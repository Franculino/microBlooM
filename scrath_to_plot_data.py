import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def main():
    print(os.chdir(str(r'C:\Users\UGE\Desktop\microBlooM')))
    flow_rate, hd = unpack_pickle(r'data\out\Folder_for_converged_data\MVN1_04\log_file\MVN1_04.pckl')
    flow_rate_berg, hd_berg = unpack_pickle(r'data/out/Folder_for_converged_data/MVN1_04_Berg_02/log_fileMVN1_04_Berg_02.pckl')
    flow_rate_rasmussen, hd_rasmussen = unpack_pickle(r'data/out/Folder_for_converged_data/MVN1_04_Rasmussen/log_fileMVN1_04_Rasmussen.pckl')
    frequency_plot(flow_rate, flow_rate_berg, flow_rate_rasmussen, 'prova_MVN1', 'x_axis',
                   'data/out/Folder_for_converged_data')
    frequency_plot(hd, hd_berg, hd_rasmussen, 'prova_MVN1_hd', 'x_axis',
                   'data/out/Folder_for_converged_data')


def unpack_pickle(path):
    with open(path, 'rb') as f:
        flow_rate, node_relative_residual, positions_of_elements_not_in_boundary, node_residual, two_MagnitudeThreshold, node_flow_change, \
            vessel_flow_change, indices_over_blue, node_flow_change_total, vessel_flow_change_total, pressure, hd = pickle.load(f)
    return flow_rate, hd


def frequency_plot(data, data_berg, data_rasmussen, title, x_axis, path_save):
    mean_val = np.mean(data)
    median_val = np.median(data)
    max_val = np.max(data)

    plt.figure(figsize=(25, 15), dpi=300)


    sns.histplot(data, bins='auto', kde=False, edgecolor='black', stat="percent", log_scale=True, label='Our', color = sns.color_palette(
        palette='Blues')[2],multiple="stack", alpha=0.5)
    sns.histplot(data_berg, bins='auto', kde=False, edgecolor='black', stat="percent", log_scale=True, label='Berg',color=sns.color_palette(
        palette='Greens_d')[2], alpha=0.5)
    sns.histplot(data_rasmussen, bins='auto', kde=False, edgecolor='black', stat="percent", log_scale=True, label='Rasmussen',
                 color=sns.color_palette(
        palette='OrRd')[2], alpha=0.2)

    plt.xlabel(x_axis, fontsize=30)
    plt.ylabel('Percentage (%)', fontsize=30)
    plt.title(title, fontsize=35)
    sns.despine(left=True)
    plt.grid(axis='y', alpha=0.5)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tick_params(axis='y', which='both', color='#f0f0f0')

    plt.axvline(mean_val, color='red', linestyle='dashed', label='Mean', linewidth=3)
    plt.axvline(median_val, color='blue', linestyle='dashed', label='Median', linewidth=3)
    plt.axvline(max_val, color='green', linestyle='dashed', label='Max', linewidth=3)

    plt.legend(loc='upper left', fontsize=25)
    plt.gca().set_facecolor('#f0f0f0')
    plt.tight_layout()

    plt.savefig(os.path.join(path_save, f'{title}.png'))
    plt.close()


if __name__ == "__main__":
    main()
