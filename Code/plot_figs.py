import seaborn as sns
import matplotlib.pyplot as plt

def plt_fcn(N_time, N_of_figures, counter, img, Fig_size_scale, Agent_4_pixel, res_dist, total_variation):

    save_count = int(N_time/N_of_figures)
    for i in range(int(counter/save_count)):
        plt.figure(num=i, figsize=(img.shape[1]/Fig_size_scale,img.shape[0]/Fig_size_scale))
        sns_plot = sns.heatmap(res_dist[:,:,i*save_count], vmin=0, vmax=Agent_4_pixel*1)
        sns_plot = sns_plot.get_figure()
        print(i)
        fig_name = str(i) + '.png'
        sns_plot.savefig(fig_name)

    plt.figure(figsize=(10,5))
    plt.plot(total_variation)
    plt.xlabel('time', fontsize=12)
    plt.ylabel('Total Variation', fontsize=12)
    plt.savefig('Total Variation')
    # plt.show()

