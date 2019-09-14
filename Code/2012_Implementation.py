import environment as env
import metropolis_hasting as mh
import probabilistic_guidance_algorithm as pga
import plot_figs as plt_fig
import time

ts = time.time()

## Hyperparameters
# Environment
data_path = 'PATH OF YOUR FIGURE FILE'
image_name = 'ITU_Logo.png'
scale_fig = 8
Agent_4_pixel = 20

# Metropolis - Hasting Algorithm
alpha = 0.99 # For Alpha-Min Acceptance Matrix

# Probabilistic Guidance Algorithm
N_time = 100
res_dist_count = 1
Time_to_print = 20

# Plot Figures
N_of_figures= 10
Fig_size_scale = 10

## Functions
img, m_row, m_column, N_bins, N_agent, x_init, v_desired, A_a = env.grid_env(data_path, image_name, scale_fig, Agent_4_pixel)
cdf_markov, M_where = mh.mh(A_a, N_bins, v_desired, alpha)
res_dist, total_variation, counter = pga.pga(N_bins, N_agent, x_init, m_row, m_column, N_time, res_dist_count, Time_to_print, cdf_markov, M_where, v_desired)
plt_fig.plt_fcn(N_time, N_of_figures, counter, img, Fig_size_scale, Agent_4_pixel, res_dist, total_variation)

print("Total Time to Solve: {}".format(time.time() - ts))