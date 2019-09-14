import time
import numpy as np
from scipy.sparse import lil_matrix

def pga(N_bins, N_agent, x_init, m_row, m_column, N_time, res_dist_count, Time_to_print, cdf_markov, M_where, v_desired):
    """ 
    Probablilistic Guidance Algorithm (PGA)
    """

    # Distribution of the agents
    curr_dist = np.random.choice(np.arange(0, N_bins), size = N_agent, p=x_init[:,0]).astype(np.int32)

    # Total Variation
    total_variation = np.zeros([N_time])

    # Initial distribution
    res_dist = np.zeros([m_row,m_column,int(N_time/res_dist_count)+1])
    for k in range(N_agent):
        i = int(curr_dist[k]/m_column)
        j = int(curr_dist[k]%m_column)
        res_dist[i,j,0] += 1

    counter = 1
    # PGA
    for t in range(N_time):
        ts_step = time.time() if t%Time_to_print == 0 else 0
        print("Time: {} of {}".format(t,N_time)) if t%Time_to_print == 0 else 0

        z = np.random.uniform(0,1, size=N_agent)   

        M_where_state = np.argmax(cdf_markov[:,curr_dist] > z, axis=0)
        curr_dist = M_where[M_where_state, curr_dist].astype(np.int32)
        
        unique, counts = np.unique(curr_dist, return_counts=True)
        count_agent = np.zeros(v_desired.size)
        unique_list = unique.tolist()
        count_agent[[int(i) for i in unique_list]] = counts
        total_variation[t] = np.sum(np.abs(count_agent/N_agent - v_desired))
        
        if t%res_dist_count == 0:
            for k in range(N_agent):
                i = int(curr_dist[k]/m_column)
                j = int(curr_dist[k]%m_column)
                res_dist[i,j,counter] += 1
            counter += 1
            
        print("Total Variation: {}".format(total_variation[t])) if t%Time_to_print == 0 else 0    
        print("Time to Solve: {}".format((time.time() - ts_step)*Time_to_print)) if t%Time_to_print == 0 else 0

    return res_dist, total_variation, counter
        

  



