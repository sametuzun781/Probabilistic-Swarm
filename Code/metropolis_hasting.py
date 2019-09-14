import time
import numpy as np
from scipy.sparse import lil_matrix

def mh(A_a, N_bins, v_desired, alpha):
    """ 
    Metropolis-Hasting Algorithm
    """
    # Define Proposal Matrix
    Y = np.ones((N_bins,1))
    Y = A_a.dot(lil_matrix(Y))
    Y = Y.transpose()
    Y.data **= -1

    proposal_matrix = A_a.transpose().multiply(Y) 
    A_a  = []
    Y = []
    print('Check: Sum of Proposal Matrix (K) Matrix: {}'.format(np.mean(proposal_matrix.transpose().dot(lil_matrix(np.ones((N_bins,1)))))))

    # Define Intermediary Matrix
    Y = (proposal_matrix.multiply(v_desired))
    Y.data **= -1
    intermediary_matrix = (proposal_matrix.multiply(v_desired)).transpose().multiply(Y)
    Y = []

    # Define Acceptance Matrix
    acceptance_matrix = alpha * intermediary_matrix.minimum(1)
    intermediary_matrix = []

    # Define Markov Matrix
    markov_matrix = (proposal_matrix.multiply(acceptance_matrix))

    acceptance_matrix.data *= -1
    acceptance_matrix.data += 1

    M_ii_sum = ((acceptance_matrix).multiply(proposal_matrix).transpose()).dot(lil_matrix(np.ones((N_bins,1))))
    acceptance_matrix = []

    proposal_matrix = proposal_matrix.tocsr()
    M_ii_sum = M_ii_sum.tocsr()

    N_bins_list = list(range(N_bins))
    markov_matrix[N_bins_list,N_bins_list] = lil_matrix(proposal_matrix[N_bins_list,N_bins_list] + M_ii_sum[N_bins_list].transpose())
    print('Check: Sum of Markov Matrix (M) Matrix: {}'.format(np.mean(markov_matrix.transpose().dot(lil_matrix(np.ones((N_bins,1)))))))
    proposal_matrix = []
    M_ii_sum = []

    # PMF to CDF for PGA
    M_new = np.zeros([5,N_bins])
    M_where = np.zeros([5,N_bins])

    ts = time.time()
    N_filled_row = np.sum(markov_matrix[:,:] != 0, axis=1)

    for i in range(N_bins):
        N_filled_row_i = int(N_filled_row[i])
        M_where[:N_filled_row_i,i] = np.nonzero(markov_matrix[:,i])[0]
        M_new[:N_filled_row_i,i]   = markov_matrix[:,i].data
    print("Bottleneck: {}".format(time.time()-ts))

    markov_matrix = []
    markov_data = []
    cdf_markov = np.zeros([5,N_bins])
    cdf_markov[0,:] = M_new[0,:]

    for i in range(5-1):
        cdf_markov[i+1,:] = cdf_markov[i,:] + M_new[i+1,:] 

    M_new = []

    return cdf_markov, M_where