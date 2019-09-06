import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2 
from keras.preprocessing.image import array_to_img

ts = time.time()

# Hyperparameters
scale_fac = 6
ite = 1
res_dist_ite = 1 #int(N_time/10)
Time_to_print = 20
How_many_fig= 10
Fig_size_scale = 10

Agent_4_pixel = 20
alpha = 0.99 # For Alpha-Min Acceptance Matrix
N_time = 100

# Define Picture and Desirex Pixels
# ITU Logo# ITU Logo# ITU Logo# ITU Logo# ITU Logo# ITU Logo# ITU Logo# ITU Logo
data_path = 'C:/Users/samet/Desktop/Guncelle/28.08.19/Behcet_Acikmese/Swarm/2012-A Markov chain approach to probabilistic swarm guidance/Code/'
# data_path = '/media/su/Windows/Users/samet/Desktop/Guncelle/28.08.19/Behcet_Acikmese/Swarm/2012-A Markov chain approach to probabilistic swarm guidance/Code/'
img = cv2.imread(data_path+'ITU_Logo.png') 
print(img.shape)
img = cv2.resize(img,(int(img.shape[1]/scale_fac),int(img.shape[0]/scale_fac)))
print(img.shape)
img_show = plt.imshow(array_to_img(img))
adana = img[:,:,0]

pixels = np.where(img[:,:,0] < 200)
img_cont = np.ones([img.shape[0],img.shape[1],3])*255
img_cont[pixels[0],pixels[1],0] = 0
img_show_cont = plt.imshow(array_to_img(img_cont))

pixels = pixels[0][:]*img.shape[1] + pixels[1][:]
pixels = pixels.tolist()
# ITU Logo# ITU Logo# ITU Logo# ITU Logo# ITU Logo# ITU Logo# ITU Logo# ITU Logo

# Define Environment 
m_row = img.shape[0] # Number of row
m_column = img.shape[1] # Number of column
m = int(m_row*m_column) # Number of bins

N = len(pixels)*Agent_4_pixel # Number of agents
print('Number of agents: {}'.format(N))


x_t =  np.ones([m,1])*1/m # Vector of probablities
#x_t =  np.zeros([m,1]) # Vector of probablities

       
#x_t[0:] = 1/4
#x_t[m_column-1] = 1/4
#x_t[m_column*(m_row-1)] = 1/4
#x_t[(m_row*m_column)-1] = 1/4
      
#size_1 = list(range(1,m_column-1)) # Ust Kenar
#size_7 = list(range(m_column*(m_row-1)+1-int(m_row/3*m_column),((m_row*m_column)-1)-int(m_row/3*m_column))) # Orta Yatay
#size_5 = list(range(m_column*(m_row-1)+1-int(m_row/2*m_column),((m_row*m_column)-1)-int(m_row/2*m_column))) # Orta Yatay
#size_8 = list(range(m_column*(m_row-1)+1-int(2*m_row/3*m_column),((m_row*m_column)-1)-int(2*m_row/3*m_column))) # Orta Yatay
#size_2 = list(range(m_column*(m_row-1)+1,(m_row*m_column)-1)) # Alt Kenar
#
#size_3 = list(range(0+m_column,m_column*(m_row-1),m_column)) # Sol Kenar
#size_9 = list(range(m_column-1+m_column-int(m_column/3),(m_row*m_column)-1-int(m_column/3),m_column)) # Orta Dikey
#size_6 = list(range(m_column-1+m_column-int(m_column/2),(m_row*m_column)-1-int(m_column/2),m_column)) # Orta Dikey
#size_10 = list(range(m_column-1+m_column-int(2*m_column/3),(m_row*m_column)-1-int(2*m_column/3),m_column)) # Orta Dikey
#size_4 = list(range(m_column-1+m_column,(m_row*m_column)-1,m_column)) # Sag Kenar
#
#x_t =  np.zeros([m,1]) # Vector of probablities
#size_list = size_1+size_2+size_3+size_4+size_5+size_6+size_7+size_8+size_9+size_10
#x_t[size_list] = 1/len(size_list)
#x_t[[332,333,334,335,336,337,338,339,340,341,342,343]] = 1/len(size_list) # Orta Nokta Kesisimi

print('Check: Sum of Vector of probablities: {}'.format(np.sum(x_t,axis=0)))

v =  np.zeros([m]) # Specified distribution

# ITU
#ind_list = [34,36,37,38,39,40,42,45,
#            50,54,58,61,
#            66,70,74,77,
#            82,86,90,93,
#            98,102,106,107,108,109]

# Say Hi
#ind_list = list(range(64, 69)) + list(range(126, 131)) + list(range(188, 193)) + [95,161] + list(range(70, 74)) + list(range(132, 136)) + [101,104,163,166,194,197] + [75,79,107,109,139,170,201]

# ITU Logo
ind_list = pixels

dens = 1/len(ind_list)-0.0001/m
v[:] = (1 - dens*len(ind_list)) / (m-len(ind_list))
v[ind_list] = dens

print('Check: Sum of Desired Distribution: {}'.format(np.sum(v,axis=0)))

A_a = np.zeros([m,m]) # Allowable transitions

for i in range(m):
    # Kenarlar
    if (0<i) and (i<m_column-1): # Ust
        A_a[i+1,i] = 1
        A_a[i-1,i] = 1
        A_a[i+m_column,i] = 1
    elif (m_column*(m_row-1) < i) and (i < (m_row*m_column)-1): # Alt
        A_a[i+1,i] = 1
        A_a[i-1,i] = 1
        A_a[i-m_column,i] = 1
    elif (i%m_column == 0) and (i != 0) and ( i != m_column*(m_row-1)): # Sol
        A_a[i+1,i] = 1
        A_a[i+m_column,i] = 1
        A_a[i-m_column,i] = 1
    elif (i%m_column == m_column-1) and (i != m_column-1) and (i != (m_row*m_column)-1): # Sag
        A_a[i-1,i] = 1
        A_a[i+m_column,i] = 1
        A_a[i-m_column,i] = 1        
    # Koseler        
    elif i == 0:
        A_a[i+1,i] = 1
        A_a[i+m_column,i] = 1
    elif i == m_column-1:
        A_a[i-1,i] = 1
        A_a[i+m_column,i] = 1
    elif i ==  m_column*(m_row-1):
        A_a[i+1,i] = 1
        A_a[i-m_column,i] = 1
    elif i ==  (m_row*m_column)-1:
        A_a[i-1,i] = 1
        A_a[i-m_column,i] = 1
    # Geri kalanlar    
    else:
        A_a[i+1,i] = 1
        A_a[i-1,i] = 1
        A_a[i+m_column,i] = 1
        A_a[i-m_column,i] = 1

# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT        
# ONLY ENVIRONMENT
# ONLY ENVIRONMENT
     
# Define Matrices
        
# Metropolis-Hasting Algorithm
K = np.zeros([m,m]) # Proposal Matrix
A_a_t = A_a.transpose()
K =  np.divide(A_a_t, A_a_t.sum(axis=1)[None,:]) 

A_a  = []
A_a_t = []
print('Check: Sum of K Matrix: {}'.format(np.mean(np.matmul(np.ones([1,m]),K))))

R = np.zeros([m,m]) # Intermediary Matrix
R_ind = np.where((K*v) != 0)

# Kv = K*v
R[R_ind[0],R_ind[1]] = (K*v).transpose()[R_ind[0],R_ind[1]]/((K*v)[R_ind[0],R_ind[1]])
R_ind = []
# Kv = []

# F = np.zeros([m,m]) # Acceptance Matrix
F = alpha * np.minimum(1,R) # Acceptance Matrix

R = []      
# M = np.zeros([m,m]) # Markov Matrix
M_ii_sum = np.zeros(m)

M = K*F
M_ii_sum = ((1-F) * K).sum(axis=0)
F = []
for j in range(m):
    M[j,j] = K[j,j] + M_ii_sum[j]
    
print('Check: Sum of M Matrix: {}'.format(np.mean(np.matmul(np.ones([1,m]),M))))
K = []

# [2]+[3]
M_new = np.zeros([5,m])
M_where = np.zeros([5,m])
for i in range(m):
    ss = np.count_nonzero(M[:,i])
    M_where[:ss,i] = np.nonzero(M[:,i])[0]
    M_new[:ss,i]   = M[np.nonzero(M[:,i])[0],i]

M = []
W = np.zeros([5,m])
W[0,:] = M_new[0,:]

for i in range(5-1):
    W[i+1,:] = W[i,:] + M_new[i+1,:] 

M_new = []

# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET
# TO RESET

# Probablilistic Guidance Algorithm (PGA)
# [1]
r = np.random.choice(np.arange(0, m), size = N, p=x_t[:,0]).astype(np.int32)

# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE
# TO CONTINOUE

T = np.zeros([N_time])


res_dist = np.zeros([m_row,m_column,int(N_time/res_dist_ite)+1])

# Initial distribution

for k in range(N):
    i = int(r[k]/m_column)
    j = int(r[k]%m_column)
    res_dist[i,j,0] += 1

for t in range(N_time):
    ts_step = time.time() if t%Time_to_print == 0 else 0
    print("Time: {} of {}".format(t,N_time)) if t%Time_to_print == 0 else 0

    z = np.random.uniform(0,1, size=N)   
    s = np.zeros([N])

    M_where_state = np.argmax(W[:,r] > z, axis=0)
    r = M_where[M_where_state, r].astype(np.int32)

    unique, counts = np.unique(r, return_counts=True)
    count_agent = np.zeros(v.size)
    unique_list = unique.tolist()
    count_agent[[int(i) for i in unique_list]] = counts
    T[t] = np.sum(np.abs(count_agent/N - v))
    
    if t%res_dist_ite == 0:
        for k in range(N):
            i = int(r[k]/m_column)
            j = int(r[k]%m_column)
            res_dist[i,j,ite] += 1
        ite += 1
        
    print("Total Variation: {}".format(T[t])) if t%Time_to_print == 0 else 0    
    print("Time to Solve: {}".format((time.time() - ts_step)*Time_to_print)) if t%Time_to_print == 0 else 0
    
#cmap = sns.cubehelix_palette(light=1, as_cmap=True)        
#cmap = sns.dark_palette("red", as_cmap=True)
#cmap = sns.diverging_palette(250, 2, as_cmap=True)

save_ite = int(N_time/How_many_fig)
for i in range(int(ite/save_ite)):
    plt.figure(num=i, figsize=(img.shape[1]/Fig_size_scale,img.shape[0]/Fig_size_scale))
#    sns.heatmap(res_dist[:,:,i*2], cmap=cmap)
#    sns.heatmap(res_dist[:,:,i*2], cmap='Blues', linewidths=.1)
#    sns.heatmap(res_dist[:,:,i*2], vmin=0.2, vmax=0.7)
    sns_plot = sns.heatmap(res_dist[:,:,i*save_ite], vmin=0, vmax=Agent_4_pixel*1)
    sns_plot = sns_plot.get_figure()
    print(i)
    fig_name = str(i) + '.png'
    sns_plot.savefig(fig_name)
    # sns_plot.clf()
       
# res_dist = np.zeros([m_row,m_column])
# for k in range(N):
#     i = int(r[k]/m_column)
#     j = int(r[k]%m_column)
#     res_dist[i,j] += 1
 
# plt.figure(figsize=(img.shape[1]/Fig_size_scale,img.shape[0]/Fig_size_scale))        
# sns.heatmap(res_dist[:,:,-1])
# sns.heatmap(res_dist[:,:])

plt.figure(figsize=(10,5))
plt.plot(T)
plt.xlabel('time', fontsize=12)
plt.ylabel('Total Variation', fontsize=12)
plt.savefig('Total Variation')
# plt.show()

print("Total Time to Solve: {}".format(time.time() - ts))