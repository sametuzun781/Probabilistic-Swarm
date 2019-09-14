import cv2
import numpy as np
from scipy.sparse import lil_matrix

def grid_env(data_path, image_name, scale_fig, Agent_4_pixel):

	# Define Image
	img = cv2.imread(data_path + image_name)
	print(img.shape)
	img = cv2.resize(img,(int(img.shape[1]/scale_fig),int(img.shape[0]/scale_fig)))
	print(img.shape)
	cv2.imshow('', (img))
	cv2.waitKey(0)

	# Define Desired Pixels
	pixels = np.where(img[:,:,0] < 200)
	img_cont = np.ones([img.shape[0],img.shape[1],3])*255
	img_cont[pixels[0],pixels[1],1] = 0
	cv2.imshow('', img_cont)
	cv2.waitKey(0)
	pixels = pixels[0][:]*img.shape[1] + pixels[1][:]
	pixels = pixels.tolist()

	# Define Environment 
	m_row = img.shape[0] # Number of row
	m_column = img.shape[1] # Number of column
	N_bins = int(m_row*m_column) # Number of bins

	# Number of agents
	N_agent = len(pixels)*Agent_4_pixel 
	print('Number of agents: {}'.format(N_agent))

	# Define initial distribution
	x_init =  np.ones([N_bins,1])*1/N_bins # Vector of probablities
	print('Check: Sum of Vector of probablities: {}'.format(np.sum(x_init,axis=0)))

	# Define desired distribution
	v_desired =  np.zeros([N_bins]) # Specified distribution
	ind_list = pixels # For Picture
	dens = 1/len(ind_list)-0.0001/N_bins
	v_desired[:] = (1 - dens*len(ind_list)) / (N_bins-len(ind_list))
	v_desired[ind_list] = dens
	print('Check: Sum of Desired Distribution: {}'.format(np.sum(v_desired,axis=0)))

	# Define adjacency matrix
	A_a = lil_matrix((N_bins,N_bins)) # Allowable transitions

	for i in range(N_bins):
		# Edges
		if (0<i) and (i<m_column-1): # Up
			A_a[i+1,i] = 1
			A_a[i-1,i] = 1
			A_a[i+m_column,i] = 1
		elif (m_column*(m_row-1) < i) and (i < (m_row*m_column)-1): # Bottom
			A_a[i+1,i] = 1
			A_a[i-1,i] = 1
			A_a[i-m_column,i] = 1
		elif (i%m_column == 0) and (i != 0) and ( i != m_column*(m_row-1)): # Left
			A_a[i+1,i] = 1
			A_a[i+m_column,i] = 1
			A_a[i-m_column,i] = 1
		elif (i%m_column == m_column-1) and (i != m_column-1) and (i != (m_row*m_column)-1): # Right
			A_a[i-1,i] = 1
			A_a[i+m_column,i] = 1
			A_a[i-m_column,i] = 1        
		# Corners        
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
		# The rest of pixels??    
		else:
			A_a[i+1,i] = 1
			A_a[i-1,i] = 1
			A_a[i+m_column,i] = 1
			A_a[i-m_column,i] = 1



	return img, m_row, m_column, N_bins, N_agent, x_init, v_desired, A_a
