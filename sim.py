import numpy as np
print ("Required shape of vectors is Nx1, i.e. column vectors")
def cosim(u,v):
	distu=np.sum(u**2,axis=0)**0.5
	distv=np.sum(v**2,axis=0)**0.5
	normu=np.copy(u/distu)
	normv=np.copy(v/distv)
	cosine=np.dot(normu.T,normv)
	angle=np.arccos(cosine)*(180/np.pi)
	return angle

def eusim(u,v):
	distu=np.sum(u**2,axis=0)**0.5
	distv=np.sum(v**2,axis=0)**0.5
	normu=np.copy(u/distu)
	normv=np.copy(v/distv)
	dist_vec=normu-normv
	mag_dist_vec=np.sum(dist_vec**2,axis=0)**0.5
	return mag_dist_vec
