#This is the first of a serious attempt on calculating elastic-plastic behavior of material by 
#variational method.
#The major tools used will be: simulated annealing, with thermal fluctuation according to Const * exp(1/T)
#and the energy minimum principle of a body under const. T and V


import numpy as np
import matplotlib.pyplot as plt
from numpy import *

unit = 1. #unit of calculation


def InitialC(n1,n2): 			#creating an n by n array of a displacement function which has cylindrical symmery, this function retruning a empty array
	return np.zeros([n1,n2,2]) 	#returning a displacement field

def Position(dis): 							#returning a position field
	field = np.zeros(dis.shape) 				#field has the same dimention as dis
	x = np.arange(0,dis.shape[0],1.) 	# x coordinate ranged from 0 to size of displacement coordinate
	y = np.arange(0,dis.shape[1],1.) 	# y coordinate ranged from 0 to size of displacement coordinate
	field[:,:,0] , field[:,:,1] = np.meshgrid(x, y) #fill elements in field with x,y coordinate
	field = field + dis #add displacment to coordinate
	return field
	
def Cdist(dis): #from displacment vector calculate distortion tensor element u_r_r, u_r_z,u_z_r, and u_z_z
	#in order to calculate strain, differential of displacment vector must be calculate first
	size_x = dis.shape[0]
	size_y = dis.shape[1]
	zox = np.zeros((size_x,1))	#used to fill in the first column when differential happens
	zoy = np.zeros((1,size_y))
	u_r = dis[:,:,0]	 #displacment r, a 2 by 2 matrix, namely, r and z 
	u_z = dis[:,:,1]	 #displacment z, a 2 by 2 matrix, namely, r and z
	#differential over different axis, assuming the symmetry at r=0. this will resulting the differential at r = 0 is zero
	u_r_r = np.append(zox,np.diff(u_r,axis = 1),axis = 1) #differenntial over axis = 1, u_r differential over r
	u_r_z = np.append(zoy,np.diff(u_r,axis = 0),axis = 0) #differential over axis = 0, u_r differential over z
	u_z_r = np.append(zox,np.diff(u_z,axis = 1),axis = 1)
	u_z_z = np.append(zoy,np.diff(u_z,axis = 0),axis = 0)
	return u_r_r, u_r_z, u_z_r, u_z_z

def Strain(dis):
	u_r_r, u_r_z, u_z_r, u_z_z = Cdist(dis) #get distortion tensor
	r_rec = np.append(1,np.reciprocal(np.arange(1,u_r_r.shape[1],1.)))#the matrix of 1 over r, used in strain calculation
	e = np.zeros((u_r_r.shape[0],u_r_r.shape[1],4)) #strain matrix contains e_rr,e_phiphi,e_zz,e_rz,other element in strian tensor is zero by boundary conditions
	e[:,:,0] = u_r_r #e_rr
	e[:,:,1] = np.multiply(r_rec,dis[:,:,0])#e_phiphi
	e[:,:,2] = u_z_z #e_zz
	e[:,:,3] = np.divide(u_r_z + u_z_r,2) #e_rz = e_zr
	#phiz and rphi component is zero, because there is no displacement along phi direction
	return e

def EE(e,lab,miu): #from strain matrix,lambda and miu(lame constant) calculate strain energy
	e_ii2 = np.add(np.square(e[:,:,0]),np.square(e[:,:,1]),np.square(e[:,:,2]))
	e_ik2 = np.multiply(2,np.square(e[:,:,3]))
	dE = np.add(np.multiply( lab/2, e_ii2) , np. multiply(miu, e_ik2))#energy of each element
	E = np.sum(dE)
	return E #returning the total energy

def rad(dis,S): #gengerating a random matrix, and a random sequence 
	#S is the function defined by Const. * exp(-1/T)
	dL = np.multiply(S, np.random.exponential(1,np.append(1000,dis.shape)))
	dL = np.multiply(np.random.choice([-1,0,0,0,0,0,0,0,1],np.append(1000,dis.shape)),dL)
	Se = np.random.randint(1000, size = 1000)
	
	return dL, Se

def vari(dis,dN,lab,miu): #give a fluctuation over the entire displacement matrix,returns new value if energy is decreased. retruns old value if energy is increased
	dis_1 = dis + dN #add variation
	dis = BC(dis)
	dis_1 = BC(dis_1)
	E_0 = Strain(dis)
	E_1 = Strain(dis_1)
	#print EE(E_0, lab, miu)
	#print "-", EE(E_1, lab, miu)
	if EE(E_0, lab, miu) >= EE(E_1, lab, miu):#if energy decreased, choose the new value
		return dis_1
		
	else:
		return dis

def BC(dis): #check boundary conditions
	dis[0,:]=[0,0] #no displacement at z=0
	dis[:,0,0] = 0 #no r direction displacement at r=0
	dis[:,dis.shape[1] - 1,0] = 0 #no r direction displacement at r = r_max
	bc = np.add(-2,np.multiply(0.01,np.square(np.arange(0,dis.shape[1]-1,1.)))) #bundary condition of a parabolic function
	for idx,bc_ in enumerate(bc): #check boundary condition
		if dis[dis.shape[0]-1,idx,1] > bc_:
			dis[dis.shape[0]-1,idx,1] = bc_
	for idx,bc_ in enumerate(bc):
		if dis[dis.shape[0]-2,idx,1] > bc_ +1:
			dis[dis.shape[0]-2,idx,1] = bc_+1
	for idx,bc_ in enumerate(bc):
		if dis[dis.shape[0]-3,idx,1] > bc_ +2:
			dis[dis.shape[0]-3,idx,1] = bc_+2
	return dis

def anneal(dis):
	lab = 0.015
	miu = 0.01
	S = [0.0001,0.00002,0.00001, 0.00001, 0.00001] #from temperture high to low,
	rad_ = np.zeros(np.append([5,1000],dis.shape)) #there are 4 times 50 times dis.shape number of element in this matrix
	rad_id = np.zeros([5,1000])
	for idx,Si in enumerate(S): #generating annealing matrix
		rad_[idx], rad_id[idx] = rad(dis,Si)
		
	for i in range(0,5):
		for j in range(0,5):
			for k in rad_id[j]:
				dis = vari(dis,rad_[i,k],lab,miu)
		print "iteration,"	,i
		print "E = ", EE(Strain(dis),lab,miu)
		#Ess = np.append(EE(Strain(dis),lab,miu),Ess)	#recording change of E
	
	E_0 = EE(Strain(dis),lab,miu)
	
	print "new iteration"
	np.random.shuffle(rad_id)
	
	for i in range(0,5):
		for j in range(0,5):
			for k in rad_id[j]:
				dis = vari(dis,rad_[i,k],lab,miu)
		print "iteration,"	,i
		print "E = ", EE(Strain(dis),lab,miu)		
		#Ess = np.append(EE(Strain(dis),lab,miu),Ess)	#recording change of E
		
	E_1 = EE(Strain(dis),lab,miu)
	
	n=0
	
	while E_1 < E_0 and n < 3:
		np.random.shuffle(rad_id)		
		print "new iteration"
		for i in range(0,5):
			for j in range(0,5):
				for k in rad_id[j]:
					dis = vari(dis,rad_[i,k],lab,miu)
			print "iteration,"	,i
			print "E = ", EE(Strain(dis),lab,miu)
			#Ess = np.append(EE(Strain(dis),lab,miu),Ess) #recording change of E
		E_1 = EE(Strain(dis),lab,miu)
		n+=1
		
	return E_1,dis

def draw(dis):
	pos = Position(dis) #get position mesh
	
	for i in range(0,pos.shape[0],1):
		plt.plot(pos[:,i,0],pos[:,i,1],'r-')
	for j in range(0,pos.shape[1],1):
		plt.plot(pos[j,:,0],pos[j,:,1],'r-')
	x = dis.shape[0]-1
	y = dis.shape[1]-1
	plt.plot(pos[:,y,0],pos[:,y,1],'b-')
	plt.plot(pos[x,:,0],pos[x,:,1],'b-')
	plt.axis([0,x+20,0,y+20])
	plt.show()


sam = InitialC(30,30)
Ex = np.zeros(20)
for i in range(0,20):
	Ex[i],sam = anneal(sam)
#print sam
#sam = BC(sam)
draw(sam)
Ex = Ex[:-1]
plt.plot(Ex)
plt.show()
