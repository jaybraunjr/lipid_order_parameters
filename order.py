import MDAnalysis as mda
import numpy as np
import math

u = mda.Universe('1.20.gro')
a = u.select_atoms('name C3')
b = u.select_atoms('name C13')

print('number of frames:',len(u.trajectory))

# magnitude of vector
def magnitude(a,b):
	v = a-b
	v = v*v
	sum = np.sum(v)
	return math.sqrt(sum)



N = len(a)
sum = 0

for j in range(N):
	if (a[j].resid != b[j].resid):
		assert(0)
	mag = magnitude(a[j].position,b[j].position)
	c_theta = (a[j].position[2]-b[j].position[2])/mag
	#print(c_theta)
	sum += (3*(c_theta*c_theta)-1)/2

sum /= N
print('Order parameter:',sum)
