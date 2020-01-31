import numpy as np
import matplotlib.pyplot as plt

def getQ(x_arr, q_arr):
    # create vectors for charge locations
    n_charges = len(q_arr)
    q = q_arr
    x0, x1, x2, x3 = x_arr

    x = np.matrix([x0,x1,x2,x3])
    # create vector for Q
    Q = np.zeros((3,3))

    # calculate Q
    for i in range(n_charges):
        Q += q[i]*(3*x[i].T*x[i] - np.linalg.norm(x[i])**2*np.identity(3))

    # get rid of error
    for i in range(len(Q)):
        for j in range(len(Q[i])):
            if abs(Q[i][j]) < 1e-10:
                Q[i][j] = 0
    return Q

def getPhi(Q):
    x = np.linspace(-2,2,100)
    y = np.linspace(-2,2,100)
    s = []
    y_start = 0
    # Get all unique vectors to all (x,y) points
    for i in x:
        for j in y[y_start:]:
            s.append([i, j, 0])
            if (i != j):
                s.append([j, i, 0])
        y_start+=1
    r = np.array(s)
    z = np.meshgrid(x, y)
    print(z)
    print(r)
    print(len(r))
    print(r.T)
    #print(r)
    #print(r.T)
    # still not right format here
    # maybe take this line below and put into a vectorized function
    phi = calc_phi(r)
    plt.contour(phi)
    #return phi
#@np.vectorize
def calc_phi(r):
    phi = np.empty(shape = len(r))
    for i in range(len(r)):
        phi[i] = (np.matmul(np.matmul(r[i].T,Q),r[i]))/(2*np.linalg.norm(r[i])**5)
    print(phi)
    return phi

# define l
l = 1

# fill x with charge locations
n_charges = 4
q = np.zeros((n_charges))
x0 = np.array([0, 0, 0])
q[0] = -1
x1 = np.array([l, 0, 0])
q[1] = 1
x2 = np.array([l, l, 0])
q[2] = -1
x3 = np.array([0, l, 0])
q[3] = 1
x = np.array([x0,x1,x2,x3])

# calculate tensor
Q = getQ(x, q)

# calculate potential
phi = getPhi(Q)
print(phi)

print("The quadrupole tensor for the distribution in 2.13 is:")
print(Q)


n_charges = 4
q = np.zeros((n_charges))
x0 = np.array([l/2, 0, 0])
q[0] = 1
x1 = np.array([0, l/2, 0])
q[1] = -1
x2 = np.array([-l/2, 0, 0])
q[2] = 1
x3 = np.array([0, -l/2, 0])
q[3] = -1
x = np.array([x0,x1,x2,x3])

# calculate tensor
Q = getQ(x, q)
print("The quadrupole tensor for the distribution in 2.15 is:")
print(getQ(x, q))
