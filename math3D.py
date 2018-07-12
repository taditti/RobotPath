import numpy as np
from math import sqrt, asin, sin, cos

def angle_axis(R):
    u=np.zeros(3)
    u[0]=R[2,1]-R[1,2]
    u[1]=R[0,2]-R[2,0]
    u[2]=R[1,0]-R[0,1]
    mag=sqrt(u[0]**2+u[1]**2+u[2]**2)
    u=u/mag
    angle=asin(mag/2)

    return angle, u

def rotation_matrix(t, u):
    R=np.array([[cos(t)+u[0]**2*(1-cos(t)),         u[0]*u[1]*(1-cos(t))-u[2]*sin(t),   u[0]*u[2]*(1-cos(t))+u[1]*sin(t)],
               [u[0]*u[1]*(1-cos(t))+u[2]*sin(t),   cos(t)+u[1]**2*(1-cos(t)),          u[1]*u[2]*(1-cos(t))-u[0]*sin(t)],
               [u[0]*u[2]*(1-cos(t))-u[1]*sin(t),   u[1]*u[2]*(1-cos(t))+u[0]*sin(t),   cos(t)+u[2]**2*(1-cos(t))       ]])
    return R

def q_from_R(matrix):
    m = matrix.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
    else:
        if m[0, 0] < -m[1, 1]:
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

    q = np.array(q)
    q *= 0.5 / sqrt(t);
    return q
