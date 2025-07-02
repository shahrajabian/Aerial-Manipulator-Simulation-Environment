import numpy as np
from math import * 

def get_Gq(q, qDot, params):

	g, mb, mL0, mL1, mL2, mL3, d, d0, d1, d2, d3 = \
		params['g'], params['mb'], params['mL0'], params['mL1'], params['mL2'], params['mL3'], params['d'], params['d0'], params['d1'], params['d2'], params['d3']
	Ixx, Iyy, Izz, Ix0, Iy0, Iz0, Ix1, Iy1, Iz1, Ix2, Iy2, Iz2, Ix3, Iy3, Iz3 = \
		params['Ixx'], params['Iyy'], params['Izz'], params['Ix0'], params['Iy0'], params['Iz0'], params['Ix1'], params['Iy1'], params['Iz1'], params['Ix2'], params['Iy2'], params['Iz2'], params['Ix3'], params['Iy3'], params['Iz3']
	X, Y, Z, phi, theta, psi, n1, n2, n3 = \
		q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8] 
	XDot, YDot, ZDot, phiDot, thetaDot, psiDot, n1Dot, n2Dot, n3Dot = \
		qDot[0], qDot[1], qDot[2], qDot[3], qDot[4], qDot[5], qDot[6], qDot[7], qDot[8] 
	Gq = np.zeros((9,1)) 

	x0 = cos(theta)
	x1 = d0*mL0
	x2 = 2*d0
	x3 = sin(n1)
	x4 = d1*x3
	x5 = x2 + x4
	x6 = 2*x4
	x7 = n1 + n2
	x8 = sin(x7)
	x9 = d2*x8
	x10 = x6 + x9
	x11 = x10 + x2
	x12 = n3 + x7
	x13 = sin(x12)
	x14 = d3*x13 + 2*x9
	x15 = x14 + x6
	x16 = x15 + x2
	x17 = g/2
	x18 = sin(theta)
	x19 = cos(phi)
	x20 = x18*x19
	x21 = cos(n1)
	x22 = d1*x21
	x23 = 2*x22
	x24 = cos(x7)
	x25 = d2*x24
	x26 = x0*(x23 + x25)
	x27 = cos(x12)
	x28 = d3*x27 + 2*x25
	x29 = x0*(x23 + x28)
	x30 = x0*x19

	Gq[0,0] = 0 
	Gq[1,0] = 0 
	Gq[2,0] = -g*(mL0 + mL1 + mL2 + mL3 + mb) 
	Gq[3,0] = x0*x17*(mL1*x5 + mL2*x11 + mL3*x16 + x1)*sin(phi) 
	Gq[4,0] = x17*(mL1*(x0*x22 + x20*x5) + mL2*(x11*x20 + x26) + mL3*(x16*x20 + x29) + x1*x20) 
	Gq[5,0] = 0 
	Gq[6,0] = -x17*(d1*mL1*(x18*x3 + x21*x30) + mL2*(x10*x18 + x19*x26) + mL3*(x15*x18 + x19*x29)) 
	Gq[7,0] = -x17*(d2*mL2*(x18*x8 + x24*x30) + mL3*(x14*x18 + x28*x30)) 
	Gq[8,0] = -d3*mL3*x17*(x13*x18 + x27*x30) 


	return Gq