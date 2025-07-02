import numpy as np
import sympy as sp

# Physical parameters
g = 9.8

mb = 5.8  # body mass
mL0 = 0.15  # link0 mass
mL1 = 0.15  # link1 mass
mL2 = 0.15  # link2 mass
mL3 = 0.15  # link3 mass
mL = 0.3 # load mass (gripper + target object)

d = 0.6  # length of the rotors' arm
d0 = 0.1  # length of the manipulator link 0
d1 = 0.2  # length of the manipulator link 1
d2 = 0.2  # length of the manipulator link 2
d3 = 0.1  # length of the manipulator link 3

# Quadrotor moments of inertia
Ixx = 0.327
Iyy = 0.327
Izz = 0.654
Ib = np.diag([Ixx, Iyy, Izz])

# Links' moments of inertia
Ix0, Iy0, Iz0 = 0, 0.97e-03, 0.97e-03
IL0_ = np.diag([Ix0, Iy0, Iz0])

Ix1, Iy1, Iz1 = 0, 0.97e-03, 0.97e-03
IL1 = np.diag([Ix1, Iy1, Iz1])

Ix2, Iy2, Iz2 = 0, 0.97e-03, 0.97e-03
IL2 = np.diag([Ix2, Iy2, Iz2])

Ix3, Iy3, Iz3 = 0, 0.97e-03, 0.97e-03
IL3 = np.diag([Ix3, Iy3, Iz3])

gamma_i = 0.05  # Propulor thrust/torque constant (m)

# Drag coeffs
k_vx = 0.35
k_vy = 0.35
k_vz = 0.45

rad = np.pi/180
# Constraints 
#max_thrust = 30.0  # Max thrust for each motor (N)
#min_thrust = 0.0   # Min thrust (N)
max_F = 120 # Max total Thrust 
max_tau_rp = 5
max_tau_y = 1 
max_tau_1 = 2 # Max Torque for servomotor of arm 1 (N.m)
max_tau_2 = 1 # Max Torque for servomotor of arm 2 (N.m)
max_tau_3 = 0.5 # Max Torque for servomotor of arm 3 (N.m)
min_z = -10
max_z = 0
max_h_pos = 30
max_phi_theta = 70*rad # rad
max_psi = 180*rad # rad
#max_n1 = 90*rad + max_phi_theta # rad
min_n1 = -180*rad
max_n1 = 180*rad
max_n2_n3 = 180*rad # rad

# Target Point
target_point = (10, -2, -1)
final_point = (20, 5, -2)
obstacles = [(3, -4, 0.2*3), (5, 1, 0.2*3), (15, 5, 0.2*3), (15, -2, 0.3*3), (19, 0, 0.3*3)]  # (x, y, radius)

parameters = {
    "g" : g,
    "mb" : mb,
    "mL0" : mL0,
    "mL1" : mL1,
    "mL2" : mL2,
    "mL3" : mL3,
    'mL' : mL,
    "d" : d,
    "d0" : d0,
    "d1" : d1,
    "d2" : d2,
    "d3" : d3,
    "Ixx" : Ixx,
    "Iyy" : Iyy,
    "Izz" : Izz,
    "Ix0" : Ix0,
    "Iy0" : Iy0,
    "Iz0" : Iz0,
    "Ix1" : Ix1,
    "Iy1" : Iy1,
    "Iz1" : Iz1,
    "Ix2" : Ix2,
    "Iy2" : Iy2,
    "Iz2" : Iz2,
    "Ix3" : Ix3,
    "Iy3" : Iy3,
    "Iz3" : Iz3,
    "gamma_i" : gamma_i,
    "k_vx" : k_vx, 
    "k_vy" : k_vy, 
    "k_vz" : k_vz,
    "max_F" : max_F, 
    "max_tau_rp" : max_tau_rp,
    "max_tau_y" : max_tau_y,
    "max_tau_1" : max_tau_1,
    "max_tau_2" : max_tau_2,
    "max_tau_3" : max_tau_3,
    "min_z" : min_z,
    "max_z" : max_z,
    "max_h_pos" : max_h_pos,
    "max_phi_theta" : max_phi_theta,
    "max_psi" : max_psi,
    "min_n1" : min_n1,
    "max_n1" : max_n1,
    "max_n2_n3" : max_n2_n3,
    "target_point" : target_point,
    "final_point" : final_point,
    "obstacles" : obstacles,
}

