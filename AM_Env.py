"""
Aerial Manipulator Simulation Environment
@authors: Mahdi Shahrajabian, Hossein Otroushi
March 2024
"""
import sys
from pathlib import Path

proj_path = Path.cwd().resolve().parent
sys.path.append(str(proj_path))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gym import spaces
from IPython.display import Video, display
import cv2

from DynamicAerialManipulator.parameters_Aerial_Manipolator import parameters
from DynamicAerialManipulator.get_Gq import get_Gq
from DynamicAerialManipulator.get_Mq import get_Mq
from DynamicAerialManipulator.get_Cq import get_Cq
from DynamicAerialManipulator.get_metrices import get_Q_matrix, get_R_IB, get_T_matrix, HTM, H2RP


class AerialManipulatorEnv:
    def __init__(self, parameters:dict,initial_state=None, dt=0.01):
        rad = np.pi/180
        # parameter of the system like: mass,length,inersia,...
        self.parameters = parameters
        if initial_state is not None:
            self.initial_state = initial_state
        # Simulation parameters 
        self.dt = dt  # Time step (s)
        self.t = 0.0
        self.time = []
        self.target_point = self.parameters["target_point"]
        self.final_point = self.parameters["final_point"]
        self.p_g = self.target_point
        self.target_flag = False
        self.final_flag = False

        """
        States -> [q, qDot] (18,1)
        q ->[x, y, z, phi, theta, psi, n1, n2, n3] 
        qDot-> [xDot, yDot, zDot, phiDot, thetaDot, psiDot, n1Dot, n2Dot, n3Dot]

        x, y, z -> position; phi, theta, psi -> attitude (roll, pitch, yaw); n1, n2, n3 -> angle of arms
        xDot, yDot, zDot -> translational velocities; phiDot, thetaDot, psiDot -> angular velocities of euler; n1Dot, n2Dot, n3Dot -> angular velocities of            arms
        """
        
        self.p_ee = (0,0,0) # end-effector postion  
        self.p_ee_prev = (0,0,0) 

        # Constraints 
        #max_thrust = self.parameters["max_thrust"]  # Max thrust for each motor (N)
        max_F = self.parameters["max_F"]  # Max total thrust 
        max_tau_rp = self.parameters["max_tau_rp"]  # Max roll and pitch torques 
        max_tau_y = self.parameters["max_tau_y"]  # Max yaw torque
        #min_thrust = self.parameters["min_thrust"]   # Min thrust (N)
        max_tau_1 = self.parameters["max_tau_1"] # Max Torque for servomotor of arm 1 (N.m)
        max_tau_2 = self.parameters["max_tau_2"] # Max Torque for servomotor of arm 2 (N.m)
        max_tau_3 = self.parameters["max_tau_3"] # Max Torque for servomotor of arm 3 (N.m)
        min_z = self.parameters["min_z"]
        max_z = self.parameters["max_z"]
        max_h_pos = self.parameters["max_h_pos"]
        max_phi_theta = self.parameters["max_phi_theta"]
        max_psi = self.parameters["max_psi"]
        min_n1 = self.parameters["min_n1"]
        max_n1 = self.parameters["max_n1"]
        max_n2_n3 = self.parameters["max_n2_n3"]

        self.low_state = np.array([
            -max_h_pos, -max_h_pos, min_z, -max_phi_theta, -max_phi_theta, -max_psi, min_n1, -max_n2_n3, -max_n2_n3,
            -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max,
            -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max, -np.finfo(np.float32).max]
            , dtype=np.float32)
        
        self.high_state = np.array([
            max_h_pos, max_h_pos, max_z, max_phi_theta, max_phi_theta, max_psi, max_n1, max_n2_n3, max_n2_n3,
            np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
            np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max]
            , dtype=np.float32)
        
        self.high_obs = np.concatenate((self.high_state,
                                        np.array([max_h_pos, max_h_pos, max_z], dtype=np.float32), 
                                        np.array([np.linalg.norm([max_h_pos, max_h_pos]), np.pi], dtype=np.float32))) 
        self.low_obs = np.concatenate((self.low_state,
                                        np.array([-max_h_pos, -max_h_pos, min_z], dtype=np.float32), 
                                        np.array([0, -np.pi], dtype=np.float32))) 

        self.min_action = np.array([0, -max_tau_rp, -max_tau_rp, -max_tau_y,
                                    -max_tau_1, -max_tau_2, -max_tau_3], dtype=np.float32)
        
        self.max_action = np.array([max_F, max_tau_rp, max_tau_rp, max_tau_y,
                                    max_tau_1, max_tau_2, max_tau_3], dtype=np.float32)
        
        
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=self.low_obs,
            high=self.high_obs,
            dtype=np.float32)

        # Initial Condition Range 
        self.initial_min_vals = np.array([-1, -1, -2, -5*rad, -5*rad, -5*rad, 85*rad, -5*rad, -5*rad,
            -0.2, -0.2, -0.2, -2*rad, -2*rad,-2*rad, -2*rad, -2*rad, -2*rad], dtype=np.float32)
        self.initial_max_vals = np.array([1, 1, -1, 5*rad, 5*rad, 5*rad, 95*rad, 5*rad, 5*rad,
            0.2, 0.2, 0.2, 2*rad, 2*rad,2*rad, 2*rad, 2*rad, 2*rad], dtype=np.float32)
        
        # Obstacles (two first elements are the centers position in x-y plane and the third is the radius)
        self.obstacles = self.parameters["obstacles"] # (x, y, radius)

        
    def _wrap_angle(self,angle):
        """Wrap angle between -pi and pi"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _generate_disturbance(self, t, state):
        # generate an external disturbance
        k_vx, k_vy, k_vz = self.parameters["k_vx"], self.parameters["k_vy"], self.parameters["k_vz"]
        R_IB = get_R_IB(state[3], state[4], state[5], symbolic=False)
        R_BI = R_IB.T  # rotation matrix from Inersia to Body 
        
        v_B = R_BI @ np.block([state[9], state[10], state[11]]).reshape(-1,1) #(3,1)
        vx_B , vy_B, vz_B = v_B

        fDrag_B = -np.block([k_vx*vx_B, k_vy*vy_B, k_vz*vz_B]).reshape(-1,1) #(3,1)
        fDrag_I = R_IB @ fDrag_B 
        return np.block([fDrag_I.flatten(), np.zeros((1,6))]).reshape(-1,1) #(9,1)
        
    def denormalize_action(self, action):
        """
        Denormalizes an action from the range [-1, 1] back to the original range [action_min, action_max].
        """ 
        return (action + 1) * (self.max_action - self.min_action) / 2 + self.min_action
    
    def _dynamics(self, t, state, inputs):
        q = state[:9].reshape(-1, 1)  # (9, 1)
        qDot = state[9:].reshape(-1, 1)  # (9, 1)
        
        # X-config 
        #  forward
        #  4↓     1↑
        #   \    /
        #    \  / 
        #    / \ 
        #   /   \
        #  3↑     2↓
        # T1, T2, T3, T4, tau_n1, tau_n2, tau_n3 = inputs  # Motor thrust and servomotor Torque inputs
        # rad = np.pi/180
        # d = self.parameters['d']
        # gamma_i = self.parameters['gamma_i']
        # ld = d * np.cos(45*rad)

        # total_thrust_B = (T1 + T2 + T3 + T4)
        # tau_phi_B = ld * (T3 + T4 - T1 - T2)
        # tau_theta_B = ld * (T1 + T4 - T2 - T3)
        # tau_psi_B = gamma_i * (T1 - T2 + T3 - T4)
        
        total_thrust_B, tau_phi_B, tau_theta_B, tau_psi_B, tau_n1, tau_n2, tau_n3 = inputs  # virtual controls and servomotor Torque inputs

        tau_B = np.array([total_thrust_B, tau_phi_B, tau_theta_B, tau_psi_B, tau_n1, tau_n2, tau_n3]).reshape(-1, 1) #(7,1)

        
        R_IB = get_R_IB(q[3], q[4], q[5], symbolic=False) # rotation matrix round XYZ from body to inersia frame
        Q = get_Q_matrix(q[3], q[4], q[5], symbolic=False) # w_b = (p, q, r)' = Q*(phiDot, thetaDot, psiDot)'

        e3 = np.array([0, 0, 1]).reshape(-1, 1)
        inv_Q = np.linalg.inv(Q)

        Generalized_tau = np.block([
                        [       -R_IB @ e3, np.zeros((3, 3)), np.zeros((3, 3))],  
                        [np.zeros((3, 1)),            inv_Q, np.zeros((3, 3))],  
                        [np.zeros((3, 1)), np.zeros((3, 3)),        np.eye(3)],  
                        ]) @ tau_B

        # generate an external disturbance
        # tau_ext = self._generate_disturbance(t, self.state)
        tau_ext = np.zeros((9,1))

        # Dynamics: Mq*q2Dot + Cq*qDot + Gq = tau + tau_ext
        Gq = get_Gq(q, qDot, self.parameters)
        Mq = get_Mq(q, qDot, self.parameters)
        Cq = get_Cq(q, qDot, self.parameters)

        Mq_inv = np.linalg.inv(Mq)
        F = -Mq_inv @ Cq @ qDot - Mq_inv @ Gq
        B = Mq_inv

        dq1dt = qDot
        dq2dt = F + B @ (Generalized_tau + tau_ext)

        # Combine into a single state derivative vector
        return np.concatenate([dq1dt, dq2dt], axis=0).flatten()

    def _runge_kutta(self, state, inputs, t):
        """Runge-Kutta 4th order integration."""
        k1 = self._dynamics(t, state, inputs)
        k2 = self._dynamics(t, state + 0.5 * self.dt * k1, inputs)
        k3 = self._dynamics(t, state + 0.5 * self.dt * k2, inputs)
        k4 = self._dynamics(t, state + self.dt * k3, inputs)

        new_state = state + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return new_state

    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        if self.initial_state is None:
            self.state = np.random.uniform(low=self.initial_min_vals, high=self.initial_max_vals) 
        else:
            self.state = self.initial_state
        self.t = 0.0
        self.time = []
        self.p_g = self.target_point
        self.target_flag = False
        self.final_flag = False
        self.state_prev = self.state
        self.p_ee = self._get_pos_ee()
        self.p_ee_prev = self.p_ee
        obs = self.state
        return obs

    def step(self, inputs):
        """Take a step in the environment."""

        # inputs = self.denormalize_action(inputs) # Denormalize actions if you used RL algorithms with inputs in the range [-1 1] 
        
        # Clip inputs to action space bounds
        #inputs = np.clip(inputs, self.min_action, self.max_action)
        self.state_prev = self.state
        self.p_ee_prev = self._get_pos_ee()
        self.state = self._runge_kutta(self.state, inputs, self.t)

        # Wrap angles to the range [-pi, pi] for (phi,theta,psi,n1,n2,n3)
        self.state[3:9] = np.vectorize(self._wrap_angle)(self.state[3:9])
        self.p_ee = self._get_pos_ee()
        # Clip states to observation space bounds
        #self.state = np.clip(self.state, self.low_state, self.high_state)

        self.t += self.dt
        self.time.append(self.t)
        obs = self.state
        done = self._done_fun()
        return obs, done

        
    def _get_pos_ee(self):
        H_b0 = HTM(0, self.parameters['d0'], 0, np.pi / 2)
        H_01 = HTM(self.state[6], 0, self.parameters['d1'], 0)
        H_12 = HTM(self.state[7], 0, self.parameters['d2'], 0)
        H_2e = HTM(self.state[8], 0, self.parameters['d3'], 0)
        
        H_be = H_b0 @ H_01 @ H_12 @ H_2e
        R_be, p_be = H2RP(H_be)
        R_IB = get_R_IB(self.state[3], self.state[4], self.state[5], symbolic=False)
        p_ee = (self.state[0:3].reshape(-1,1) + R_IB @ p_be).flatten()
        return p_ee

    def _done_fun(self):
        # Can add reward too 
        done = False
        
        # Infeasible states 
        for i in range(len(self.state)):
            if self.state[i] > self.high_state[i]:
                self.state[i] = self.high_state[i]
                done = True
                break
            if self.state[i] < self.low_state[i]:
                self.state[i] = self.low_state[i]
                done = True
                break
                
        # Collision detection with obstacles 
        if not done:
            for i in range(len(self.obstacles)):
                obs = self.obstacles[i]
                if np.linalg.norm(obs[:2] - self.state[:2]) - obs[2] < 1.25*self.parameters['d']:
                    done = True
                    break
                      
        # Ground Contact 
        if not done:
            if np.absolute(self.state[2])<0.6:
                done = True

        return done