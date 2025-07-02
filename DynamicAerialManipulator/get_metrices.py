import sympy as sp
import numpy as np

def get_R_IB(phi, theta, psi, symbolic=True):
    if symbolic == True:
        # Rotation matrices
        Rx = sp.Matrix([
            [1, 0, 0],
            [0, sp.cos(phi), -sp.sin(phi)],
            [0, sp.sin(phi), sp.cos(phi)]
        ])

        Ry = sp.Matrix([
            [sp.cos(theta), 0, sp.sin(theta)],
            [0, 1, 0],
            [-sp.sin(theta), 0, sp.cos(theta)]
        ])

        Rz = sp.Matrix([
            [sp.cos(psi), -sp.sin(psi), 0],
            [sp.sin(psi), sp.cos(psi), 0],
            [0, 0, 1]
        ])

        R_IB = Rz @ Ry @ Rx  # rotation matrix round XYZ from body to inersia frame
    else:
        phi = phi.reshape(())
        theta = theta.reshape(())
        psi = psi.reshape(())
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

        R_IB = Rz @ Ry @ Rx  # rotation matrix round XYZ from body to inersia frame       


    return R_IB


def get_Q_matrix(phi, theta, psi, symbolic=True):
    """ 
    To transform the Euler angles rate from the inertial frame ( phiDot, thetaDot, psiDot ) 
    to angular velocity components in the body-fixed frame (p,q,r) for a rotation sequence XYZ (from body to inersia),
    the transformation matrix Q relates these quantities:
    wb_B = (p, q, r)' = Q*(phiDot, thetaDot, psiDot)'   
    """
    if symbolic == True:
        Q = sp.Matrix([
            [1, 0, -sp.sin(theta)],
            [0, sp.cos(phi), sp.cos(theta) * sp.sin(phi)],
            [0, -sp.sin(phi), sp.cos(theta) * sp.cos(phi)]
        ])
        
    else:
        phi = phi.reshape(())
        theta = theta.reshape(())
        psi = psi.reshape(())

        Q = np.array([
            [1, 0, -np.sin(theta)],
            [0, np.cos(phi), np.cos(theta) * np.sin(phi)],
            [0, -np.sin(phi), np.cos(theta) * np.cos(phi)]
    ])
    
    return Q 


def get_T_matrix(phi, theta, psi, symbolic=True):
    """ 
    To transform the Euler angles rate from the inertial frame ( phiDot, thetaDot, psiDot ) 
    to angular velocity components in the inersia frame (wx,wy,wz) for a rotation sequence XYZ (from body to inersia),
    the transformation matrix T relates these quantities:
    wb_I = (wx,wy,wz)' = T*(phiDot, thetaDot, psiDot)' 
    wb_I = (wx,wy,wz)' = R_IB*wb_B = R_IB*(p, q, r)'  

    """

    if symbolic == True:
        R_IB = get_R_IB(phi, theta, psi, symbolic=True)
        Q = get_Q_matrix(phi, theta, psi, symbolic=True)
        T = sp.simplify(R_IB @ Q)
    else:
        phi = phi.reshape(())
        theta = theta.reshape(())
        psi = psi.reshape(())
        
        R_IB = get_R_IB(phi, theta, psi, symbolic=False)
        Q = get_Q_matrix(phi, theta, psi, symbolic=False)
        T = R_IB @ Q

    return T

def HTM(theta, d, a, alpha):
    """
    Homogeneous Transformation Matrix (HTM) using NumPy
    
    Args:
        theta: Rotation angle about z-axis (in radians).
        d: Translation along z-axis.
        a: Translation along x-axis.
        alpha: Rotation angle about x-axis (in radians).

    Returns:
        H: Homogeneous Transformation Matrix (4x4 NumPy array)
    """
    # Rotation about z-axis
    Rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])
    
    # Translation along z-axis
    Trans_z = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])
    
    # Translation along x-axis
    Trans_x = np.array([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Rotation about x-axis
    Rot_x = np.array([
        [1, 0,           0,            0],
        [0, np.cos(alpha), -np.sin(alpha), 0],
        [0, np.sin(alpha),  np.cos(alpha), 0],
        [0, 0,           0,            1]
    ])
    
    # Homogeneous Transformation Matrix
    H = Rot_z @ Trans_z @ Trans_x @ Rot_x  # Matrix multiplication using @
    
    return H


def H2RP(H):
    """
    Extract the rotation matrix and position vector from a Homogeneous Transformation Matrix.
    
    Parameters:
    H (numpy.ndarray): A 4x4 homogeneous transformation matrix.
    
    Returns:
    tuple:
        - R (numpy.ndarray): A 3x3 rotation matrix.
        - P (numpy.ndarray): A 3x1 position vector.
    """
    # Ensure H is a 4x4 matrix
    H = np.reshape(H, (4, 4))
    
    # Extract the rotation matrix (upper-left 3x3 submatrix)
    R = H[:3, :3]
    
    # Extract the position vector (first three entries of the fourth column)
    P = H[:3, 3].reshape(3, 1)
    
    return R, P
