#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag


#%% Functions
def quat_mul(a,b):
    u0=a[3]
    v0=b[3]
    u=a[0:3].reshape(3,)
    v=b[0:3].reshape(3,)

    value=u0*v0-np.matmul(u.T,v)
    axis=u0*v+v0*u + np.cross(u,v) 
    quaternion=np.append(axis,value)

    return quaternion

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    R=q.as_matrix()
    q=q.as_quat()
    new_p = np.zeros((3, 1))
    new_v = np.zeros((3, 1))
    new_q = Rotation.identity()

    A=a_m-a_b
    W=w_m-w_b
    new_p= p +v*dt +( 0.5*(np.matmul(R, A)+ g)*(dt**2) ).reshape(3,1)
    new_v=v+((np.matmul(R, A)+g)*dt).reshape(3,1)

    dc=W*dt
    dcq=Rotation.from_rotvec(dc.flatten()).as_quat()
    new_q=quat_mul(q,dcq)
    new_q=Rotation.from_quat(new_q)
    

    return new_p, new_v, new_q, a_b, w_b, g



def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """
    

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    I=np.identity(3)
    R=q.as_matrix()
    
     
    A=(a_m-a_b).flatten()
    W=(w_m-w_b).flatten()
    dc=W*dt
    dcq=Rotation.from_rotvec(dc.flatten()).as_matrix()

    Fx=np.identity(18)
    Fx[:3,3:6]=I*dt
    Fx[3:6, 6:9]=-R@ np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])*dt
    Fx[3:6, 9:12]=-R*dt
    Fx[3:6, 15:18]= I*dt
    Fx[6:9, 6:9]= dcq.T
    Fx[6:9, 12:15]= -I*dt
    
    Vi=(accelerometer_noise_density**2) * (dt**2) * I
    thetai=(gyroscope_noise_density**2) * (dt**2) *I
    Ai= accelerometer_random_walk**2 *dt*I
    omegai=gyroscope_random_walk**2*dt*I

    Qi=np.zeros((12,12))
    Qi[:3, :3]= Vi
    Qi[3:6, 3:6]= thetai
    Qi[6:9, 6:9]=Ai
    Qi[9:12, 9:12]=omegai

    Fi=np.zeros((18,12))
    Fi[3:6, :3]=I
    Fi[6:9, 3:6]=I
    Fi[9:12, 6:9]=I
    Fi[12:15, 9:12]=I

    FPF=np.matmul(Fx, np.matmul(error_state_covariance, Fx.T))
    FQF=np.matmul(Fi,np.matmul(Qi, Fi.T))
    error_state_covariance=FPF+FQF

    # return an 18x18 covariance matrix
    return error_state_covariance


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    p, v, q, a_b, w_b, g = nominal_state
    innovation = np.zeros((2, 1))

    R = q.as_matrix()
    q = q.as_quat()
    Pc = np.matmul(R.T, (Pw - p))
    Zc=Pc[2]
    Xc=Pc[0]/Zc
    Yc=Pc[1]/Zc
    Ic=np.array([Xc,Yc]).reshape(2,1)
    newuv=np.array([Pc[0], Pc[1]]).reshape(2, 1) / Pc[2]
    innovation=uv.reshape(2,1)-Ic
    normI=np.linalg.norm(innovation)
    
    if(normI>error_threshold):
        return nominal_state, error_state_covariance, innovation
    
    uv=uv.flatten()
    uvm=np.array([[1,0,-newuv[0,0]],[0,1,-newuv[1,0]]]).reshape(2,3)
    delztpc=(1/Zc) * uvm
    Pl=Pc.flatten()
    Pch=np.array([[0, -Pl[2], Pl[1]], [Pl[2], 0, -Pl[0]], [-Pl[1], Pl[0], 0]]).reshape(3,3)
    H = np.zeros((2, 18))
    H[0:2, 0:3]=np.matmul(delztpc,- R.T)
    H[0:2, 6:9]=np.matmul(delztpc, Pch)
    K = error_state_covariance @ H.T @ np.linalg.inv(H @ error_state_covariance @ H.T + Q)
    delx=np.matmul(K, innovation)

    new_p=p+delx[0:3]
    new_v=v+delx[3:6]
    new_a_b=a_b+delx[9:12]
    new_w_b=w_b+delx[12:15]
    new_g=g+delx[15:18]

    delq=Rotation.from_rotvec(delx[6:9].ravel()).as_quat()
    new_q=quat_mul(q,delq)
    new_q=Rotation.from_quat(new_q)
    I=np.eye(18)
    phi=I-np.matmul(K,H)
    sigmat=np.matmul(phi,np.matmul(error_state_covariance,phi.T))+ np.matmul(K, np.matmul(Q, K.T))
   
    return (new_p, new_v, new_q, new_a_b, new_w_b, new_g), sigmat, innovation
    # return new_nominal_state, new_error_state_covariance, innovation
