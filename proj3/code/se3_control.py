import numpy as np
import math
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2
        
        self.Km=1.5e-9
        self.Kf=6.11e-8

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        Kpv=[7.5,7.5,20]; Kdv=[4.4,4.4,7];Krv=[2600,2600,150]; Kwv=[130,130,80]
        Kp=np.diag(Kpv); Kd=np.diag(Kdv); Kr=np.diag(Krv);Kw=np.diag(Kwv)

        xstate=state["x"]
        vstate=state["v"]
        q=state["q"]
        wstate=state["w"]
        xflat=flat_output["x"]
        xdflat=flat_output["x_dot"]
        xddflat=flat_output["x_ddot"]
        xdddflat=flat_output["x_dddot"]
        xddddflat=flat_output["x_ddddot"]
        yaw=flat_output["yaw"]
        yawd=flat_output["yaw_dot"]

        rddotdes= xddflat-np.matmul(Kd, vstate-xdflat)-np.matmul(Kp, xstate-xflat)
        rddotdes=rddotdes.reshape((3,1))
        weight=np.array([[0],[0],[self.mass*self.g]])
        Fdes=self.mass*rddotdes+weight

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        # R = np.array([[r00, r01, r02],
        #             [r10, r11, r12],
        #             [r20, r21, r22]])
        R = Rotation.from_quat(state['q']).as_matrix() 
        
        zax=np.array([[0],[0], [1]])
        b3=np.matmul(R,zax)
        u1=np.matmul(b3.T,Fdes)
        
        b3des=Fdes/np.linalg.norm(Fdes)
        asi=np.array([[np.cos(yaw)],[np.sin(yaw)], [0]])
        asi=asi.reshape((3,))
        b3des=b3des.reshape((3,))
        b2=np.cross(b3des,asi)
        b2des=b2/np.linalg.norm(b2)

        Rdes=np.array([np.cross(b2des,b3des),b2des, b3des])
        Rdes=Rdes.T

        err=np.matmul(Rdes.T, R)-np.matmul(R.T,Rdes)
        err1=np.array([[err[2][1]], [err[0][2]], [err[1][0]]])
        errR=0.5*err1
        t=np.matmul(Kr,errR)
        mul=-t.reshape((3,))-np.matmul(Kw, wstate)
        u2=np.matmul(self.inertia,mul)

        gamma=self.k_drag/self.k_thrust
        l=self.arm_length

        U=np.append(u1,u2)
        A=np.array([[1,1,1,1],
                    [0,l, 0, -l],
                    [-l,0,l,0],
                    [gamma, -gamma, gamma,-gamma],
                    ])
        
        F=np.matmul(np.linalg.inv(A), U)
        g=np.sign(F)
        h=np.absolute(F)
        cmd_motor_speeds = g * np.sqrt(h/ self.k_thrust)
        cmd_motor_speeds = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)
        cmd_thrust=u1[0][0]
        cmd_moment=u2

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}

        return control_input
        
