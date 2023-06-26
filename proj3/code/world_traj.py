import numpy as np
import math as m
from scipy.sparse.linalg import spsolve as sp
from scipy.sparse import lil_matrix as lm


from .graph_search import graph_search

class WorldTraj(object):
    def calc_time(self,d, v,m):
        time=d/v
        time[0]*=3
        time[-1]*=3
        time=time*np.sqrt(1.65/time)
        cs=np.cumsum(time, axis=0)
        on=np.zeros(1)
        sum_t = np.vstack((on, cs)).flatten()

        i=8*(m-1)

        matrix1=lm((i,3))
        time=time.clip(0.25,np.inf)
        # print(time)

        return time, matrix1, i, sum_t

    
    def prune2(points,path):
        if (len(points)%2 == 0):
            g1 = points[len(points)-1]
            points = path[::2,:]
            g2 = np.vstack((points, g1))
            points = g2
        else:
            points = path[::2,:]
    
    
    
    def min_snap(self,t):
        t=t[0]
        snap = np.array([[0          , 0         , 0         , 0         , 0        , 0       , 0   , 1],
                            [t**7       , t**6      , t**5      , t**4      , t**3     , t**2    , t   , 1],
                            [7 * t**6   , 6 * t**5  , 5 * t**4  , 4 * t**3  , 3 * t**2 , 2 * t   , 1   , 0],
                            [42 * t**5  , 30 * t**4 , 20 * t**3 , 12 * t**2 , 6 * t    , 2       , 0   , 0],
                            [210 * t**4 , 120 * t**3, 60 * t**2 , 24 * t    , 6        , 0       , 0   , 0],
                            [840 * t**3 , 360 * t**2, 120 * t   , 24        , 0        , 0       , 0   , 0],
                            [2520 * t**2, 720 * t   , 120       , 0         , 0        , 0       , 0   , 0],
                            [5040 * t   , 720       , 0         , 0         , 0        , 0       , 0   , 0]])
        
        return snap
    
    def min_upc(self,matrix2, count, time,ar):
        if count != len(time) - 1:
            matrix2[5 + 8*count, 14 + 8*count] = -1
            matrix2[6 + 8*count, 13 + 8*count] = -2
            matrix2[7 + 8*count, 12 + 8*count] = -6
            matrix2[8 + 8*count, 11 + 8*count] = -24
            matrix2[9 + 8*count, 10 + 8*count] = -120
            matrix2[10 + 8*count, 9 + 8*count] = -720
            matrix2[8*count + 3:8*count + 11, 8*count:8*count + 8] = ar
        else:
            matrix2[8*count + 3:8*count + 11, 8*count:8*count + 8] = ar[:5, :]
        
        return matrix2   
    """
    """
    def __init__(self, world, start, goal):
    
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        # self.resolution = np.array([0.15, 0.15, 0.15])
        # self.resolution = np.array([0.24, 0.24, 0.24])
        self.margin = 0.5
        if start[2]>1.5 and start[2]<2:
            self.velocity=9.4
        elif start[2]>5:
            self.velocity=9.7
        else:
            self.velocity=10
        

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        
        # self.velocity = 6.35
        
        # print(np.shape(self.path))
        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.
        # self.points = self.prune_path(self.path)
        # STUDENT CODE HERE
        pt = list(self.path)
        
        pm2=len(pt) - 2
        count = 0
        while count != len(pt) - 2:
            if count > len(pt) - 2:
                break
            a1=pt[count]-pt[count+1]
            a2=pt[count+1]-pt[count+2]
            axis = np.cross(a1, a2)   
            distance = np.linalg.norm(a1)     
            norm_axis=np.linalg.norm(axis)
            if norm_axis == 0:
                del pt[count + 1]
                count =count- 1
            elif distance > 0.01:
                del pt[count]
            count =count+ 1

        self.points=np.array(pt)
        self.distance=np.linalg.norm(self.points[1::]-self.points[0:-1], axis=1).reshape(-1, 1)
        self.waqt = self.distance / self.velocity
        self.waqt[0] = 2.7*self.waqt[0]
        self.waqt[-1] = 2.7*self.waqt[-1]
        self.waqt *= (np.sqrt(1.65) / np.sqrt(self.waqt))
        self.waqt_array = np.hstack((0, np.cumsum(self.waqt)))

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        m=self.points.shape[0]
        
        time, mat1, k, half_time=self.calc_time(self.distance, self.velocity,m)
    
        mat2=lm((k,k))
        for i in range(m-1):
            mat1[8 * i + 3] = self.points[i]
            mat1[8 * i + 4] = self.points[i + 1]
        
        final_time=np.sum(time)

        c=0
        for j in time:
            snap=self.min_snap(j)
            mat2[[0, 1, 2], [6, 5, 4]] = [1, 2, 6]
            M2=self.min_upc(mat2, c, time, snap)
            c=c+1
        
        M2=M2.tocsc()
        M3=sp(M2, mat1).toarray()

        if t<= (final_time):
            sgn=np.sign(half_time-t)
            pos = np.where(sgn > 0)[0][0] - 1

            fmc = np.array([[(t - half_time[pos])**7, (t - half_time[pos])**6, (t - half_time[pos])**5, (t - half_time[pos])**4, (t - half_time[pos])**3, (t - half_time[pos])**2, (t - half_time[pos]), 1],
                              [7 * (t - half_time[pos])**6, 6 * (t - half_time[pos])**5, 5 * (t - half_time[pos])**4, 4 * (t - half_time[pos])**3, 3 * (t - half_time[pos])**2, 2 * (t - half_time[pos]), 1, 0],
                              [42 * (t - half_time[pos])**5, 30 * (t - half_time[pos])**4, 20 * (t - half_time[pos])**3, 12 * (t - half_time[pos])**2, 6 * (t - half_time[pos]), 2, 0, 0],
                              [210 * (t - half_time[pos])**4, 120 * (t - half_time[pos])**3, 60 * (t - half_time[pos])**2, 24 * (t - half_time[pos]), 6, 0, 0, 0],
                              [840 * (t - half_time[pos])**3, 360 * (t - half_time[pos])**2, 120 * (t - half_time[pos]), 24, 0, 0, 0, 0]])

            coeff=M3[8*pos: 8*(pos+1),:]

            final_matrix=fmc@coeff
            x=final_matrix[0,:]
            x_dot=final_matrix[1,:]
            x_ddot=final_matrix[2,:]
            x_dddot =final_matrix[3,:]
            x_ddddot=final_matrix[4,:]
        
        elif t>final_time:
            yaw = 0
            yaw_dot = 0
            x_dot    = np.zeros((3,))
            x_ddot   = np.zeros((3,))
            x_dddot  = np.zeros((3,))
            x_ddddot = np.zeros((3,))
            x = self.points[-1]


        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output