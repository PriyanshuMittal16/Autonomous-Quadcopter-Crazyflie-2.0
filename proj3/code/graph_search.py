from collections import defaultdict
from heapq import heappush, heappop, heapify  # Recommended.
import numpy as np
import inspect
from flightsim.world import World


from .occupancy_map import OccupancyMap # Recommended.


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    distance_node = defaultdict(lambda: float("inf"))
    distance_node[start_index] = 0
    heuristic_node=defaultdict(lambda: float("inf"))
    heuristic_node[start_index]=np.linalg.norm((np.array(goal_index)) - np.array(start_index))
    parent_node={}
    clist=[(0, start_index)]
    si=start_index

    i=0
   

    while clist:
        node = heappop(clist)[1]
        array_node=np.array(node)
        if(node==goal_index):
            goal_path=[]
            goal_path.append(goal)
            tempt=goal_index
            
            while tempt:
                p=parent_node[tempt]
                if p==si:
                    goal_path.insert(0, start)
                    break
                goal_path.insert(0, occ_map.index_to_metric_center(p))
                tempt=p
            
            
            goal_path=np.array(goal_path)
            return goal_path,i
 
        i=i+1
        neighbor = np.array([[1, -1, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, -1, 0],
                         [0, 1, 0],
                         [-1, -1, 0],
                         [-1, 0, 0],
                         [-1, 1, 0],
                         [1, - 1, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [0, -1, 1],
                         [0, 1, 1],
                         [- 1, - 1, 1],
                         [-1, 0, 1],
                         [- 1, 1, 1],
                         [0, 0, 1],
                         [1, -1, -1],
                         [1, 0, -1],
                         [1, 1, -1],
                         [0, -1, -1],
                         [0, 1, -1],
                         [-1, - 1, - 1],
                         [- 1, 0, - 1],
                         [- 1, 1, - 1],
                         [0, 0, - 1]])
        Every_boundary = neighbor + array_node
        Every_boundary = Every_boundary[np.all(Every_boundary >= 0, axis=1), :]
        mapp = occ_map.map
        mapsize = mapp.shape
        Every_boundary = Every_boundary[np.all(Every_boundary < mapsize, axis=1), :]
        x = Every_boundary[:, 0]
        y = Every_boundary[:, 1]
        z = Every_boundary[:, 2]
        Every_boundary = Every_boundary[np.where(mapp[x, y, z] == 0)]
        boundary=Every_boundary
        
        for k in boundary:
            next_node=tuple(k)
            
            c=k-node
            e=np.linalg.norm(c)
            v=heuristic_node[node]
            v=v+e
            d=v

            if astar:
                H=k-goal_index
                heuristic=np.linalg.norm(H)
                d=v+heuristic
            
            calc=v*heuristic                

            if (v<heuristic_node[next_node]):
                heuristic_node[next_node]=v
                distance_node[next_node]=d
                parent_node[next_node]=node
                heappush(clist,(d,next_node))





        





    

