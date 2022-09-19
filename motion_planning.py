import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import a_star, iterative_astar, ucs, heuristic, create_grid, create_grid_and_edges
# from planning_utils import bfs

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

import matplotlib.pyplot as plt
import networkx as nx


class States(Enum):
    # States fields changed from integers to auto()

    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                # self.all_waypoints = self.calculate_box() deleted
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
                    # Changed from self.takeoff_transition() to self.plan_path()
            # State.PLANNING added
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    # Deleted calculate_box(self)

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1],
                          self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...\n")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5
        # self.inFile()

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: retrieve current global position
        # TODO: convert to current local position using global_to_local()

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))

        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # TODO: New create_graph_grid
        # grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        grid, edges, north_offset, east_offset = create_grid_and_edges(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))

        # Define starting point on the grid (this is just grid center)
        # TODO: convert start position to current position rather than map center

        grid_start = (-north_offset, -east_offset)
        print('grid start = ', grid_start)

        # Set goal as some arbitrary position on the grid
        # TODO: adapt to set goal as latitude / longitude position and convert

        grid_goal = (-north_offset + 10, -east_offset + 10)
        print('grid goal = ', grid_goal)

        print('Local Start and Goal: ', grid_start, grid_goal)

        print('Found %5d edges' % len(edges))

        plt.figure(1)
        plt.imshow(grid, origin='lower', cmap='Greys')

        # Stepping through each edge
        for e in edges:
            p1 = e[0] 
            p2 = e[1]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')

        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.savefig('graph.png')
        # plt.show()

        # TODO: create the graph with the weight of the edges
        G = nx.Graph()
        for e in edges:
            p1 = e[0]
            p2 = e[1]
            dist = np.linalg.norm(np.array(p2) - np.array(p1))
            G.add_edge(p1, p2, weight=dist)

        # # Run A* to find a path from start to goal
        # # path, _ = a_star(grid, heuristic, grid_start, grid_goal)

        # # TODO: Run Iterative A* Search to find a path from start to goal
        # # path, _ = iterative_astar(grid, heuristic, grid_start, grid_goal)

        # # TODO: Run Uniform-Cost Search to find a path from start to goal
        # # path, _ = ucs(grid, grid_start, grid_goal)

        # # TODO: BFS Graph search
        # # path = bfs(grid, grid_start, grid_goal)

        # TODO: 3 Fixed points
        fixed_point_1 = (-north_offset - 10, -east_offset - 10)
        fixed_point_2 = (-north_offset - 20, -east_offset + 10)
        fixed_point_3 = (-north_offset + 10, -east_offset + 20)

        # TODO: 3 Fixed Points for A star
        path1, _1 = a_star(grid, heuristic, grid_start, fixed_point_1)
        path2, _2 = a_star(grid, heuristic, fixed_point_1, fixed_point_2)
        path3, _3 = a_star(grid, heuristic, fixed_point_2, fixed_point_3)

        # TODO: 3 Fixed Points for Iterative A star
        # path1, _1 = iterative_astar(grid, heuristic, grid_start, fixed_point_1)
        # path2, _2 = iterative_astar(grid, heuristic, fixed_point_1, fixed_point_2)
        # path3, _3 = iterative_astar(grid, heuristic, fixed_point_2, fixed_point_3)

        path = (path1 + path2 + path3)
        _ = _1 + _2 + _3

        plt.figure(2)
        plt.imshow(grid, origin='lower', cmap='Greys') 

        for e in edges:
            p1 = e[0]
            p2 = e[1]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')
        plt.plot([grid_start[1], grid_start[1]], [grid_start[0], grid_start[0]], 'r-')

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-')
        plt.plot([grid_goal[1], grid_goal[1]], [grid_goal[0], grid_goal[0]], 'r-')

        plt.plot(grid_start[1], grid_start[0], 'gx')
        plt.plot(grid_goal[1], grid_goal[0], 'gx')

        plt.xlabel('EAST')
        plt.ylabel('NORTH')

        # save result png
        plt.savefig('path_found.png')
        plt.show()

        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        # TODO: prune path to minimize number of waypoints

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints

        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
