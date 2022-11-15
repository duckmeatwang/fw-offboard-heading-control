#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Quaternion, TwistStamped
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.msg import State, AttitudeTarget
from std_msgs.msg import Float64
import tf

import math
import numpy as np
import bisect
from datetime import datetime
import heapq
from pandas import DataFrame
import time
from scipy.spatial.transform import Rotation as Rot
from math import sin, cos, atan2, sqrt, acos, pi, hypot


global path_x, path_y, path_yaw, num, i

i = 0
num = 0
enu_pos = []
all_distance = []
point_list = []
ref_path = []


def rot_mat_2d(angle):

    r = Rot.from_euler('z', angle)

    return r.as_dcm()[0:2, 0:2]

def angle_mod(x, zero_2_2pi=False, degree=False):

    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


def plan_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature,
                        step_size=0.1, selected_types=None):

    if selected_types is None:
        planning_funcs = _PATH_TYPE_MAP.values()
    else:
        planning_funcs = [_PATH_TYPE_MAP[ptype] for ptype in selected_types]

    # calculate local goal x, y, yaw
    l_rot = rot_mat_2d(s_yaw)  # rotation matrix
    #le_xy = np.stack([g_x - s_x, g_y - s_y]).T @ l_rot 
    le_xy = [g_x - s_x, g_y - s_y]
    local_goal_x = le_xy[0]
    local_goal_y = le_xy[1]
    local_goal_yaw = g_yaw - s_yaw

    lp_x, lp_y, lp_yaw, modes, lengths = _dubins_path_planning_from_origin(local_goal_x, local_goal_y, local_goal_yaw, curvature, step_size, planning_funcs)

    # Convert a local coordinate path to the global coordinate
    rot = rot_mat_2d(-s_yaw)
    lp = np.transpose(np.stack([lp_x, lp_y]))
    converted_xy = np.matmul(lp, rot)
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = angle_mod(np.array(lp_yaw) + s_yaw)

    return x_list, y_list, yaw_list, modes, lengths


def _mod2pi(theta):
    return angle_mod(theta, zero_2_2pi=True)


def _calc_trig_funcs(alpha, beta):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_ab = cos(alpha - beta)
    return sin_a, sin_b, cos_a, cos_b, cos_ab


def _LSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "S", "L"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
    if p_squared < 0:  # invalid configuration
        return None, None, None, mode
    tmp = atan2((cos_b - cos_a), d + sin_a - sin_b)
    d1 = _mod2pi(-alpha + tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(beta - tmp)
    return d1, d2, d3, mode


def _RSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "S", "R"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
    if p_squared < 0:
        return None, None, None, mode
    tmp = atan2((cos_a - cos_b), d - sin_a + sin_b)
    d1 = _mod2pi(alpha - tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(-beta + tmp)
    return d1, d2, d3, mode


def _LSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = -2 + d ** 2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - atan2(-2.0, d1)
    d2 = _mod2pi(-alpha + tmp)
    d3 = _mod2pi(-_mod2pi(beta) + tmp)
    return d2, d1, d3, mode


def _RSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = d ** 2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((cos_a + cos_b), (d - sin_a - sin_b)) - atan2(2.0, d1)
    d2 = _mod2pi(alpha - tmp)
    d3 = _mod2pi(beta - tmp)
    return d2, d1, d3, mode


def _RLR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "L", "R"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    d3 = _mod2pi(alpha - beta - d1 + d2)
    return d1, d2, d3, mode


def _LRL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "R", "L"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (- sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + _mod2pi(d2))
    return d1, d2, d3, mode


_PATH_TYPE_MAP = {"LSL": _LSL, "RSR": _RSR, "LSR": _LSR, "RSL": _RSL,
                  "RLR": _RLR, "LRL": _LRL, }


def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, curvature,
                                        step_size, planning_funcs):
    dx = end_x
    dy = end_y
    d = hypot(dx, dy) * curvature

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    best_cost = float("inf")
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    for planner in planning_funcs:
        d1, d2, d3, mode = planner(alpha, beta, d)
        if d1 is None:
            continue

        cost = (abs(d1) + abs(d2) + abs(d3))
        if best_cost > cost:  # Select minimum length one.
            b_d1, b_d2, b_d3, b_mode, best_cost = d1, d2, d3, mode, cost

    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list = _generate_local_course(lengths, b_mode,
                                                      curvature, step_size)

    lengths = [length / curvature for length in lengths]

    return x_list, y_list, yaw_list, b_mode, lengths


def _interpolate(length, mode, max_curvature, origin_x, origin_y,
                 origin_yaw, path_x, path_y, path_yaw):
    if mode == "S":
        path_x.append(origin_x + length / max_curvature * cos(origin_yaw))
        path_y.append(origin_y + length / max_curvature * sin(origin_yaw))
        path_yaw.append(origin_yaw)
    else:  # curve
        ldx = sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":  # left turn
            ldy = (1.0 - cos(length)) / max_curvature
        elif mode == "R":  # right turn
            ldy = (1.0 - cos(length)) / -max_curvature
        gdx = cos(-origin_yaw) * ldx + sin(-origin_yaw) * ldy
        gdy = -sin(-origin_yaw) * ldx + cos(-origin_yaw) * ldy
        path_x.append(origin_x + gdx)
        path_y.append(origin_y + gdy)

        if mode == "L":  # left turn
            path_yaw.append(origin_yaw + length)
        elif mode == "R":  # right turn
            path_yaw.append(origin_yaw - length)

    return path_x, path_y, path_yaw


def _generate_local_course(lengths, modes, max_curvature, step_size):
    p_x, p_y, p_yaw = [0.0], [0.0], [0.0]

    for (mode, length) in zip(modes, lengths):
        if length == 0.0:
            continue

        # set origin state
        origin_x, origin_y, origin_yaw = p_x[-1], p_y[-1], p_yaw[-1]

        current_length = step_size
        while abs(current_length + step_size) <= abs(length):
            p_x, p_y, p_yaw = _interpolate(current_length, mode, max_curvature,
                                        origin_x, origin_y, origin_yaw,
                                        p_x, p_y, p_yaw)
            current_length += step_size

        p_x, p_y, p_yaw = _interpolate(length, mode, max_curvature, origin_x,
                                    origin_y, origin_yaw, p_x, p_y, p_yaw)

    return p_x, p_y, p_yaw

class uav_data(object):

    def __init__(self):
        self.imu_msg = Imu()
        self.gps_msg = Odometry()
        self.hdg_msg = Float64()
        self.vel_msg = TwistStamped()

        self.gps_pose = [0,0,0]
        self.imu_x = 0
        self.roll, self.pitch, self.yaw = 0,0,0
        self.quat = [0,0,0,0]
        self.vel_pose = [0,0,0]
        self.heading = 0
        self.vel = 0

        rospy.Subscriber("/plane_cam_0/mavros/state", State, self.state_cb)
        rospy.Subscriber("/plane_cam_0/mavros/global_position/local", Odometry, self.gps_callback)
        rospy.Subscriber("/plane_cam_0/mavros/global_position/compass_hdg", Float64, self.hdg_callback)
        rospy.Subscriber("/plane_cam_0/mavros/global_position/raw/gps_vel", TwistStamped, self.vel_callback)

        rospy.wait_for_service("/plane_cam_0/mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("plane_cam_0/mavros/set_mode", SetMode)
        
        self.path_pub = rospy.Publisher('/plane_cam_0/mavros/setpoint_trajectory/desired', Path, queue_size=50)
        self.path_record = Path()
        self.setpoint_pub = rospy.Publisher("/plane_cam_0/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.pose = PoseStamped()
        self.attitude_pub = rospy.Publisher("/plane_cam_0/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)
        self.set_attitude = AttitudeTarget()

    def state_cb(self, msg):
        self.current_state = msg

    def gps_callback(self, msg):
        self.gps_msg = msg
        self.gps_pose[0] = msg.pose.pose.position.x
        self.gps_pose[1] = msg.pose.pose.position.y
        self.gps_pose[2] = msg.pose.pose.position.z

    def imu_callback(self, msg):
        self.imu_msg = msg
        self.quat[0] = msg.orientation.w
        self.quat[1] = msg.orientation.x
        self.quat[2] = msg.orientation.y
        self.quat[3] = msg.orientation.z

    def hdg_callback(self, msg): #ned
        self.hdg_msg = msg
        self.heading = msg.data
        #print(self.vel_pose)

    def vel_callback(self, msg):
        self.vel_msg = msg
        self.vel_pose[0] = msg.twist.linear.x
        self.vel_pose[1] = msg.twist.linear.y
        self.vel_pose[2] = msg.twist.linear.z
        self.vel = np.sqrt(self.vel_pose[0]**2+self.vel_pose[1]**2)
        

    def DataUpdating(self, path_pub, path_record):
        """
        data update
        """
        global path_x, path_y, path_yaw, num, i

        current_time = rospy.Time.now()

        br = tf.TransformBroadcaster()
        # translate matrix
        br.sendTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                         rospy.Time.now(), "odom", "map")

        rate = rospy.Rate(50)

        num = len(path_yaw)
        #print('num')
        #print(num)

        x = path_x[i]
        y = path_y[i]
        yaw = path_yaw[i]
        #print('record path')
        #print(x, y, yaw)

        quat = tf.transformations.quaternion_from_euler(0, 0, yaw) #roll pitch yaw

        # pose setting
        self.pose = PoseStamped()
        self.pose.header.stamp = current_time
        self.pose.header.frame_id = 'odom'
        self.pose.pose.position.x = x
        self.pose.pose.position.y = y
        self.pose.pose.position.z = 100
        self.pose.pose.orientation.x = quat[0]
        self.pose.pose.orientation.y = quat[1]
        self.pose.pose.orientation.z = quat[2]
        self.pose.pose.orientation.w = quat[3]

        # path setting
        self.path_record.header.stamp = current_time
        self.path_record.header.frame_id = 'odom'
        self.path_record.poses.append(self.pose)
        # print('path_record')
        # print(self.path_record)

        # number of path 
        # if len(self.path_record.poses) > 1000:
        #     self.path_record.poses.pop(0)

        if i < (num-1):
            self.path_pub.publish(self.path_record)
            
            

        self.path_pub.publish(self.path_record)
        rate.sleep()

    def node(self, event):
    
        global path_x, path_y, path_yaw, i

        # path smooth
        start_x, start_y, start_yaw = self.gps_pose[0], self.gps_pose[1], np.deg2rad(self.heading)
        end_x, end_y, end_yaw = 200, 200, np.deg2rad(45.0)
        #end_x, end_y, end_yaw = -349.606, -787.27, np.deg2rad(45.0)
        # dx = [self.gps_pose[0], -359.67, -354.642, -349.606]
        # dy = [self.gps_pose[1], -770, -778.63, -787.27]
        # dx = [self.gps_pose[0], self.gps_pose[0]+100, self.gps_pose[0]+150, self.gps_pose[0]+200]
        # dy = [self.gps_pose[1], self.gps_pose[1]+150, self.gps_pose[1]+200, self.gps_pose[1]+300]

        curvature = 0.05

        path_x, path_y, path_yaw, mode, lengths = plan_dubins_path(start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)

        rate = rospy.Rate(20)

        self.current_state = State()
        self.path_record = Path()

        num = len(path_yaw)
        print('num = ')
        print(num)

        # pop
        #ref_x = [rx[i] for i in range(0, num, 100)]
        # ref_x = []
        # ref_y = []
        # ref_yaw = []
        # for i in range(num):
        #     if i%50 == 0:
        #         ref_x.append(rx[i])
        #         ref_y.append(ry[i])
        #         ref_yaw.append(ryaw[i])

        # ref_num = len(ref_x)
        # print('ref_num = ')
        # print(ref_num)

        # ########### calaulate every point distance ############
        # for i in range(ref_num-1):
        #     dd = np.sqrt(np.square(self.gps_pose[0]-ref_x[i])+np.square(self.gps_pose[1]-ref_y[i]))
        #     all_distance.append(dd)
        # #print(all_distance)
        # a = min(all_distance)
        # index = all_distance.index(a)

        # wp_1 = [ref_x[index], ref_y[index], ref_yaw[index]]
        # print('wp_1 = ')
        # print(wp_1)

        # ######### go to the closest point #########
        # quat = tf.transformations.quaternion_from_euler(0, 0,wp_1[2]) 
        # self.pose.pose.position.x = wp_1[0]
        # self.pose.pose.position.y = wp_1[1]
        # self.pose.pose.position.z = 100
        # self.pose.pose.orientation.x = quat[0]
        # self.pose.pose.orientation.y = quat[1]
        # self.pose.pose.orientation.z = quat[2]
        # self.pose.pose.orientation.w = quat[3]
        # #set_attitude.body_rate.x = 0     # roll rate
        # #set_attitude.body_rate.y = 0     # pitch rate
        # #self.set_attitude.body_rate.z = wp_1[2]    # yaw rate
        # self.set_attitude.thrust = 0.1

        # for i in range(100):   
        #     if(rospy.is_shutdown()):
        #         break

        #     self.attitude_pub.publish(self.set_attitude)
        #     self.setpoint_pub.publish(self.pose)
        #     rate.sleep()
        

        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'

        last_req = rospy.Time.now()

        enu_pos.append([self.gps_pose[0], self.gps_pose[1], self.gps_pose[2]])


        while not rospy.is_shutdown():
    
            if(self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(1.0)):
                if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                    rospy.loginfo("OFFBOARD enabled")
            
                last_req = rospy.Time.now()

            else:
                
                if i < (num-1):
                    print('start record!!')

                    ################## reference path #####################
                    self.DataUpdating(self.path_pub, self.path_record)
                    # reference point
                    x = path_x[i]
                    y = path_y[i]
                    yaw = path_yaw[i]

                    ############### calculate ######################
                    print('uav _position = ')
                    print(self.gps_pose)
                    print('ref_point = ')
                    print(x, y, yaw)
                    ############### parameter L1 ####################
                    L1 = np.sqrt(np.square(self.gps_pose[0]-x)+np.square(self.gps_pose[1]-y))  # current distance between uav and target path
                    L1_vector = [x-self.gps_pose[0], y-self.gps_pose[1]]
                    print('L1 = ')
                    print(L1)
                    aa = self.vel
                    bb = L1_vector
                    Lx=np.sqrt(np.dot(aa, aa))
                    Ly=np.sqrt(np.dot(bb, bb))
                    
                    cos_angle=np.dot(aa, bb)/(Lx*Ly)
                    print('cos_angle = ')
                    print(cos_angle)
                    angle=np.arccos(cos_angle)
                    angle2=angle*360/2/np.pi
                    print('angle2')
                    print(angle2)
                    ############## parameter 2 ######################


                    ############## acc velocity #####################
                    a = 2*np.square(self.vel)*np.sin(angle2[0])/L1
                    heading_dot = a/self.vel
                    print('heading_dot = ')
                    print(heading_dot)

                    # if cos_angle[1] > 0:
                    #     # left 
                    #     heading_dot = a/self.vel
                    #     print('heading_dot = ')
                    #     print(heading_dot)

                    # elif cos_angle[1] < 0:
                    #     # right
                    #     heading_dot = a/self.vel

                    # else:
                    #     heading_dot = 0

                    ################################################                
    
                    # quat = tf.transformations.quaternion_from_euler(0, 0, yaw) #roll pitch yaw

                    # self.pose.pose.position.x = x
                    # self.pose.pose.position.y = y
                    self.pose.pose.position.z = 100
                    # self.pose.pose.orientation.x = quat[0]
                    # self.pose.pose.orientation.y = quat[1]
                    # self.pose.pose.orientation.z = quat[2]
                    # self.pose.pose.orientation.w = quat[3]
                    # set_attitude.body_rate.x = 0     # roll rate
                    # set_attitude.body_rate.y = 0     # pitch rate
                    self.set_attitude.body_rate.z = heading_dot    # yaw rate
                    self.set_attitude.thrust = 0.1
                    
                    self.attitude_pub.publish(self.set_attitude)
                    self.setpoint_pub.publish(self.pose)
                        
                    print('yaw rate command = ')
                    print(heading_dot)
                    #time.sleep(3)
                    rate.sleep()
                    i +=1


                rate.sleep()


if __name__ == '__main__':
    rospy.init_node('dubins_L1', anonymous=True)
    dt = 1.0/20
    pathplan_run = uav_data()
    rospy.Timer(rospy.Duration(dt), pathplan_run.node)
    rospy.spin()

    # df = DataFrame({'enu_pos': enu_pos })
    # df.to_excel('data_1109.xlsx', sheet_name='sheet1', index=False)

    # df = DataFrame({'x': rx })
    # df.to_excel('data_1109.xlsx', sheet_name='sheet1', index=False)
