/*
 * Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
 * Project: FreiCAR
 * Do NOT distribute this code to anyone outside the FreiCAR project
 */

/* A ROS implementation of the Pure pursuit path tracking algorithm (Coulter 1992).
   Terminology (mostly :) follows:
   Coulter, Implementation of the pure pursuit algoritm, 1992 and
   Sorniotti et al. Path tracking for Automated Driving, 2017.
 */

#include <string>
#include <cmath>
#include <algorithm>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/transform_datatypes.h>
#include <tf2/transform_storage.h>
#include <tf2/buffer_core.h>
#include <tf2/convert.h>
#include <tf2/utils.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <kdl/frames.hpp>
#include <raiscar_msgs/ControlReport.h>
#include "raiscar_msgs/ControlCommand.h"
#include "std_msgs/Bool.h"
#include "controller.h"
#include <fstream>
#include <sstream>
#include <iostream>

#include <kdl/frames.hpp>

using std::string;



class PurePursuit: public controller
{
public:

    //! Constructor
    PurePursuit();

    //! Run the controller.
    void run();
    float distance(float x2, float y2, float z2, float x1, float y1, float z1);  //sg1
private:
    void controller_step(nav_msgs::Odometry odom);
    double ld_dist_;
    int index = 0 ;
    double sum_pos_error = 0;  //sg1
    double mean_pos_error = 0;
};



PurePursuit::PurePursuit()
{
    // Get parameters from the parameter server
    nh_private_.param<double>("lookahead_dist", ld_dist_, 0.6);
    std::cout << "Pure Pursuit controller started..." << std::endl;

}

/*
 * Implement your controller here! The function gets called each time a new odometry is incoming.
 * The path to follow is saved in the variable "path_". Once you calculated the new control outputs you can send it with
 * the pub_acker_ publisher.
 */

//sg1 - Distance for error calculation
float PurePursuit::distance(float x2, float y2, float z2, float x1, float y1, float z1)
{
    return sqrt(pow(x2 -x1,2) + pow(y2 - y1,2) + pow(z2 - z1,2));
}
void PurePursuit::controller_step(nav_msgs::Odometry odom)
{

    try{
        geometry_msgs::TransformStamped tf_msg;
        geometry_msgs::TransformStamped front_axis_tf_msg;

        tf2::Stamped<tf2::Transform> map_t_fa;
        tf_msg = tf_buffer_.lookupTransform(map_frame_id_, rear_axis_frame_id_, ros::Time(0));
        front_axis_tf_msg = tf_buffer_.lookupTransform(map_frame_id_, front_axis_frame_id_, ros::Time(0));

        tf2::convert(tf_msg, map_t_fa);

        if(path_.size() > 0 && index < path_.size()-1)
        {
//            sendGoalMsg(false);
//            completion_advertised_ = false;
            sendGoalMsg(false);
            float x = map_t_fa.getRotation().x();
            float y = map_t_fa.getRotation().y();
            float z = map_t_fa.getRotation().z();
            float w = map_t_fa.getRotation().w();

            tf2::Quaternion q(x,y,z,w);
            tf2::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);



            float alpha = atan2(path_[index].getOrigin().y() - map_t_fa.getOrigin().y(), path_[index].getOrigin().x() - map_t_fa.getOrigin().x()) - yaw;

            double steering_angle = atan2(2 * L_ * sin(alpha), ld_dist_);
//            float pid_vel_out = 0.0;
//            if (des_v_ >= 0) {
//                pid_vel_out = vel_pid.step((des_v_ - odom.twist.twist.linear.x), ros::Time::now());
//                //std::cout << "throttle" << pid_vel_out << std::endl;
//            } else {
//                pid_vel_out = des_v_;
//
//                vel_pid.resetIntegral();
//            }


            steering_angle = steering_angle / (70.0 * M_PI / 180.0);
            std::cout<<steering_angle<<std::endl;
            cmd_control_.steering = steering_angle;//  DUMMY_STEERING_ANGLE should be a value in degree
            if(steering_angle < -0.3  || steering_angle > 0.3 )
            {

                cmd_control_.throttle = 0.09;
            }
//            else if(steering_angle < -0.5 || steering_angle > 0.5){
////                    cmd_control_.throttle_mode = 1;
//                cmd_control_.throttle = -0.1;
//                cmd_control_.brake = 1;
//                ros::Duration(1).sleep();
//
//                    ros::Duration(1).sleep();
//                    cmd_control_.throttle = 0.05;
//            }
//            else if(steering_angle < -0.4 || steering_angle > 0.4)
//            {
//                cmd_control_.brake = 1;
//                cmd_control_.brake = 0;
//                cmd_control_.throttle = 0.04;
//            }
            else
            {
                cmd_control_.throttle = 0.12;
            }
            cmd_control_.throttle_mode = 0;
//            cmd_control_.throttle = std::min(cmd_control_.throttle, 0.10f);
//            cmd_control_.throttle = std::max(std::min((double) cmd_control_.throttle, 1.0), 0.0);
            pub_acker_.publish(cmd_control_);

            if(stop_sign == true){
                std::cout<< "did i stop? " << std::endl;
                ros::Duration(0.5).sleep();
                cmd_control_.throttle = 0;
                cmd_control_.brake = 1;
//                 cmd_control_.steering = 0;
                pub_acker_.publish(cmd_control_);
                std::cout<< "STOP IN CONTROLLER" << cmd_control_.throttle << std::endl;
                ros::Duration(1).sleep();
                cmd_control_.throttle = 0.12;
                cmd_control_.brake = 0;
                std::cout<< "i woke up " << std::endl;
                std::cout<< "START AGAIN " << cmd_control_.throttle << std::endl;
                pub_acker_.publish(cmd_control_);
            }
            if(HLC_stop == true){
                std::cout<< "did i stop? " << std::endl;
                cmd_control_.throttle = 0;
                cmd_control_.brake = 1;
//                 cmd_control_.steering = 0;
                pub_acker_.publish(cmd_control_);
                ros::Duration(1).sleep();
                cmd_control_.brake = 0;
                pub_acker_.publish(cmd_control_);
                std::cout<< "STOP IN CONTROLLER" << cmd_control_.throttle << std::endl;

//                ros::Duration(1).sleep();
//                cmd_control_.throttle = 0.12;
//                cmd_control_.brake = 0;
//                std::cout<< "i woke up " << std::endl;
//                std::cout<< "START AGAIN " << cmd_control_.throttle << std::endl;
//                pub_acker_.publish(cmd_control_);
            }
            if(min_depth_dist < 0.2){
                std::cout<< "starting overtake" << std::endl;
                cmd_control_.throttle = -0.1;
                cmd_control_.steering = 0;
                pub_acker_.publish(cmd_control_);
                ros::Duration(0.20).sleep();
                cmd_control_.throttle = 0.09;
                cmd_control_.steering =steering_angle / (70.0 * M_PI / 180.0);
                pub_acker_.publish(cmd_control_);
                index = index+3;

            }
            float pos_error_x = path_[index].getOrigin().x() - map_t_fa.getOrigin().x();
            float pos_error_y = path_[index].getOrigin().y() - map_t_fa.getOrigin().y();

            float pos_error = 0;
            //sg1 Calculate pos_error and moving mean over time as idx increase (i.e mean position error over time)
            //pos_error = distance(path_[index].getOrigin().x(),path_[index].getOrigin().y(),path_[index].getOrigin().z(),
            //                   map_t_fa.getOrigin().x(),map_t_fa.getOrigin().y(), map_t_fa.getOrigin().z());
            pos_error = fabs(sin(alpha) * ld_dist_);
            if (fabs(pos_error_x) < 0.84 && fabs(pos_error_y) < 0.84)
            {

                sum_pos_error = sum_pos_error + pos_error;
                mean_pos_error = sum_pos_error / index;
//                std::cout << "pos_error ist " << pos_error << "at time " << index << std::endl;
//                std::fstream file;
//                file.open(path, std::fstream::app);
//                if (!file.is_open()) {
//                    std::cout << "File doesnt exist ";
//                }
//                file << pos_error << ",";
//                file.close();
                //  std::cout << "index "<< index<< std::endl;
                index = index + 1 ;
                if (index == path_.size()-6)
                {
                    std::cout << "GOAL REACHED BY PURE PURSUIT " << index << std::endl;
                    sendGoalMsg(true);
//                    index = 0;
                    ros::Duration(1).sleep();
                    sendGoalMsg(false);
                    index = 1;
               }



            }
            //std::cout << "index" << index << std::endl;
            //std::cout << "mean_pos_error ist " << mean_pos_error << "at time " << index << std::endl;
        }

        else{
            cmd_control_.throttle = 0;
            pub_acker_.publish(cmd_control_);


        }


    }

    catch (tf2::TransformException &ex)
    {
        ROS_WARN_STREAM(ex.what());
    }


}

void PurePursuit::run()
{
    ros::spin();
}

int main(int argc, char**argv)
{
    ros::init(argc, argv, "pure_pursuit_controller");

    PurePursuit controller;
    controller.run();

    return 0;
}