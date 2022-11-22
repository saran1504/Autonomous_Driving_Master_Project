#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Bool.h>
#include <raiscar_msgs/ControlCommand.h>

ros::Publisher control_pub;
ros::Publisher control_mode_pub;

bool control_mode_state = false;
bool reverse_gear = false;

/* receives the joy message and converts it to a ControlCommand */
void PS4Callback(const sensor_msgs::Joy::ConstPtr& joy) {
    static raiscar_msgs::ControlCommand control_command;
    control_command.throttle_mode = 0;
    // PS4 controller threshold button: joy->buttons[7]
    control_command.throttle = std::max(std::min(joy->axes[1]/0.75, 1.0), -1.0);
    control_command.steering = std::max(std::min(joy->axes[2]/0.75, 1.0), -1.0);
    control_pub.publish(control_command);

    if (joy->buttons[9]) {
	    control_mode_state = !control_mode_state;
	    std_msgs::Bool bool_msg;
	    bool_msg.data = control_mode_state;
	    control_mode_pub.publish(bool_msg);
    }
}
/* receives the joy message and converts it to a ControlCommand */
void XBOX360Callback(const sensor_msgs::Joy::ConstPtr& joy) {
    static raiscar_msgs::ControlCommand control_command;
    control_command.throttle_mode = 0;
    float g_throttle = 1.-((joy->axes[5]+1)/2);
    float g_brake = 1. - ((joy->axes[2] + 1) / 2);
    bool hand_brake = joy->buttons[0];
    bool switch_direction = joy->buttons[1];

    if (switch_direction){
        reverse_gear = !reverse_gear;
        ros::Duration(0.3).sleep();
    }

    if (g_throttle > 0.0) {
        if (!reverse_gear)
            control_command.throttle = g_throttle;
        else
            control_command.throttle = -g_throttle;
    }

    if (g_brake > 0.02) {
        control_command.throttle = 0;
        control_command.brake = g_brake;
    }else{
        control_command.brake = 0;
    }

    control_command.hand_brake = hand_brake;

    // axes[0] at 0, goes from +1 to -1
    control_command.steering = joy->axes[0];
    control_pub.publish(control_command);
    if (joy->buttons[9]) {
	    control_mode_state = !control_mode_state;
	    std_msgs::Bool bool_msg;
	    bool_msg.data = control_mode_state;
	    control_mode_pub.publish(bool_msg);
    }

}

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "freicar_joy");
    ros::NodeHandle node_handle;
    std::cout << "-------------------------------------------\n"
                << "Using joystick controls:\n"
                << "L2: backward throttle, R2: forward throttle\n"
                << "L   : steering\n"
                << "X|A : handbrake\n"
                << "-------------------------------------------" << std::endl;
    control_pub = node_handle.advertise<raiscar_msgs::ControlCommand>("control", 10);;
    control_mode_pub = node_handle.advertise<std_msgs::Bool>("control_mode", 10);

    std::string controller_type;
    if (!ros::param::get("~controller_type", controller_type)) {
		ROS_ERROR("ERROR: could not find parameter: controller_type.\ncheck the launch file and use \"xbox360\" or \"ps4\".");
		std::exit(EXIT_FAILURE);
	}
    ros::Subscriber joy_sub;
    if (controller_type == "xbox360")
        joy_sub = node_handle.subscribe<sensor_msgs::Joy>("joy", 1, XBOX360Callback);
    else if (controller_type == "ps4")
        joy_sub = node_handle.subscribe<sensor_msgs::Joy>("joy", 1, PS4Callback);
    ros::spin();
    return 0;
}
