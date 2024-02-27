#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"

void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
  cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;

}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("chessboard_transform", options);

  image_transport::ImageTransport it(node);
  image_transport::Subscriber sub = it.subscribe("kinect/camera/image", 1, image_callback);
  RCLCPP_INFO(node->get_logger(), "Chessboard transform node started");

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
