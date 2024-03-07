#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>

#include "chessboard_transform_params.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/LinearMath/Vector3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_broadcaster.h"

using namespace std;
using namespace chessboard_transform;
using std::placeholders::_1;
using std::placeholders::_2;

//                                                                                                //
// ========================================= Constants ========================================== //
//                                                                                                //

/**
 * The dictionary to use for the ArUco markers.
 */
static const auto ARUCO_DICT = cv::aruco::DICT_4X4_50;

/**
 * The size of one side of the chessboard, in mm.
 */
static const float CHESSBOARD_SIZE = 377.825f;

/**
 * The order of the ArUco markers on the chessboard.
 */
static const int ARUCO_ORDER[] = { 2, 3, 0, 1 };

/**
 * Lookup table that determines which corner of each ArUco marker corresponds with its corner of the
 * chessboard.
 */
static const int ARUCO_LOOKUP[] = { 3, 0, 1, 2 };

//                                                                                                //
// ================================= ChessboardTransform class ================================== //
//                                                                                                //

/**
 * This class is responsible for managing the chessboard transform node.
 */
class ChessboardTransform
{
public:
  rclcpp::Node::SharedPtr node;  //< The node that this class is managing.

  /**
   * Construct a new Chessboard Transform object. This will initialize the node and parameter
   * listener.
   */
  explicit ChessboardTransform()
  {
    node = rclcpp::Node::make_shared("chessboard_transform");
    param_listener_ = make_unique<ParamListener>(node);
    params_ = make_unique<Params>(param_listener_->get_params());

    // Initialize the tf broadcaster.
    tf_pub_ = make_unique<tf2_ros::TransformBroadcaster>(node);

    // Initialize the image transport and publisher.
    it_ = make_unique<image_transport::ImageTransport>(node);
    image_pub_ = make_unique<image_transport::Publisher>(
        it_->advertise(params_->warped_chessboard_topic, 1));

    // Initialize the camera subscriber. We must bind the callback to this object so that we can
    // access our params and publishers from within the callback.
    auto bound_callback = bind(&ChessboardTransform::image_callback, this, _1, _2);
    camera_sub_ = make_unique<image_transport::CameraSubscriber>(
        it_->subscribeCamera(params_->camera_base_topic, 1, bound_callback));
  }

private:
  unique_ptr<ParamListener> param_listener_;                  //< Parameter listener for this node.
  unique_ptr<Params> params_;                                 //< The parameters for this node.
  unique_ptr<image_transport::ImageTransport> it_;            //< Image transport for this node.
  unique_ptr<image_transport::Publisher> image_pub_;          //< Image publisher for this node.
  unique_ptr<image_transport::CameraSubscriber> camera_sub_;  //< Camera subscriber for this node.
  unique_ptr<tf2_ros::TransformBroadcaster> tf_pub_;  //< Transform broadcaster for this node.

  /**
   * Return the aruco params used for this node.
   *
   * @return aruco params
   */
  cv::Ptr<cv::aruco::DetectorParameters> aruco_params()
  {
    auto params = new cv::aruco::DetectorParameters();
    return cv::Ptr<cv::aruco::DetectorParameters>(params);
  }

  /**
   * Find the corners of the chessboard using aruco markers in the image.
   *
   * We search the list of detected markers for each element of ARUCO_ORDER. If the proper marker is
   * found, we add its corners to our output list. If any corner is not found, we return an empty
   * list.
   *
   * @param[in] img OpenCV image to find the chessboard in.
   * @return The corners of the chessboard, or an empty vector if the chessboard was not found.
   */
  vector<cv::Point2f> find_chessboard_corners(cv::Mat img)
  {
    // Static variables to avoid re-initialization and re-allocation.
    static const auto aruco_detector_params = aruco_params();
    static const auto aruco_dictionary = cv::aruco::getPredefinedDictionary(ARUCO_DICT);
    static vector<int> aruco_ids;
    static vector<vector<cv::Point2f>> aruco_corners;

    // Detect the aruco markers in the image.
    aruco_ids.clear();
    aruco_corners.clear();
    cv::aruco::detectMarkers(img, aruco_dictionary, aruco_corners, aruco_ids,
                             aruco_detector_params);
    if (aruco_ids.size() < 4) return {};

    // Find the proper markers in order and add the correct corners to our output list.
    vector<cv::Point2f> chessboard_corners;
    for (int i = 0; i < 4; ++i) {
      // Search for the next marker in the order.
      auto id = ARUCO_ORDER[i];
      int marker_index = 0;
      for (; marker_index < 4; ++marker_index) {
        if (aruco_ids[marker_index] == id) break;
      }
      if (marker_index == 4) return {};

      // Determine which corner of the marker matches the corner of the chessboard and add it to our
      // output list.
      auto marker_corners = aruco_corners[marker_index];
      cv::Point2f correct_marker_corner = marker_corners[ARUCO_LOOKUP[id]];
      chessboard_corners.push_back(correct_marker_corner);
    }

    return chessboard_corners;
  }

  /**
   * Callback to be called when a new image is received. This will identify the chessboard in the
   * image and publish the transform from the camera to the chessboard.
   *
   * @param image The image that was received.
   * @param cinfo The camera info that was received.
   */
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image,
                      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cinfo)
  {
    // The corners of the real-world chessboard, measured in mm.
    static const vector<cv::Point2f> IRL_CHESSBOARD_CORNERS = {
      cv::Point2f(0.0, 0.0),
      cv::Point2f(0.0, CHESSBOARD_SIZE),
      cv::Point2f(CHESSBOARD_SIZE, CHESSBOARD_SIZE),
      cv::Point2f(CHESSBOARD_SIZE, 0.0),
    };

    auto now = rclcpp::Clock().now();

    // Convert the image to an OpenCV image.
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(node->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // Extract camera info.
    string camera_frame = cinfo->header.frame_id;
    cv::Mat camera_matrix(3, 3, CV_64F, (void*)cinfo->k.data());
    cv::Mat dist_coeffs(1, 5, CV_64F, (void*)cinfo->d.data());

    // Construct header that can be used for both the image and the transform.
    std_msgs::msg::Header header;
    header.stamp = now;
    header.frame_id = camera_frame;

    // Undistort the image.
    static cv::Mat undistorted;
    cv::undistort(cv_ptr->image, undistorted, camera_matrix, dist_coeffs);

    // Estimate the pose of the chessboard in the image.
    auto chessboard_corners = find_chessboard_corners(undistorted);
    if (chessboard_corners.size() != 4) return;
    auto transform = cv::getPerspectiveTransform(IRL_CHESSBOARD_CORNERS, chessboard_corners);
    static cv::Mat _camera_matrix, rot_matrix;
    static cv::Vec4d trans_matrix;
    cv::decomposeProjectionMatrix(transform, _camera_matrix, rot_matrix, trans_matrix);

    // Convert pose to tf2 format.
    tf2::Vector3 trans_matrix_tf(trans_matrix[0] / trans_matrix[3],
                                 trans_matrix[1] / trans_matrix[3],
                                 trans_matrix[2] / trans_matrix[3]);
    tf2::Matrix3x3 rot_matrix_tf(
        rot_matrix.at<double>(0, 0), rot_matrix.at<double>(0, 1), rot_matrix.at<double>(0, 2),
        rot_matrix.at<double>(1, 0), rot_matrix.at<double>(1, 1), rot_matrix.at<double>(1, 2),
        rot_matrix.at<double>(2, 0), rot_matrix.at<double>(2, 1), rot_matrix.at<double>(2, 2));

    // Publish the transform from the camera to the chessboard.
    tf2::Transform tf_transform(rot_matrix_tf, trans_matrix_tf);
    geometry_msgs::msg::TransformStamped transform_stamped;
    transform_stamped.header = header;
    transform_stamped.child_frame_id = params_->chessboard_frame;
    transform_stamped.transform = tf2::toMsg(tf_transform);
    tf_pub_->sendTransform(transform_stamped);

    // Publish the warped chessboard image.
    cv::Mat warped;
    static const int BOARD_SIZE_INT = static_cast<int>(round(CHESSBOARD_SIZE));
    cv::warpPerspective(undistorted, warped, transform, cv::Size(BOARD_SIZE_INT, BOARD_SIZE_INT));
    sensor_msgs::msg::Image::SharedPtr msg =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, warped).toImageMsg();
    image_pub_->publish(msg);
  }
};

//                                                                                                //
// ============================================ main ============================================ //
//                                                                                                //

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto chessboard_transform = make_shared<ChessboardTransform>();
  rclcpp::spin(chessboard_transform->node);
  rclcpp::shutdown();
  return 0;
}