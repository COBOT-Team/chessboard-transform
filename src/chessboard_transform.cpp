#include <opencv2/calib3d.hpp>
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
 * The size of one side of an aruco marker, in mm.
 */
static const float ARUCO_SIZE = 18.5f;

/**
 * The order of the ArUco markers on the chessboard.
 */
static const int ARUCO_ORDER[] = { 3, 0, 1, 2 };

/**
 * Lookup table that maps chessboard corners to the proper aruco corners.
 */
static const int ARUCO_LOOKUP[] = { 2, 7, 8, 13 };

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
    found_perspective_transform_ = false;

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

    RCLCPP_INFO(node->get_logger(), "Starting chessboard transform");
  }

private:
  unique_ptr<ParamListener> param_listener_;                  //< Parameter listener for this node.
  unique_ptr<Params> params_;                                 //< The parameters for this node.
  unique_ptr<image_transport::ImageTransport> it_;            //< Image transport for this node.
  unique_ptr<image_transport::Publisher> image_pub_;          //< Image publisher for this node.
  unique_ptr<image_transport::CameraSubscriber> camera_sub_;  //< Camera subscriber for this node.
  unique_ptr<tf2_ros::TransformBroadcaster> tf_pub_;  //< Transform broadcaster for this node.

  cv::Mat perspective_transform_;
  bool found_perspective_transform_;

  /**
   * Return the aruco params used for this node.
   *
   * @return aruco params
   */
  cv::aruco::DetectorParameters aruco_params()
  {
    auto params = cv::aruco::DetectorParameters();
    return params;
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
    static const cv::aruco::ArucoDetector aruco_detector(aruco_dictionary, aruco_detector_params);
    static vector<int> aruco_ids;
    static vector<vector<cv::Point2f>> aruco_corners;

    // Detect the aruco markers in the image.
    aruco_ids.clear();
    aruco_corners.clear();
    aruco_detector.detectMarkers(img, aruco_corners, aruco_ids);
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

      // Add all four corners of the marker to our output list.
      for (auto corner : aruco_corners[marker_index]) {
        chessboard_corners.push_back(corner);
      }
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
      cv::Point2f(0.0, CHESSBOARD_SIZE),
      cv::Point2f(CHESSBOARD_SIZE, CHESSBOARD_SIZE),
      cv::Point2f(CHESSBOARD_SIZE, 0.0),
      cv::Point2f(0.0, 0.0),
    };

    // To increase accuracy, we use all four corners of each ArUco marker to estimate the pose of
    // the chessboard. This is the real-world position of the corners of the chessboard, measured in
    // meters.
    static const double HALF_CB = CHESSBOARD_SIZE / 2000.0;  // Half of chessboard size, in meters.
    static const double ARUCO = ARUCO_SIZE / 1000.0;         // Aruco size, in meters.
    static const vector<cv::Point3d> PNP_CHESSBOARD_CORNERS = {
      cv::Point3d(-HALF_CB, HALF_CB, 0.0),
      cv::Point3d(-HALF_CB + ARUCO, HALF_CB, 0.0),
      cv::Point3d(-HALF_CB + ARUCO, HALF_CB - ARUCO, 0.0),
      cv::Point3d(-HALF_CB, HALF_CB - ARUCO, 0.0),

      cv::Point3d(HALF_CB - ARUCO, HALF_CB, 0.0),
      cv::Point3d(HALF_CB, HALF_CB, 0.0),
      cv::Point3d(HALF_CB, HALF_CB - ARUCO, 0.0),
      cv::Point3d(HALF_CB - ARUCO, HALF_CB - ARUCO, 0.0),

      cv::Point3d(HALF_CB - ARUCO, -HALF_CB + ARUCO, 0.0),
      cv::Point3d(HALF_CB, -HALF_CB + ARUCO, 0.0),
      cv::Point3d(HALF_CB, -HALF_CB, 0.0),
      cv::Point3d(HALF_CB - ARUCO, -HALF_CB, 0.0),

      cv::Point3d(-HALF_CB, -HALF_CB + ARUCO, 0.0),
      cv::Point3d(-HALF_CB + ARUCO, -HALF_CB + ARUCO, 0.0),
      cv::Point3d(-HALF_CB + ARUCO, -HALF_CB, 0.0),
      cv::Point3d(-HALF_CB, -HALF_CB, 0.0),
    };

    auto now = rclcpp::Clock().now();

    // Check for new parameters.
    if (param_listener_->is_old(*params_)) {
      *params_ = param_listener_->get_params();
    }

    // Convert the image to an OpenCV image.
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(node->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // Check for camera info.
    // TODO: Does this actually work?
    if (!cinfo) {
      RCLCPP_ERROR(node->get_logger(), "No camera info received");
      return;
    }

    // Extract camera info into OpenCV format.
    string camera_frame = cinfo->header.frame_id;
    cv::Mat camera_matrix(3, 3, CV_64F, (void*)cinfo->k.data());
    cv::Mat dist_coeffs(1, 5, CV_64F, (void*)cinfo->d.data());

    // Find the chessboard in the image.
    auto chessboard_corners = find_chessboard_corners(cv_ptr->image);

    // If the chessboard was found in the image, update TF and perspective transform.
    bool found = chessboard_corners.size() == 16;
    if (found) {
      // Estimate pose.
      cv::Vec3d rvec;
      cv::Vec3d tvec;
      cv::Mat rmat(3, 3, CV_64F);
      cv::solvePnP(PNP_CHESSBOARD_CORNERS, chessboard_corners, camera_matrix, dist_coeffs, rvec,
                   tvec, false, cv::SOLVEPNP_IPPE);
      cv::Rodrigues(rvec, rmat);

      // Convert pose to tf2 format.
      tf2::Vector3 tvec_tf(tvec[0], tvec[1], tvec[2]);
      tf2::Matrix3x3 rmat_tf(rmat.at<double>(0, 0), rmat.at<double>(0, 1), rmat.at<double>(0, 2),
                             rmat.at<double>(1, 0), rmat.at<double>(1, 1), rmat.at<double>(1, 2),
                             rmat.at<double>(2, 0), rmat.at<double>(2, 1), rmat.at<double>(2, 2));

      // Publish the transform from the camera to the chessboard.
      tf2::Transform tf_transform(rmat_tf, tvec_tf);
      geometry_msgs::msg::TransformStamped transform_stamped;
      transform_stamped.header.stamp = now;
      transform_stamped.header.frame_id = params_->chessboard_frame;
      transform_stamped.child_frame_id = camera_frame;
      transform_stamped.transform = tf2::toMsg(tf_transform.inverse());
      tf_pub_->sendTransform(transform_stamped);

      // Update perspective transform.
      found_perspective_transform_ = true;
      vector<cv::Point2f> outer_corners;
      for (int i = 0; i < 4; ++i) outer_corners.push_back(chessboard_corners[ARUCO_LOOKUP[i]]);
      perspective_transform_ = cv::getPerspectiveTransform(outer_corners, IRL_CHESSBOARD_CORNERS);
    } else {
      RCLCPP_WARN(node->get_logger(), "Couldn't find chessboard");
    }

    if (found_perspective_transform_) {
      // Set the header for the warped image message.
      std_msgs::msg::Header img_header;
      img_header.stamp = now;
      img_header.frame_id = params_->chessboard_frame;

      // Create a perspective-transformed image of the chessboard.
      static cv::Mat warped;
      static const int BOARD_SIZE_INT = static_cast<int>(round(CHESSBOARD_SIZE));
      cv::warpPerspective(cv_ptr->image, warped, perspective_transform_,
                          cv::Size(BOARD_SIZE_INT, BOARD_SIZE_INT));

      // If we cannot currently see the chessboard, draw a red square around the image. This
      // indicates that the perspective transform may not be accurate.
      if (!found)
        cv::rectangle(warped, cv::Point(0, 0), cv::Point(BOARD_SIZE_INT, BOARD_SIZE_INT),
                      cv::Scalar(255, 0, 0), 16);

      // Publish the perspective-transformed image.
      sensor_msgs::msg::Image::SharedPtr msg =
          cv_bridge::CvImage(img_header, sensor_msgs::image_encodings::RGB8, warped).toImageMsg();
      image_pub_->publish(msg);
    }
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