cmake_minimum_required(VERSION 3.8)
project(chessboard_transform)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs  REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(OpenCV REQUIRED)

generate_parameter_library(
  chessboard_transform_params
  src/parameters.yaml
)

add_executable(chessboard_transform src/chessboard_transform.cpp)
target_include_directories(chessboard_transform PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(chessboard_transform PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(chessboard_transform PUBLIC
  cv_bridge
  image_transport
  rclcpp
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_include_directories(chessboard_transform PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(chessboard_transform PRIVATE
  chessboard_transform_params
)

target_link_libraries(chessboard_transform PUBLIC
  ${OpenCV_LIBS}
)

install(TARGETS chessboard_transform
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
