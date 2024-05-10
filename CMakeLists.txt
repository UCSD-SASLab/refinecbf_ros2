cmake_minimum_required(VERSION 3.8)
project(refinecbf_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(std_srvs REQUIRED)
find_package(crazyflie_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

# ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/disturbance_node.py
  scripts/obstacle_node.py
  scripts/refine_cbf_node.py
  scripts/hj_reachability_node.py
  scripts/modify_environment.py
  scripts/template/nominal_controller.py
  scripts/config.py
  scripts/crazyflie/cf_nominal_controller.py
  scripts/crazyflie/cf_hw_interface.py
  scripts/crazyflie/cf_visualization.py
  scripts/template/nominal_controller.py
  scripts/config.py
  DESTINATION lib/${PROJECT_NAME}  
)


rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Array.msg"
  "msg/ControlArray.msg"
  "msg/HiLoArray.msg"
  "msg/Obstacle.msg"
  "msg/Obstacles.msg"
  "msg/ValueFunctionMsg.msg"
  "srv/ActivateObstacle.srv"
  "srv/ModifyEnvironment.srv"
  "srv/HighLevelCommand.srv"
 )

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
