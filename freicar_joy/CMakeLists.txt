cmake_minimum_required(VERSION 3.1)
project(freicar_joy)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(catkin REQUIRED COMPONENTS
	roscpp
	roslib
	sensor_msgs
	raiscar_msgs
)

catkin_package(CATKIN_DEPENDS 
	roscpp
	roslib
	sensor_msgs
	raiscar_msgs
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	# ${SUBDIRS}
)
# freicar_agent node -------------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_node src/main.cpp)
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${CMAKE_THREAD_LIBS_INIT}
										   ${catkin_LIBRARIES}
)