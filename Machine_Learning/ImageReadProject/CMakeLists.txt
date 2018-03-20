cmake_minimum_required(VERSION 2.8.9)
project(main)

#Bring the headers, such as Student.h into the project


#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")

find_package( OpenCV REQUIRED )
include_directories( include ${OpenCV_INCLUDE_DIRS} )

add_executable(main ${SOURCES})
target_link_libraries( main ${OpenCV_LIBS} )
