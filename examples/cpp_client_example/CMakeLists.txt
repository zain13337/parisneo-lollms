cmake_minimum_required(VERSION 3.12)
project(MySocketIOApp)

# Include ExternalProject module
include(ExternalProject)

# Set up the external project
ExternalProject_Add(
    socketio-client-cpp
    PREFIX ${CMAKE_BINARY_DIR}/external
    GIT_REPOSITORY https://github.com/socketio/socket.io-client-cpp.git
    GIT_TAG master # You can change this to a specific release/tag if needed
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)

# Add your project's executable or library here
add_executable(MySocketIOApp main.cpp)

# Include the socket.io-client-cpp header files
target_include_directories(MySocketIOApp PRIVATE 
    ${CMAKE_BINARY_DIR}/external/include
    ${CMAKE_BINARY_DIR}/external/include/socket.io-client-cpp # Add this line
)
# This is required for googletest
find_package(sioclient)
# Link your project with the socket.io-client-cpp library
target_include_directories(MySocketIOApp PRIVATE ${CMAKE_BINARY_DIR}/external/include)
target_link_directories(MySocketIOApp PRIVATE ${CMAKE_BINARY_DIR}/external/lib)
target_link_libraries(MySocketIOApp PRIVATE sioclient)
