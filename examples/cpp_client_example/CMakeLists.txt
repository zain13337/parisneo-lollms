cmake_minimum_required(VERSION 3.12)
project(LollmsClientApp)

# Include ExternalProject module
include(ExternalProject)

# Set up the external project
ExternalProject_Add(
    socketio-client-cpp
    PREFIX ${CMAKE_BINARY_DIR}/external
    GIT_REPOSITORY https://github.com/socketio/socket.io-client-cpp.git
    GIT_TAG master # You can change this to a specific release/tag if needed
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DSIO_DISABLE_LOGGING=ON # Add this line to disable logging
)

# Add your project's executable or library here
add_executable(LollmsClientApp main.cpp src/ASCIIColors.cpp)

# Include the socket.io-client-cpp header files
target_include_directories(LollmsClientApp PRIVATE 
    ${CMAKE_BINARY_DIR}/external/include
    ${CMAKE_BINARY_DIR}/external/include/socket.io-client-cpp # Add this line
)
# This is required for googletest
find_package(sioclient)
# Link your project with the socket.io-client-cpp library
target_include_directories(LollmsClientApp PRIVATE ${CMAKE_BINARY_DIR}/external/include)
target_link_directories(LollmsClientApp PRIVATE ${CMAKE_BINARY_DIR}/external/lib)
target_include_directories(LollmsClientApp PRIVATE ./inc)
target_link_libraries(LollmsClientApp PRIVATE sioclient)
