cmake_minimum_required(VERSION 3.1)
project(SocketIOClientExample)

set(CMAKE_CXX_STANDARD 11)

# Set the path to the Socket.IO Client C++ library
set(SOCKET_IO_CLIENT_CPP_PATH /path/to/socket.io-client-cpp)

# Add the Socket.IO Client C++ library include directory
include_directories(${SOCKET_IO_CLIENT_CPP_PATH}/src)

# Add the Socket.IO Client C++ library source files
set(SOCKET_IO_CLIENT_CPP_SOURCE_FILES
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_packet.h
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_packet.cpp
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_packet_op.h
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_packet_op.cpp
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_packet_socket.h
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_packet_socket.cpp
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_streambuf.h
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_streambuf.cpp
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_socket.h
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_socket.cpp
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_socket_impl.h
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_socket_impl.cpp
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_client_impl.h
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/internal/sio_client_impl.cpp
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/sio_client.h
    ${SOCKET_IO_CLIENT_CPP_PATH}/src/sio_client.cpp
)

# Add the executable target
add_executable(SocketIOClientExample main.cpp ${SOCKET_IO_CLIENT_CPP_SOURCE_FILES})
