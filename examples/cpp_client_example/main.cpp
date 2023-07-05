#include <iostream>
#include <string>
#include <socket.io-client-cpp/src/sio_client.h>

class SocketIOClient {
public:
    SocketIOClient(const std::string& serverUrl) : client_(sio::socket::create())
    {
        client_->set_open_listener([&]() {
            onConnected();
        });

        client_->set_close_listener([&](sio::client::close_reason const& reason) {
            onDisconnected();
        });

        client_->set_fail_listener([&]() {
            onConnectionFailed();
        });

        client_->set_reconnect_listener([&](unsigned int reconnectionAttempts, unsigned int delay) {
            onReconnecting(reconnectionAttempts, delay);
        });

        client_->set_socket_close_listener([&]() {
            onSocketClosed();
        });

        client_->connect(serverUrl);
    }

    void generateText(const std::string& prompt)
    {
        sio::message::list messages;
        messages.push(sio::string_message::create(prompt));
        messages.push(sio::int_message::create(-1));
        messages.push(sio::int_message::create(1024));
        client_->socket()->emit("generate_text", messages);
    }

    void cancelGeneration()
    {
        client_->socket()->emit("cancel_generation");
    }

    void onConnected()
    {
        std::cout << "Connected to the LoLLMs server" << std::endl;
        // Perform actions upon successful connection
        // ...
    }

    void onDisconnected()
    {
        std::cout << "Disconnected from the server" << std::endl;
        // Perform actions upon disconnection
        // ...
    }

    void onConnectionFailed()
    {
        std::cout << "Connection to the server failed" << std::endl;
        // Perform actions upon connection failure
        // ...
    }

    void onReconnecting(unsigned int reconnectionAttempts, unsigned int delay)
    {
        std::cout << "Reconnecting to the server (attempt " << reconnectionAttempts << ") in " << delay << " milliseconds" << std::endl;
        // Perform actions upon reconnection attempt
        // ...
    }

    void onSocketClosed()
    {
        std::cout << "Socket closed" << std::endl;
        // Perform actions upon socket closure
        // ...
    }

private:
    sio::client client_;
};

int main()
{
    // Create a SocketIOClient instance and connect to the server
    SocketIOClient client("http://localhost:9601");

    // Event handler for receiving generated text chunks
    client.client_->socket()->on("text_chunk", [&](const sio::event& event) {
        const std::string chunk = event.get_message()->get_string();
        std::cout << "Received chunk: " << chunk << std::endl;
        // Append the chunk to the output or perform any other required actions
        // ...
    });

    // Event handler for receiving generated text
    client.client_->socket()->on("text_generated", [&](const sio::event& event) {
        const std::string text = event.get_message()->get_string();
        std::cout << "Text generated: " << text << std::endl;
        // Toggle button visibility or perform any other required actions
        // ...
    });

    // Event handler for error during text generation
    client.client_->socket()->on("buzzy", [&](const sio::event& event) {
        const std::string error = event.get_message()->get_string();
        std::cerr << "Server is busy. Wait for your turn. Error: " << error << std::endl;
        // Handle the error or perform any other required actions
        // ...
    });

    // Event handler for generation cancellation
    client.client_->socket()->on("generation_canceled", [&](const sio::event& event) {
        // Toggle button visibility or perform any other required actions
        // ...
    });

    // Trigger the "generate_text" event when needed
    std::string prompt = "Enter your prompt here";
    client.generateText(prompt);

    // Trigger the "cancel_generation" event when needed
    client.cancelGeneration();

    // Run the event loop
    client.client_->sync_close();

    return 0;
}
