#include <iostream>
#include <string>
#include <sio_client.h>
#include <iostream>
#include <string>
#include <thread>

using namespace sio;


class SocketIOClient {
public:
    SocketIOClient(const std::string& serverUrl) : connected_(false)
    {
        // Set up event listeners
        setupEventListeners();

        // Connect to the server
        client_.connect(serverUrl);
    }

    void generateText(const std::string& prompt)
    {
        if (connected_) {
        } else {
            std::cerr << "Not connected to the server. Cannot generate text." << std::endl;
        }
    }

    void cancelGeneration()
    {
        if (connected_) {
            client_.socket()->emit("cancel_generation");
        } else {
            std::cerr << "Not connected to the server. Cannot cancel generation." << std::endl;
        }
    }

    // Getter for client_
    const sio::client& getClient() const {
        return client_;
    }
    void closeConnection() {
        client_.close(); // Ou utilisez une autre méthode de fermeture selon la bibliothèque sio
    }    

private:
    client client_;
    bool connected_;

    void setupEventListeners()
    {
        client_.set_open_listener([&]() {
            onConnected();
        });

        client_.set_close_listener([&](sio::client::close_reason const& reason) {
            onDisconnected();
        });

        client_.set_fail_listener([&]() {
            onConnectionFailed();
        });

        client_.set_reconnect_listener([&](unsigned int reconnectionAttempts, unsigned int delay) {
            onReconnecting(reconnectionAttempts, delay);
        });

        client_.set_socket_close_listener((const sio::client::socket_listener &)[&]() {
            onSocketClosed();
        });

        // Event handler for receiving generated text chunks
        client_.socket()->on("text_chunk", [&](const sio::event& event) {
            const std::string chunk = event.get_message()->get_string();
            std::cout << "Received chunk: " << chunk << std::endl;
            // Append the chunk to the output or perform any other required actions
            // ...
        });

        // Event handler for receiving generated text
        client_.socket()->on("text_generated", [&](const sio::event& event) {
            const std::string text = event.get_message()->get_string();
            std::cout << "Text generated: " << text << std::endl;
            // Toggle button visibility or perform any other required actions
            // ...
        });

        // Event handler for error during text generation
        client_.socket()->on("buzzy", [&](const sio::event& event) {
            const std::string error = event.get_message()->get_string();
            std::cerr << "Server is busy. Wait for your turn. Error: " << error << std::endl;
            // Handle the error or perform any other required actions
            // ...
        });

        // Event handler for generation cancellation
        client_.socket()->on("generation_canceled", [&](const sio::event& event) {
            // Toggle button visibility or perform any other required actions
            // ...
        });
    }

    void onConnected()
    {
        std::cout << "Connected to the LoLLMs server" << std::endl;
        connected_ = true;
        // Perform actions upon successful connection
        // ...
    }

    void onDisconnected()
    {
        std::cout << "Disconnected from the server" << std::endl;
        connected_ = false;
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
        connected_ = false;
        // Perform actions upon socket closure
        // ...
    }
    
    

};

int main()
{
    // Create a SocketIOClient instance and connect to the server
    SocketIOClient client("http://localhost:9601");
    std::cout<<"Created"<<std::endl;
    // Wait for the connection to be established before sending events
    while (!client.getClient().opened())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait for 100ms
    }
    std::cout<<"Opened"<<std::endl;

    // Trigger the "generate_text" event when needed
    std::string prompt = "Enter your prompt here";
    client.generateText(prompt);

    // Trigger the "cancel_generation" event when needed
    client.cancelGeneration();

    // Run the event loop to keep the connection alive
    while (true)
    {
        // You can add some logic here to break the loop when needed
        // For example, when the user wants to exit the program
    }
    std::cout<<"Done"<<std::endl;

    return 0;
}