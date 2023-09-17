#include <iostream>
#include <string>
#include <sio_client.h>
#include <iostream>
#include <string>
#include <thread>
#include <ASCIIColors.h>
using namespace sio;


class lollmsClient {
public:
    lollmsClient(const std::string& serverUrl) : connected_(false)
    {
        // Set up event listeners
        setupEventListeners();

        std::cout<<"Built listeners"<<std::endl;
        client_.set_logs_quiet();
        // Connect to the server
        client_.connect(serverUrl);
    }

    void generateText(const std::string& prompt, int n_predicts=128)
    {
        if (connected_) {
            object_message::ptr om = object_message::create();
            om->get_map()["prompt"]=sio::string_message::create(prompt);
            om->get_map()["n_predicts"]=sio::int_message::create(n_predicts);
            client_.socket()->emit("generate_text", om);            
            // client_.socket()->emit("generate_text", sio::object_message(*message));

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
        std::cout<<"Adding open listener"<<std::endl;
        client_.set_open_listener([&]() {
            onConnected();
        });

        std::cout<<"Adding close listener"<<std::endl;
        client_.set_close_listener([&](sio::client::close_reason const& reason) {
            onDisconnected();
        });

        std::cout<<"Adding fail listener"<<std::endl;
        client_.set_fail_listener([&]() {
            onConnectionFailed();
        });

        std::cout<<"Adding reconnect listener"<<std::endl;
        client_.set_reconnect_listener([&](unsigned int reconnectionAttempts, unsigned int delay) {
            onReconnecting(reconnectionAttempts, delay);
        });
        /*
        std::cout<<"Adding socket close listener"<<std::endl;
        client_.set_socket_close_listener((const sio::client::socket_listener &)[&]() {
            onSocketClosed();
        });

        */

        std::cout<<"Adding lollms server listener"<<std::endl;
        // Event handler for receiving generated text chunks
        client_.socket()->on("text_chunk", [&](const sio::event& event) {
            sio::message::ptr message = event.get_message();
            if (message->get_map().find("chunk") != message->get_map().end()) {
                sio::message::ptr chunkMessage = message->get_map()["chunk"];
                if (chunkMessage->get_flag() == sio::message::flag_string) {
                    std::string chunk = chunkMessage->get_string();
                    std::cout << chunk;
                    // Append the chunk to the output or perform any other required actions
                    // ...
                } else {
                    std::cerr << "Received 'chunk' data is not a string." << std::endl;
                }
            } else {
                std::cerr << "Received event 'text_chunk' without 'chunk' data." << std::endl;
            }
        });

        // Event handler for receiving generated text
        client_.socket()->on("text_generated", [&](const sio::event& event) {
            sio::message::ptr message = event.get_message();
            if (message->get_map().find("text") != message->get_map().end()) {
                sio::message::ptr chunkMessage = message->get_map()["text"];
                if (chunkMessage->get_flag() == sio::message::flag_string) {
                    std::string chunk = chunkMessage->get_string();
                    std::cout << chunk;
                    // Append the chunk to the output or perform any other required actions
                    // ...
                } else {
                    std::cerr << "Received 'text' data is not a string." << std::endl;
                }
            } else {
                std::cerr << "Received event 'text_generated' without 'text' data." << std::endl;
            }
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
            std::cout << "Generation canceled" << std::endl;
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
    ASCIIColors::red(R"(      ___       ___           ___       ___       ___           ___      )");
    ASCIIColors::red(R"(     /\__\     /\  \         /\__\     /\__\     /\__\         /\  \     )");
    ASCIIColors::red(R"(    /:/  /    /::\  \       /:/  /    /:/  /    /::|  |       /::\  \    )");
    ASCIIColors::red(R"(   /:/  /    /:/\:\  \     /:/  /    /:/  /    /:|:|  |      /:/\ \  \   )");
    ASCIIColors::red(R"(  /:/  /    /:/  \:\  \   /:/  /    /:/  /    /:/|:|__|__   _\:\~\ \  \  )");
    ASCIIColors::red(R"( /:/__/    /:/__/ \:\__\ /:/__/    /:/__/    /:/ |::::\__\ /\ \:\ \ \__\ )");
    ASCIIColors::red(R"( \:\  \    \:\  \ /:/  / \:\  \    \:\  \    \/__/~~/:/  / \:\ \:\ \/__/ )");
    ASCIIColors::red(R"(  \:\  \    \:\  /:/  /   \:\  \    \:\  \         /:/  /   \:\ \:\__\   )");
    ASCIIColors::red(R"(   \:\  \    \:\/:/  /     \:\  \    \:\  \       /:/  /     \:\/:/  /   )");
    ASCIIColors::red(R"(    \:\__\    \::/  /       \:\__\    \:\__\     /:/  /       \::/  /    )");
    ASCIIColors::red(R"(     \/__/     \/__/         \/__/     \/__/     \/__/         \/__/     )");

    ASCIIColors::yellow("By ParisNeo");


    
    ASCIIColors::red("Lollms c++ Client V 1.0");
    // Create a lollmsClient instance and connect to the server
    lollmsClient client("http://localhost:9601");
    std::cout<<"Created"<<std::endl;
    // Wait for the connection to be established before sending events
    while (!client.getClient().opened())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait for 100ms
    }
    std::cout<<"Opened"<<std::endl;

    // Trigger the "generate_text" event when needed
    std::string prompt = "SYSTEM:Act as assiatant, a great and helpful AI agent that does multiple tasks. Help user do what he needs to do.\nUser:write a python hello word code\nAssistant:";
    client.generateText(prompt,1024);

    // Trigger the "cancel_generation" event when needed
    // client.cancelGeneration();

    // Run the event loop to keep the connection alive
    while (true)
    {
        // You can add some logic here to break the loop when needed
        // For example, when the user wants to exit the program
    }
    std::cout<<"Done"<<std::endl;

    return 0;
}