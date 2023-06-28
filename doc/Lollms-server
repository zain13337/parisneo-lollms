# Lord of Large Language Models (LoLLMs) Documentation

LoLLMs (Lord of Large Language Models) is a powerful text generation framework that leverages distributed or centralized architecture with multiple service nodes. It enables users to generate text using various models and personalities. This documentation will guide you through the installation, setup, and usage of LoLLMs.

## Installation

To install LoLLMs, follow these steps:

1. Open your terminal or command prompt.
2. Run the following command:

```shell
pip install lollms
```

## Setup

After installing LoLLMs, you need to set it up before using it. Follow the steps below:

1. Open your terminal or command prompt.
2. Run the following command:

```shell
lollms-settings
```

3. In the settings prompt, select the desired binding, such as `ctransformer`, from the available options.
4. Choose a model from the provided options.
5. Optionally, select one of the preconditioned personalities. There are 260 personalities available.
6. Save the settings.

## Starting the LoLLMs Server

To start the LoLLMs server, follow these steps:

1. Open your terminal or command prompt.
2. Run the following command:

```shell
lollms-server
```

3. This will start a local server on `localhost:9600`.

You can also run multiple LoLLMs servers on different hosts and ports. To specify a different host and port, use the following command:

```shell
lollms-server --host <hostname> --port <port>
```

Replace `<hostname>` with the desired hostname and `<port>` with the desired port number.

## Using the LoLLMs Playground

LoLLMs provides a user-friendly playground for generating text. You can use the pre-built playground or create your own code to interact with the LoLLMs server.

To use the LoLLMs playground:

1. Clone the [LoLLMs Playground repository](https://github.com/ParisNeo/lollms-playground) from GitHub.
2. Open the playground in your web browser.

## Generating Text with LoLLMs

To generate text using LoLLMs, you can use the provided socket.io interface. The following code snippet demonstrates how to generate text using the socket.io interface:

```javascript
socket.emit('generate_text', {
  prompt,
  personality: -1,
  n_predicts: 1024,
  parameters: {
    temperature: temperatureValue,
    top_k: topKValue,
    top_p: topPValue,
    repeat_penalty: repeatPenaltyValue,
    repeat_last_n: repeatLastNValue,
    seed: parseInt(seedValue)
  }
});
```

In the code snippet:

- `prompt`: The input text or prompt to generate text from.
- `personality`: The ID of the mounted personalities in the server. Use `-1` to revert to simple generation without personality support.
- `n_predicts`: The number of predicted texts to generate.
- `parameters`: Additional parameters for text generation, such as `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, and `seed`.

Feel free to adjust the parameters according to your requirements.

## Conclusion

Congratulations! You have successfully installed, set up, and used LoLLMs. You can now generate text using the provided interface or the playground. Explore the capabilities of LoLLMs and enjoy generating text with large language models!
