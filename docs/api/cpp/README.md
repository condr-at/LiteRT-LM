# C++ API Overview

The LiteRT-LM C++ API provides a powerful and flexible way to run language models on edge devices. It is designed for high performance and gives you fine-grained control over the model execution.

The C++ API is composed of a few key components:

*   **Engine**: The `Engine` is the main entry point for the API. It is responsible for loading the model and creating sessions.
*   **Session**: A `Session` represents a single conversation with the model. It holds the state of the conversation and provides methods for running the model.
*   **Conversation**: The `Conversation` class is a high-level wrapper around the `Session` that provides a more convenient way to interact with the model.

For more details on each of these components, please see the following documents:

*   [Conversation API](./conversation.md)
*   [Constrained Decoding](./constrained-decoding.md)
*   [Tool Use](./tool-use.md)
*   [Advanced: ANTLR for Tool Use](./tool-use-antlr.md)
