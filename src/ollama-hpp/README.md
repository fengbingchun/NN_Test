# ollama-hpp
Modern, Header-only C++11-20 bindings for the [Ollama](https://ollama.ai) API.

Access the full power of local language models in C++ with just a few lines of code:

```C++
#include "ollama.hpp"

std::cout << ollama::generate("llama3:8b", "Why is the sky blue?") << std::endl;
```

## Quick Start
Download the header in singleheader/ollama.hpp and include it with your project to get started. No additional files or linking are required. C++11 is the minimum required language specification and C++14/17/20 are also supported.

For example:
`g++ your_source_file.cpp -Iollama-hpp/singleheader -std=c++11`

Ensure you have an active instance of the ollama server and any models you plan to run. For more details, see https://ollama.com. You can check if your ollama server is running using: `sudo systemctl status ollama`

## Building examples
To build the ollama-hpp examples and test cases, use:
`make -j8`

To run the examples and test cases, use:

`build/test` <br>
`build/examples`

## Full API

The test cases do a good job of providing discrete examples for each of the API features supported. I recommend reviewing these first in `test/test.cpp` to understand what the library and Ollama API provide.

- [ollama-hpp](#ollama-hpp)
  - [Quick Start](#quick-start)
  - [Building examples](#building-examples)
  - [Full API](#full-api)
    - [Ollama Class and Singleton](#ollama-class-and-singleton)
    - [Ollama Response](#ollama-response)
    - [Set Server Parameters](#set-server-parameters)
    - [Get Server Status](#get-server-status)
    - [Get Server Version](#get-server-version)
    - [Load a Model into Memory](#load-a-model-into-memory)
    - [Pull, Copy, and Delete Models](#pull-copy-and-delete-models)
    - [Retrieve Model Info](#retrieve-model-info)
    - [List locally-available and running models](#list-locally-available-and-running-models)
    - [Exception Handling](#exception-handling)
    - [Basic Generation](#basic-generation)
    - [Using Options](#using-options)
    - [Streaming Generation](#streaming-generation)
    - [Asynchronous Streaming Generation](#asynchronous-streaming-generation)
    - [Using Images](#using-images)
    - [Generation using Images](#generation-using-images)
    - [Basic Chat Generation](#basic-chat-generation)
    - [Chat with Multiple Messages](#chat-with-multiple-messages)
    - [Streaming Chat Generation](#streaming-chat-generation)
    - [Chat with Images](#chat-with-images)
    - [Embedding Generation](#embedding-generation)
    - [Debug Information](#debug-information)
    - [Manual Requests](#manual-requests)
    - [Thinking](#thinking)
    - [Handling Context](#handling-context)
    - [Context Length](#context-length)
  - [Single-header vs Separate Headers](#single-header-vs-separate-headers)
  - [About this software](#about-this-software)
  - [License](#license)


### Ollama Class and Singleton
The `Ollama` class defines the logic required to interface with an ollama server.

```C++
Ollama my_server("http://localhost:11434");
std::cout << my_server.generate("llama3:8b", "Why is the sky blue?") << std::endl;
```

For convenience, a static singleton of the Ollama class is included in the `ollama` namespace which defaults to http://localhost:11434. Using the static singleton is preferred and will be easiest for most people. This allows you to make calls to a default server immediately simply by including the header. All calls to the Ollama class are also valid for the singleton:

```C++
// No object creation required; the static singleton is usable as soon as the file is included
// http://localhost:11434 is the default server location
ollama::generate("llama3:8b", "Why is the sky blue?") << std::endl;
```

### Ollama Response
The `ollama::response` class contains a response from the server. This is the default class returned for most generations. It can be flexibly represented as `nlohmann::json` or a `std::string` depending on context.

```C++
ollama::response response = ollama::generate("llama3:8b", "Why is the sky blue?");

// A JSON object created from the server response
nlohmann::json data = response.as_json();

// A string representing the JSON data received
std::string json_string = response.as_json_string();

// Usually contains just the human-readable response from the generation/chat
std::string simple_string = response.as_simple_string();
```

When interacting with streams or strings, `ollama::response` will default to using the `as_simple_string` representation. This usually contains the human-readable portion of the response.

```C++
// Will print the generated human-readable portion of response, not the JSON
std::cout << response << std::endl;
```

### Set Server Parameters
The `Ollama` object contains a series of intelligent defaults used to communicate with an ollama server. You will not typically have to change these, but can do so if required:

```C++
// Optional. By default, the server URL is set to http://localhost:11434. 
// Use this function if your server resides at a different URL.
ollama::setServerURL("http://localhost:11434");    

// Optional. Set the read and write timeouts in seconds for server interactions.
// If you have a large model with a long response time you may need to increase these.
ollama::setReadTimeout(120);
ollama::setWriteTimeout(120);
```

### Get Server Status
Verify that the Ollama server is running with `ollama::is_running()`

```C++
bool running = ollama::is_running();
```

### Get Server Version
Return a `std::string` containing the version of the ollama server using `ollama::get_version()`

```C++
std::string version = ollama::get_version();
```

### Load a Model into Memory
This can optionally be used to deliberately load a model into memory prior to use. This occurs automatically when a request is made for an unloaded model, but can be useful for loading a model in advance.

```C++
bool model_loaded = ollama::load_model("llama3:8b");
```

### Pull, Copy, and Delete Models
You can easily pull, copy, and delete models locally available within your ollama server. For the full list of models available in the Ollama library, see https://ollama.com/library.

```C++
// Pull a model by specifying a model name.
bool model_pulled = ollama::pull_model("llama3:8b");

// Copy a model by specifying a source model and destination model name.
bool model_copied = ollama::copy_model("llama3:8b", "llama3_copy");

// Delete a model by specifying a model name.
bool model_deleted = ollama::delete_model("llama3_copy");
```

### Retrieve Model Info
Model information can be pulled for a specified model name. This is returned as an `nlohmann::json` object.

```C++
// Request model info from the Ollama server.
nlohmann::json model_info = ollama::show_model_info("llama3:8b");
std::cout << "Model family is " << model_info["details"]["family"] << std::endl;
```

### List locally-available and running models
You can query a list of locally-available models on your ollama server using the following. This is returned as a `std::vector` of `std::string`.

```C++
// List the models available locally in the ollama server
std::vector<std::string> models = ollama::list_models();
```

You can similarly query a list of currently-running models on your ollama server using:

```C++
// List the models available locally in the ollama server
std::vector<std::string> models = ollama::list_running_models();
```

For detailed parameters for these models, you can obtain the verbose JSON model descriptions using `ollama::list_model_json()` and `ollama::running_model_json()`.

### Exception Handling
Most calls will throw `ollama::exception` in the event of an error, with details on the exception that has occurred. Exceptions are enabled by default.

```C++
try { 
    ollama::generate("Non-existent-model", "Requesting this model will throw an error"); 
} 
catch(ollama::exception& e) { std::cout << e.what() << std::endl; }
```

You can also dynamically enable and disable exceptions. If exceptions are disabled, functions will return an empty `ollama::response` or false where appropriate instead of throwing `ollama::exception`.

```C++ 
ollama::allow_exceptions(false);
```

### Basic Generation
A generation call can be made by specifying a model name and prompt. This will return an `ollama::response`.

```C++
ollama::response response = ollama::generate("llama3:8b", "Why is the sky blue?");

std::cout << response << std::endl;
```

As mentioned previously, you can access `ollama::response` as an `nlohmann::json` object or `std::string` depending on your preference.

The default call does not use streaming from the Ollama server, so the reply will block and be received as one response.

### Using Options
All generative calls can include options specified through an `ollama::options` object. This class extends `nlohmann::json` and can support the options specified [here](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-request-with-options).

```C++
ollama::options options;

// Access and set these options like any other json type.
options["seed"] = 1;
options["temperature"] = 0;
options["num_predict"] = 18;

// Options can be included with any generative function
ollama::response response = ollama::generate("llama3:8b", "Why is the sky blue?", options);
```

### Streaming Generation
You can use a streaming generation to bind a callback function that is invoked every time a token is received. This is useful when you have larger responses and want to show tokens as they arrive.

```C++

bool on_receive_response(const ollama::response& response)
{   
    // Print the token received
    std::cout << response << std::flush;

    // The server will set "done" to true for the last response
    if (response.as_json()["done"]==true) std::cout << std::endl;

    // Return true to continue streaming this response; return false to stop immediately.
    return true;
}

// This function will be called every token
std::function<bool(const ollama::response&)> response_callback = on_receive_response;  

// Bind the callback to the generation
ollama::generate("llama3:8b", "Why is the sky blue?", response_callback);
```
This function uses a blocking socket call so it will still block the primary thread until all tokens are received.

### Asynchronous Streaming Generation
You can launch a streaming call in a thread if you don't want it to block the primary thread. This will allow asynchronous execution.

```C++

std::atomic<bool> done{false};

bool on_receive_response(const ollama::response& response)
{   
    std::cout << response << std::flush;

    if (response.as_json()["done"]==true) { done=true;  std::cout << std::endl;}

    // Return true to continue streaming this response; return false to stop immediately.
    return !done;
}

// Use std::function to define a callback from an existing function
// You can also use a lambda with an equivalent signature
std::function<bool(const ollama::response&)> response_callback = on_receive_response;  

// You can launch the generation in a thread with a callback to use it asynchronously.
std::thread new_thread( [response_callback]{ 
  ollama::generate("llama3:8b", "Why is the sky blue?", response_callback); } );

// Prevent the main thread from exiting while we wait for an asynchronous response.
while (!done) { std::this_thread::sleep_for(std::chrono::microseconds(100) ); }
new_thread.join();
```
The return value of the function determines whether to continue streaming or stop. This is useful in cases where you want to stop immediately instead of waiting for an entire response to return.

### Using Images
Generations can include images for vision-enabled models such as `llava`. The `ollama::image` class can load an image from a file and encode it as a [base64](https://en.wikipedia.org/wiki/Base64) string.

```C++
ollama::image image = ollama::image::from_file("llama.jpg");
```

Images can also be represented as a base64 string literal.

```C++
ollama::image base64_image = ollama::image::from_base64_string("iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz0AEYBxVSF+FAP5FDvcfRYWgAAAAAElFTkSuQmCC");
```

### Generation using Images
Generative calls can also include images. 

```C++
ollama::image image = ollama::image::from_file("llama.jpg");

ollama::response response = 
  ollama::generate("llava", "What do you see in this image?", options, image);
```

Multiple images can be included using the `ollama::images` container.

```C++
ollama::image image = ollama::image::from_file("llama.jpg");
ollama::image image2 = ollama::image::from_file("another_llama.jpg");

// Include a list of images here
ollama::images images={image, image2};

ollama::response response = 
  ollama::generate("llava", "What do you see in these images?", options, images);
```

### Basic Chat Generation
The Ollama chat API can be used as an alternative to basic generation. This allows the user to send a series of messages to the server and obtain the next response in the conversation.

`ollama::message` represents a single chat message in the conversation. It is composed of a role, content, and an optional series of images.

```C++
ollama::message message("user", "Why is the sky blue?");
```

Sending a message to the server will return the next message in the conversation.
```C++
ollama::message message("user", "Why is the sky blue?");

ollama::response response = ollama::chat("llama3:8b", message);
```

Like any generative call, chat calls can also include options.

```C++
ollama::options options;
options["seed"] = 1;

ollama::message message("user", "Why is the sky blue?");

ollama::response response = ollama::chat("llama3:8b", message, options);
```

### Chat with Multiple Messages

You can use a collection of messages in a chat. This allows chain-of-thought prompting and can be useful for setting up a conversation.

```C++
ollama::message message1("user", "What are nimbus clouds?");
ollama::message message2("assistant", "Nimbus clouds are dark rain clouds.");
ollama::message message3("user", "What are some other kinds of clouds?");

ollama::messages messages = {message1, message2, message3};

ollama::response response = ollama::chat("llama3:8b", messages);
```

### Streaming Chat Generation
The default chat generation does not stream tokens and will return the entire reply as one response. You can bind a callback function to handle a streamed response for each token, just like a standard generation.

```C++

bool on_receive_response(const ollama::response& response)
{   
  std::cout << response << std::flush;

  if (response.as_json()["done"]==true) std::cout << std::endl;

  // Return true to continue streaming, or false to stop immediately
  return true;
}

std::function<bool(const ollama::response&)> response_callback = on_receive_response;  

ollama::message message("user", "Why is the sky blue?");       

ollama::chat("llama3:8b", message, response_callback, options);
```

### Chat with Images
The `ollama::message` class can contain an arbitrary number of `ollama::image` objects to be sent to the server. This allows images to be sent in each message of a chat for vision-enabled models. 

```C++
ollama::image image = ollama::image::from_file("llama.jpg");

// We can optionally include images with each message. 
//Vision-enabled models will be able to utilize these.
ollama::message message_with_image("user", "What do you see in this image?", image);
ollama::response response = ollama::chat("llava", message_with_image);
```
### Embedding Generation
Embeddings can be generated from a specified model name and prompt.

```C++
ollama::response response = ollama::generate_embeddings("llama3:8b", "Why is the sky blue?");
```

Like any other generative function, options can be included during generation.

```C++
ollama::options options;
options["num_predict"] = 20;

ollama::response response = 
  ollama::generate_embeddings("llama3:8b", "Why is the sky blue?", options);
```

### Debug Information
Debug logging for requests and replies to the server can easily be turned on and off. This is useful if you want to see the actual JSON sent and received from the server.

```C++
ollama::show_requests(true);
ollama::show_replies(true);
```

### Manual Requests
For those looking for greater control of the requests sent to the ollama server, manual requests can be created through the `ollama::request` class. This class extends `nlohmann::json` and can be treated as a standard JSON object.

```C++
ollama::request request(ollama::message_type::generation);
request["model"]="mistral";
request["prompt"]="Why is the sky blue?";
request["system"] = "Talk like a pirate for the next reply."
std::cout << ollama::generate(request) << std::endl;
```
This provides the most customization of the request. Users should take care to ensure that valid fields are provided, otherwise an exception will likely be thrown on response. Manual requests can be made for generate, chat, and embedding endpoints.

### Thinking
Models which support [thinking](https://ollama.com/blog/thinking) can be supported through manual requests:

```C++
ollama::request request(ollama::message_type::generation);

request["think"]=true;
request["model"]="qwen3:14b";
request["prompt"]="Why is the sky blue?";
request["system"]="Respond as though you are an experienced meteorologist.";

ollama::response response = ollama::generate(request);

std::cout << "Model thoughts:" << response.as_json()["thinking"] << std::endl;
std::cout << "Model response:" << response << std::endl;
```
Model thoughts can also be handled with a streaming response in order to see tokens from thought in real time.

### Handling Context
Context from previous generate requests can be used by including a past `ollama::response` with `generate`:

**Note:** The Context parameter has been [deprecated](https://github.com/ollama/ollama/pull/7878) in newer versions of Ollama. It is recommended to use the `chat` endpoint for maintaining context over time.
```C++
std::string model = "llama3.1:8b";
ollama::response context = ollama::generate(model, "Why is the sky blue?");
ollama::response response = ollama::generate(model, "Tell me more about this.", context);
```

This will provide the past user prompt and response to the model when making a new generation. Context can be chained over multiple messages and will contain the entire conversation history from the first prompt:

```C++
ollama::response first_response = ollama::generate(model, "Why is the sky blue?");
ollama::response second_response = ollama::generate(model, "Tell me more about this.", first_response);
ollama::response third_response = ollama::generate(model, "What was the first question that I asked you?", second_response);
```

Context can also be added as JSON when creating manual requests:
```C++
ollama::response response = ollama::generate("llama3.1:8b", "Why is the sky blue?");

ollama::request request(ollama::message_type::generation);
request["model"]="llama3.1:8b";
request["prompt"]="Why is the sky blue?";
request["context"] = response.as_json()["context"];
std::cout << ollama::generate(request) << std::endl;
```

Note that the `chat` endpoint has no specialized context parameter; context is simply supplied through the message history of the conversation:

```C++
ollama::message message1("user", "What are nimbus clouds?");
ollama::message message2("assistant", "Nimbus clouds are dense, moisture-filled clouds that produce rain.");
ollama::message message3("user", "What was the first question I asked you?");

ollama::messages messages = {message1, message2, message3};

std::cout << ollama::chat("llama3.1:8b", messages) << std::endl;
```
### Context Length
Most language models have a maximum input context length that they can accept. This length determines the number of previous tokens that can be provided along with the prompt as an input to the model before information is lost. Llama 3.1, for example, has a maximum context length of 128k tokens; a much smaller number of <b>2048</b> tokens is often enabled by default from Ollama in order to reduce memory usage. You can increase the size of the context window using the `num_ctx` parameter in `ollama::options` for tasks where you need to retain a long conversation history:

```C++
// Set the size of the context window to 8192 tokens.
ollama::options options;
options["num_ctx"] = 8192; 

// Perform a simple generation which includes model options.
std::cout << ollama::generate("llama3.1:8b", "Why is the sky blue?", options) << std::endl;
```

Keep in mind that increasing context length will increase the model size in memory when loading to a GPU. You should ensure your hardware has sufficient memory to hold the larger model when configuring for long-context tasks.

## Single-header vs Separate Headers
For convenience, ollama-hpp includes a single-header version of the library in `singleheader/ollama.hpp` which bundles the core ollama.hpp code with single-header versions of nlohmann json, httplib, and base64.h. Each of these libraries is available under the MIT license and their respective licenses are included.
The single-header include can be regenerated from these standalone files by running `./make_single_header.sh`

If you prefer to include the headers for these libraries separately, you can do so by including the standard header located in `include/ollama.hpp`. 

## About this software

Ollama is a high-quality REST server and API providing an interface to run language models locally via llama.cpp.

Ollama was made by Jeffrey Morgan (@jmorganca) and the Ollama team and is available under the MIT License. To support this project or for more details go to https://github.com/ollama or https://ollama.ai 

This library is a header-only C++ integration of the Ollama API providing access to most API features while integrating them with std library classes or popular header-only libraries within the community. The following external libraries are used:

* **nlohmnann JSON** is a feature-rich header-only C++ JSON implementation. This library was created by Niels Lohmann and is available under the MIT License. For more details visit: https://github.com/nlohmann/json

* **httplib** is a header-only C++ http/https library. This library was created by Yuji Hirose and is available under the MIT License. For more details visit: https://github.com/yhirose/cpp-httplib

* **Base64.h** is a header-only C++ library for encoding and decoding Base64 values. This library was created by tomykaira and is available under the MIT License. For more details visit: https://gist.github.com/tomykaira/f0fd86b6c73063283afe550bc5d77594

## License
MIT License

Copyright (c) 2025 James Montgomery (jmont)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

