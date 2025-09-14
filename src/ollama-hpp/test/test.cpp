#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "ollama.hpp"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <string>

// Use a seed and 0 temperature to generate deterministic outputs. num_predict determines the number of tokens generated.
// Note that this is static. We will use these options for other generations.
static ollama::options options;

static std::string test_model = "llama3:8b", image_test_model = "llava";

TEST_SUITE("Ollama Tests") {

    TEST_CASE("Initialize Options") {

        ollama::show_requests(false);
        ollama::show_replies(false);
        ollama::allow_exceptions(true);

        options["seed"] = 1;
        options["temperature"] = 0;
        options["num_predict"] = 18;

        CHECK(true);
    }

    TEST_CASE("Check if Ollama is Running") {

        CHECK( ollama::is_running() );
    }

    TEST_CASE("Get Version") {

        CHECK( ollama::get_version()!="" );
    }

    TEST_CASE("Set Server Parameters") {

        // Optional. By default, the server URL is set to http://localhost:11434. Use this function if you need to point to a different URL.
        ollama::setServerURL("http://localhost:11434");    

        // Optional. Set the read and write timeouts in seconds for receiving from and sending data to ollama.
        // If you have a large model with a long response time you may need to increase these.
        ollama::setReadTimeout(120);
        ollama::setWriteTimeout(120);

        CHECK(true);
    }

    TEST_CASE("Load Model") {

        CHECK( ollama::load_model(test_model) );
    }

    TEST_CASE("Pull, Copy, and Delete Models") {

        // Pull a model by specifying a model name.
        CHECK( ollama::pull_model(test_model) == true );

        // Copy a model by specifying a source model and destination model name.
        CHECK( ollama::copy_model(test_model, test_model+"_copy") ==true );

        // Delete a model by specifying a model name.
        CHECK( ollama::delete_model(test_model+"_copy") == true );
    }

    TEST_CASE("Model Info") {

        // Request model info from the Ollama server.
        nlohmann::json model_info = ollama::show_model_info("llama3:8b");
        //std::cout << "Model family is " << model_info["details"]["family"] << std::endl;

        CHECK(model_info["details"]["family"] == "llama");
    }

    TEST_CASE("List Local Models") {

        // List the models available locally in the ollama server
        std::vector<std::string> models = ollama::list_models();    

        bool contains_model = (std::find(models.begin(), models.end(), test_model) != models.end() );

        CHECK( contains_model );
    }

    TEST_CASE("List Running Models") {

        // List the models available locally in the ollama server
        std::vector<std::string> models = ollama::list_running_models();    

        bool contains_model = (std::find(models.begin(), models.end(), test_model) != models.end() );

        CHECK( contains_model );
    }

    TEST_CASE("Exception Handling") {

        bool exception_handled = false;

        // If exceptions are enabled, functions will throw ollama::exception if invalid parameters are used or an invalid response is received.
        try { 
            ollama::generate("Non-existent-model", "Requesting this model will throw an error"); 
        } 
        catch(ollama::exception& e) { exception_handled = true; }

        CHECK( exception_handled );
    }

    TEST_CASE("Basic Generation") {

        ollama::response response = ollama::generate(test_model, "Why is the sky blue?", options);

        CHECK( response.as_json().contains("response") == true );
    }

    TEST_CASE("Generation with Context") {

        ollama::response context = ollama::generate(test_model, "Why is the sky blue?", options);

        ollama::response response = ollama::generate(test_model, "Tell me more about this.", context, options);

        CHECK( response.as_json().contains("response") == true );
    }

    std::atomic<bool> done{false};
    std::string streamed_response;


    bool on_receive_response(const ollama::response& response)
    {   
        streamed_response+=response.as_simple_string();
        if (response.as_json()["done"]==true) done=true;

        return !done;
    }


    TEST_CASE("Streaming Generation") {

        streamed_response=""; done.store(false);

        std::function<bool(const ollama::response&)> response_callback = on_receive_response;
        ollama::generate(test_model, "Why is the sky blue?", response_callback, options);

        std::string expected_response = "What a great question!\n\nThe sky appears blue because of a phenomenon called Rayleigh scattering,";

        CHECK( streamed_response != "" );
    }

    TEST_CASE("Streaming Generation with Context") {

        streamed_response=""; done.store(false);

        ollama::response context = ollama::generate(test_model, "Why is the sky blue?", options);

        std::function<bool(const ollama::response&)> response_callback = on_receive_response;  
        ollama::generate(test_model, "Tell me more about this.", context, response_callback, options);

        CHECK( streamed_response!="" );
    }

    TEST_CASE("Non-Singleton Generation") {

        Ollama my_ollama_server("http://localhost:11434");

        // You can use all of the same functions from this instanced version of the class.
        ollama::response response = my_ollama_server.generate(test_model, "Why is the sky blue?", options);

        std::string expected_response = "What a great question!\n\nThe sky appears blue because of a phenomenon called Rayleigh scattering,";

        CHECK(response.as_json().contains("response") == true);
    }

    TEST_CASE("Single-Message Chat") {

        ollama::message message("user", "Why is the sky blue?");

        ollama::response response = ollama::chat(test_model, message, options);

        std::string expected_response = "What a great question!\n\nThe sky appears blue because of a phenomenon called Rayleigh scattering,";

        CHECK(response.as_json().contains("message") == true);
    }

    TEST_CASE("Multi-Message Chat") {

        ollama::message message1("user", "What are nimbus clouds?");
        ollama::message message2("assistant", "Nimbus clouds are dense, moisture-filled clouds that produce rain.");
        ollama::message message3("user", "What are some other kinds of clouds?");

        ollama::messages messages = {message1, message2, message3};

        ollama::response response = ollama::chat(test_model, messages, options);

        std::string expected_response = "";

        CHECK(response.as_json().contains("message") == true);
    }

    TEST_CASE("Chat with Streaming Response") {

        ollama::show_requests(false);
        ollama::show_replies(false);

        streamed_response="";
        done.store(false);

        std::function<bool(const ollama::response&)> response_callback = on_receive_response;  
        
        ollama::message message("user", "Why is the sky blue?");       
        
        ollama::chat(test_model, message, response_callback, options);

        CHECK(streamed_response!="");
    }

    TEST_CASE("Chat with Asynchronous Streaming Response") {

        ollama::show_requests(false);
        ollama::show_replies(false);

        streamed_response="";
        done.store(false);

        std::function<bool(const ollama::response&)> response_callback = on_receive_response;  
        
        ollama::message message("user", "Why is the sky blue?");       
        
        std::thread new_thread( [message, response_callback]{ ollama::chat(test_model, message, response_callback, options); } );

        while (!done) { std::this_thread::sleep_for(std::chrono::microseconds(100) ); }
        new_thread.join();

        CHECK(streamed_response!="");
    }

    TEST_CASE("Chat with Asynchronous Interrupted Streaming Response") {

        ollama::show_requests(false);
        ollama::show_replies(false);

        streamed_response="";
        done.store(false);

        std::function<bool(const ollama::response&)> response_callback = on_receive_response;  
        
        ollama::message message("user", "Why is the sky blue?");       
        
        std::thread new_thread( [message, response_callback]{ ollama::chat(test_model, message, response_callback, options); } );

        unsigned int microsec_waited = 0;

        // Interrupt the stream by setting the atomic return variable false after two seconds.
        while (!done) { std::this_thread::sleep_for(std::chrono::microseconds(100) ); microsec_waited+=100; if (microsec_waited==2000000) { done.store(true); } }
        new_thread.join();

        CHECK(streamed_response!="");
    }

    TEST_CASE("Generation with Image") {

        ollama::show_requests(false);

        options["num_predict"] = 12;

        ollama::image image = ollama::image::from_file("llama.jpg");

        ollama::response response = ollama::generate(image_test_model, "What do you see in this image?", options, image);

        CHECK( response.as_json().contains("response") == true );
    }

    TEST_CASE("Generation with Multiple Images") {

        ollama::show_requests(false);

        options["num_predict"] = 14;

        ollama::image image = ollama::image::from_file("llama.jpg");
        ollama::image base64_image = ollama::image::from_base64_string("iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz0AEYBxVSF+FAP5FDvcfRYWgAAAAAElFTkSuQmCC");

        ollama::images images={image, base64_image};

        ollama::response response = ollama::generate(image_test_model, "What do you see in this image?", options, images);
        std::string expected_response = " The image features a large, fluffy white and gray llama";

        CHECK(response.as_json().contains("response") == true);
    }

    TEST_CASE("Chat with Image") {

        ollama::show_requests(false);

        options["num_predict"] = 12;

        ollama::image image = ollama::image::from_file("llama.jpg");

        // We can optionally include images with each message. Vision-enabled models will be able to utilize these.
        ollama::message message_with_image("user", "What do you see in this image?", image);
        ollama::response response = ollama::chat(image_test_model, message_with_image, options);

        std::string expected_response = " The image features a large, fluffy white llama";

        CHECK(response.as_json().contains("message") == true);
    }

    TEST_CASE("Embedding Generation") {

        options["num_predict"] = 18;

        ollama::response response = ollama::generate_embeddings(test_model, "Why is the sky blue?");

        CHECK(response.as_json().contains("embeddings") == true);
    }

    TEST_CASE("Manual Requests") {

        ollama::request request(ollama::message_type::generation);
        request["model"] = test_model;
        request["prompt"] = "Why is the sky blue?";
        request["stream"] = false;
        ollama::response response = ollama::generate(request);

        CHECK(response.as_json().contains("response") == true);

        request = ollama::request(ollama::message_type::chat);
        request["model"] = test_model;
        ollama::messages messages = { ollama::message("user","Why is the sky blue?") };
        request["messages"] = messages.to_json();
        request["stream"] = false;
        response = ollama::chat(request);

        CHECK(response.as_json().contains("message") == true);

        request = ollama::request(ollama::message_type::embedding);
        request["model"] = test_model;
        request["input"] = "Why is the sky blue?";
        request["stream"] = false;
        response = ollama::generate_embeddings(request);

        CHECK(response.as_json().contains("embeddings") == true);                
    }



    TEST_CASE("Enable Debug Logging") {
        
        ollama::show_requests(true);
        ollama::show_replies(true);

        CHECK(true);
    }

    /*
    TEST_CASE("Create and Check Blobs") {

        ollama::show_requests(true);

        // Create a blob on the ollama server using the following digest
        try { ollama::create_blob("sha256:29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2"); std::cout << "Blob was created on Ollama server." << std::endl; }
        catch( ollama::exception e) { std::cout << "Error when creating blob: " << e.what() << std::endl;} 

        // Check if a blob with the following digest exists.
        CHECK(ollama::blob_exists("sha256:29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2") == true);
    }    
*/
}