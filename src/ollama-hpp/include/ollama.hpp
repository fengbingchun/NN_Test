#ifndef OLLAMA_HPP
#define OLLAMA_HPP

/*  MIT License

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
*/

/*  About this software:

    Ollama is a high-quality REST server and API providing an interface to run
    language models locally via llama.cpp.

    Ollama was made by Jeffrey Morgan (@jmorganca) and the Ollama team and is 
    available under the MIT License. To support this project or for more details
    go to https://github.com/ollama or https://ollama.com/. 

    This library is a header-only C++ integration of the Ollama API providing access
    to most API features while integrating them with std library classes or popular
    header-only libraries within the community. The following external libraries are 
    used:
*/

/* 
    httplib is a header-only C++ http/https library.
    This library was created by Yuji Hirose and is available under the MIT License.
    For more details visit: https://github.com/yhirose/cpp-httplib
*/
#include "httplib.h"

/* 
    nlohmnann JSON is a feature-rich header-only C++ JSON implementation.
    This library was created by Niels Lohmann and is available under the MIT License.
    For more details visit: https://github.com/nlohmann/json
*/
#include "json.hpp"

/* 
    Base64.h is a header-only C++ library for encoding and decoding Base64 values.
    This library was created by tomykaira and is available under the MIT License.
    For more details visit: 
    https://gist.github.com/tomykaira/f0fd86b6c73063283afe550bc5d77594
*/
#include "Base64.h"

#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <numeric>
#include <functional>
#include <exception>
#include <initializer_list>

// Namespace types and classes
namespace ollama
{
    using json = nlohmann::json;
    using base64 = macaron::Base64;    

    static bool use_exceptions = true;    // Change this to false to avoid throwing exceptions within the library.    
    static bool log_requests = false;      // Log raw requests to the Ollama server. Useful when debugging.       
    static bool log_replies = false;       // Log raw replies from the Ollama server. Useful when debugging.

    inline void allow_exceptions(bool enable) {use_exceptions = enable;}
    inline void show_requests(bool enable) {log_requests = enable;}
    inline void show_replies(bool enable) {log_replies = enable;}

    enum class message_type { generation, chat, embedding };

    class exception : public std::exception {
    private:
        std::string message;

    public:
        exception(const std::string& msg) : message(msg) {}
        const char* what() const noexcept override { return message.c_str(); }
    };

    class invalid_json_exception : public ollama::exception { public: using exception::exception; };

    class image {
        public:
            image(const std::string base64_sequence, bool valid = true) 
            {
                this->base64_sequence = base64_sequence; this->valid = valid;
            }
            ~image(){};

            static image from_file(const std::string& filepath)
            {
                bool valid = true;
                std::ifstream file(filepath, std::ios::binary);
                if (!file) {
                    if (ollama::use_exceptions) throw ollama::exception("Unable to open image file from path.");
                    valid = false; return image("", valid);
                }

                std::string file_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

                return image(macaron::Base64::Encode(file_contents), valid);
            }

            static image from_base64_string(const std::string& base64_string)
            {
                return image(base64_string);
            }

            const std::string as_base64_string() const
            {
                return base64_sequence;
            }

            bool is_valid(){return valid;}

            operator std::string() const { return base64_sequence; }

            operator std::vector<ollama::image>() const { std::vector<ollama::image> images; images.push_back(*this); return images; }
            operator std::vector<std::string>() const { std::vector<std::string> images; images.push_back(*this); return images; }

        private:
            std::string base64_sequence;
            bool valid;
    };

    class images: public std::vector<std::string> {

        public:
            images(): std::vector<std::string>(0)
            {

            }
            images(const std::initializer_list<ollama::image>& list) {
                for (ollama::image value : list) {
                    this->push_back(value);
                }
            }        
            ~images(){};
            std::vector<std::string> to_strings()
            {
                std::vector<std::string> strings;
                for (auto it = this->begin(); it != this->end(); ++it)
                    strings.push_back(*it);
                
                return strings;
            }

    };

    class options: public json {
        
        public:
            options(): json() { this->emplace( "options", nlohmann::json::object() );  }        

        nlohmann::json& operator[](const std::string& key) 
        {
            if ( !this->at("options").contains(key) ) this->at("options").emplace( key, nlohmann::json::object() );                       
            return this->at("options").at(key); 
        }
        nlohmann::json& operator[](const char* key) { return this->operator[](std::string(key)); };

        const nlohmann::json& operator[](const std::string& key) const { return this->at("options").at(key); }
        const nlohmann::json& operator[](const char* key) const { return this->operator[](std::string(key)); };

    };

    class message: public json {
        public:
            message(const std::string& role, const std::string& content, const std::vector<ollama::image>& images): json() { (*this)["role"] = role; (*this)["content"] = content; (*this)["images"] = images; }
            message(const std::string& role, const std::string& content): json() { (*this)["role"] = role; (*this)["content"] = content; }           
            message() : json() {}
            ~message() {}

            std::string as_json_string() const { return this->dump(); }
            
            operator std::string() const { return this->as_json_string(); }           

    };

    class messages: public std::vector<message> {

        public:
            messages(): std::vector<message>(0) {}            
            messages(ollama::message message): messages() { this->push_back(message); }

            messages(const std::initializer_list<message>& list) {
                for (message value : list) {
                    this->push_back(value);
                }
            }        
            ~messages(){};
            const std::vector<std::string> to_strings() const
            {
                std::vector<std::string> strings;
                for (auto it = this->begin(); it != this->end(); ++it)
                    strings.push_back(*it);
                
                return strings;
            }        
        
            const std::vector<json> to_json() const 
            { 
                std::vector<json> output;  
                for (auto it = this->begin(); it != this->end(); ++it)
                    output.push_back(*it);
                return output;
            }

            operator std::vector<json>() const {return std::vector<json>();}
            operator std::vector<std::string>() const { return this->to_strings(); }    
        
    };

    class request: public json {

        public:

            // Create a request for a generation.
            request(const std::string& model,const std::string& prompt, const json& options=nullptr, bool stream=false, const std::vector<std::string>& images=std::vector<std::string>()): request()
            {   
                (*this)["model"] = model;
                (*this)["prompt"] = prompt;
                (*this)["stream"] = stream;

                if (options!=nullptr) (*this)["options"] = options["options"];
                if (!images.empty()) (*this)["images"] = images;

                type = message_type::generation;
            }

            // Create a request for a chat completion.
            request(const std::string& model, const ollama::messages& messages, const json& options=nullptr, bool stream=false, const std::string& format="json", const std::string& keep_alive_duration="5m"): request()
            {
                (*this)["model"] = model;
                (*this)["messages"] = messages.to_json();
                (*this)["stream"] = stream;

                if (options!=nullptr) (*this)["options"] = options["options"];
                (void)format; //(*this)["format"] = format; // Commented out as providing the format causes issues with some models.
                (*this)["keep_alive"] = keep_alive_duration;
                type = message_type::chat;

            }
            // Request for a chat completion with a single message
            request(const std::string& model, const ollama::message& message, const json& options=nullptr, bool stream=false, const std::string& format="json", const std::string& keep_alive_duration="5m") :request(model, messages(message), options, stream, format, keep_alive_duration ){}
           
            request(message_type type): request() { this->type = type; }

            request(): json() {}
            ~request(){};

            static ollama::request from_embedding(const std::string& model, const std::string& input, const json& options=nullptr, bool truncate=true, const std::string& keep_alive_duration="5m")
            {
                ollama::request request(message_type::embedding);

                request["model"] = model;
                request["input"] = input;
                if (options!=nullptr) request["options"] = options["options"];
                request["truncate"] = truncate;
                request["keep_alive"] = keep_alive_duration;
                
                return request;
            }

            const message_type& get_type() const { return type; }

        private:

        message_type type;
    };

    class response {

        public:

            response(const std::string& json_string, message_type type=message_type::generation): type(type)
            {
                this->json_string = json_string;
                try 
                {
                    json_data = json::parse(json_string); 
                    
                    if (type==message_type::generation && json_data.contains("response")) simple_string=json_data["response"].get<std::string>(); 
                    else
                    if (type==message_type::embedding && json_data.contains("embeddings")) simple_string=json_data["embeddings"].get<std::string>();
                    else
                    if (type==message_type::chat && json_data.contains("message")) simple_string=json_data["message"]["content"].get<std::string>();
                                         
                    if ( json_data.contains("error") ) error_string =json_data["error"].get<std::string>();
                }
                catch(...) { if (ollama::use_exceptions) throw ollama::invalid_json_exception("Unable to parse JSON string:"+this->json_string); valid = false; }
            }
            
            response() {json_string = ""; valid = false;}
            ~response(){};

            bool is_valid() const {return valid;};

            const std::string& as_json_string() const
            {
                return json_string;
            }

            const json& as_json() const
            {
                return json_data;                
            }

            const std::string& as_simple_string() const
            {
                return simple_string;               
            }

            bool has_error() const
            {
                if ( json_data.contains("error") ) return true;
                return false;                
            }

            const std::string& get_error() const
            {
                return error_string;
            }

            friend std::ostream& operator<<(std::ostream& os, const ollama::response& response) { os << response.as_simple_string(); return os; }

            const message_type& get_type() const
            {
                return type;
            }

            //operator std::string() const { return this->as_simple_string(); }
            operator std::string() const { return this->as_simple_string(); }
            //const operator std::string() const { return this->as_simple_string(); }           


        private:

        std::string json_string;
        std::string simple_string;
        std::string error_string;

        json json_data;        
        message_type type;
        bool valid;        
    };

}

class Ollama
{
    using json = nlohmann::json;

    public:

        Ollama(const std::string& url)
        {
            this->server_url = url;
            this->cli = new httplib::Client(url);
            this->setReadTimeout(120);
        }

        Ollama(): Ollama("http://localhost:11434") {}
        ~Ollama() { delete this->cli; }

    ollama::response generate(const std::string& model,const std::string& prompt, const ollama::response& context, const json& options=nullptr, const std::vector<std::string>& images=std::vector<std::string>())
    {
        ollama::request request(model, prompt, options, false, images);
        if ( context.as_json().contains("context") ) request["context"] = context.as_json()["context"];
        return generate(request);
    }

    ollama::response generate(const std::string& model,const std::string& prompt, const json& options=nullptr, const std::vector<std::string>& images=std::vector<std::string>())
    {
        ollama::request request(model, prompt, options, false, images);
        return generate(request);
    }

    // Generate a non-streaming reply as a string.
    ollama::response generate(ollama::request& request)
    {
        ollama::response response;

        request["stream"] = false;
        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;      

        if (auto res = this->cli->Post("/api/generate",request_string, "application/json"))
        {
            if (ollama::log_replies) std::cout << res->body << std::endl;

            response = ollama::response(res->body);
            if ( response.has_error() ) { if (ollama::use_exceptions) throw ollama::exception("Ollama response returned error: "+response.get_error() ); }
           
        }
        else
        {
            if (ollama::use_exceptions) throw ollama::exception("No response returned from server "+this->server_url+". Error was: "+httplib::to_string( res.error() ));
        }

        return response;        
    }

    bool generate(const std::string& model,const std::string& prompt, ollama::response& context, std::function<bool(const ollama::response&)> on_receive_token, const json& options=nullptr, const std::vector<std::string>& images=std::vector<std::string>())
    {
        ollama::request request(model, prompt, options, true, images);
        if ( context.as_json().contains("context") ) request["context"] = context.as_json()["context"];
        return generate(request, on_receive_token);
    }

    bool generate(const std::string& model,const std::string& prompt, std::function<bool(const ollama::response&)> on_receive_token, const json& options=nullptr, const std::vector<std::string>& images=std::vector<std::string>())
    {
        ollama::request request(model, prompt, options, true, images);
        return generate(request, on_receive_token);
    }


    // Generate a streaming reply where a user-defined callback function is invoked when each token is received.
    bool generate(ollama::request& request, std::function<bool(const ollama::response&)> on_receive_token)
    {
        request["stream"] = true;

        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;

        std::shared_ptr<std::vector<std::string>> partial_responses = std::make_shared<std::vector<std::string>>();

        auto stream_callback = [on_receive_token, partial_responses](const char *data, size_t data_length)->bool{
            
            std::string message(data, data_length);
            bool continue_stream = true;

            if (ollama::log_replies) std::cout << message << std::endl;
            try 
            {   
                partial_responses->push_back(message);
                std::string total_response = std::accumulate(partial_responses->begin(), partial_responses->end(), std::string(""));                
                ollama::response response(total_response);
                partial_responses->clear();  
                continue_stream = on_receive_token(response); 
            }
            catch (const ollama::invalid_json_exception& e) { /* Partial response was received. Will do nothing and attempt to concatenate with the next response. */ }
            
            return continue_stream;
        };

        if (auto res = this->cli->Post("/api/generate", request_string, "application/json", stream_callback)) { return true; }
        else if (res.error()==httplib::Error::Canceled) { /* Request cancelled by user. */ return true; }        
        else { if (ollama::use_exceptions) throw ollama::exception( "No response from server returned at URL "+this->server_url+" Error: "+httplib::to_string( res.error() ) ); } 

        return false;
    }

    ollama::response chat(const std::string& model, const ollama::messages& messages, json options=nullptr, const std::string& format="json", const std::string& keep_alive_duration="5m")
    {
        ollama::request request(model, messages, options, false, format, keep_alive_duration);
        return chat(request);
    }


    // Generate a non-streaming reply as a string.
    ollama::response chat(ollama::request& request)
    {
        ollama::response response;

        request["stream"] = false;        
        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;      

        if (auto res = this->cli->Post("/api/chat",request_string, "application/json"))
        {
            if (ollama::log_replies) std::cout << res->body << std::endl;

            response = ollama::response(res->body, ollama::message_type::chat);
            if ( response.has_error() ) { if (ollama::use_exceptions) throw ollama::exception("Ollama response returned error: "+response.get_error() ); }
           
        }
        else
        {
            if (ollama::use_exceptions) throw ollama::exception("No response returned from server "+this->server_url+". Error was: "+httplib::to_string( res.error() ));
        }

        return response;
    }

    bool chat(const std::string& model, const ollama::messages& messages, std::function<bool(const ollama::response&)> on_receive_token, const json& options=nullptr, const std::string& format="json", const std::string& keep_alive_duration="5m")
    {
        ollama::request request(model, messages, options, true, format, keep_alive_duration);
        return chat(request, on_receive_token);
    }


    bool chat(ollama::request& request, std::function<bool(const ollama::response&)> on_receive_token)
    {
        ollama::response response;        
        request["stream"] = true;

        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;      

        std::shared_ptr<std::vector<std::string>> partial_responses = std::make_shared<std::vector<std::string>>();

        auto stream_callback = [on_receive_token, partial_responses](const char *data, size_t data_length)->bool{
            
            std::string message(data, data_length);
            bool continue_stream = true;

            if (ollama::log_replies) std::cout << message << std::endl;
            try 
            {   
                partial_responses->push_back(message);
                std::string total_response = std::accumulate(partial_responses->begin(), partial_responses->end(), std::string(""));                
                ollama::response response(total_response, ollama::message_type::chat);
                partial_responses->clear();  

                if ( response.has_error() ) { if (ollama::use_exceptions) throw ollama::exception("Ollama response returned error: "+response.get_error() ); }
                continue_stream = on_receive_token(response);
            }
            catch (const ollama::invalid_json_exception& e) { /* Partial response was received. Will do nothing and attempt to concatenate with the next response. */ }

            return continue_stream;
        };

        if (auto res = this->cli->Post("/api/chat", request_string, "application/json", stream_callback)) { return true; }
        else if (res.error()==httplib::Error::Canceled) { /* Request cancelled by user. */ return true; }
        else { if (ollama::use_exceptions) throw ollama::exception( "No response from server returned at URL"+this->server_url+" Error: "+httplib::to_string( res.error() ) ); }

        return false;
    }

    bool create_model(const std::string& modelName, const std::string& modelFile, bool loadFromFile=true)
    {

        // Generate the JSON request
        json request;
        request["name"] = modelName;

        if (loadFromFile)
        {
            // Open the file
            std::ifstream file(modelFile, std::ios::binary);

            // Check if the file is open
            if (!file.is_open()) {
                if (ollama::use_exceptions) throw ollama::exception("Failed to open file "+modelFile);
                return false;
            }

            // Read the entire file into a string using iterators
            std::string file_contents((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());
            
            request["modelFile"] = file_contents;                                                
        }
        else request["modelFile"] = modelFile;

        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;  

        std::string response;

        if (auto res = this->cli->Post("/api/create",request_string, "application/json"))
        {
            if (ollama::log_replies) std::cout << res->body << std::endl;

            json chunk = json::parse(res->body);
            if (chunk["status"]=="success") return true;        
        }
        else
        {
            if (ollama::use_exceptions) throw ollama::exception("No response returned: "+httplib::to_string( res.error() ) );
        }

        return false;

    }

    bool load_model(const std::string& model)
    {
        json request;
        request["model"] = model;
        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;

        // Send a blank request with the model name to instruct ollama to load the model into memory.
        if (auto res = this->cli->Post("/api/generate", request_string, "application/json"))
        {
            if (ollama::log_replies) std::cout << res->body << std::endl;
            json response = json::parse(res->body);
            return response["done"];        
        }
        else
        { 
            if (ollama::use_exceptions) throw ollama::exception("No response returned from server when loading model: "+httplib::to_string( res.error() ) ); 
        }

        // If we didn't get a response from the server indicating the model was created, return false.        
        return false;                
    }

    bool is_running()
    {
        auto res = cli->Get("/");
        if (res) if (res->body=="Ollama is running") return true;
        return false;
    }

    json list_model_json()
    {
        json models;
        if (auto res = cli->Get("/api/tags"))
        {
            if (ollama::log_replies) std::cout << res->body << std::endl;
            models = json::parse(res->body);
        }
        else { if (ollama::use_exceptions) throw ollama::exception("No response returned from server when querying model list: "+httplib::to_string( res.error() ) );}        

        return models;
    }

    std::vector<std::string> list_models()
    {
        std::vector<std::string> models;

        json json_response = list_model_json();
        
        for (auto& model: json_response["models"])
        {
            models.push_back(model["name"]);
        }

        return models;
    }

    json running_model_json()
    {
        json models;
        if (auto res = cli->Get("/api/ps"))
        {
            if (ollama::log_replies) std::cout << res->body << std::endl;
            models = json::parse(res->body);
        }
        else { if (ollama::use_exceptions) throw ollama::exception("No response returned from server when querying running models: "+httplib::to_string( res.error() ) );}        

        return models;
    }

    std::vector<std::string> list_running_models()
    {
        std::vector<std::string> models;

        json json_response = running_model_json();
        
        for (auto& model: json_response["models"])
        {
            models.push_back(model["name"]);
        }

        return models;
    }

    bool blob_exists(const std::string& digest)
    {
        if (auto res = cli->Head("/api/blobs/"+digest))
        {
            if (res->status==httplib::StatusCode::OK_200) return true;
            if (res->status==httplib::StatusCode::NotFound_404) return false;            
        }
        else { if (ollama::use_exceptions) throw ollama::exception("No response returned from server when checking if blob exists: "+httplib::to_string( res.error() ) );}        

        return false;
    }

    bool create_blob(const std::string& digest)
    {
        if (auto res = cli->Post("/api/blobs/"+digest))
        {
            if (res->status==httplib::StatusCode::Created_201) return true;
            if (res->status==httplib::StatusCode::BadRequest_400) { if (ollama::use_exceptions) throw ollama::exception("Received bad request (Code 400) from Ollama server when creating blob."); }            
        }
        else { if (ollama::use_exceptions) throw ollama::exception("No response returned from server when creating blob: "+httplib::to_string( res.error() ) );}        

        return false;

    }

    json show_model_info(const std::string& model, bool verbose=false)
    {
        json request, response;
        request["name"] = model;
        if (verbose) request["verbose"] = true;

        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;

        if (auto res = cli->Post("/api/show", request_string, "application/json"))
        {
            if (ollama::log_replies) std::cout << "Reply was " << res->body << std::endl;
            try
            { 
                response = json::parse(res->body); 
            }
            catch(...)
            { if (ollama::use_exceptions) throw ollama::exception("Received bad response from Ollama server when querying model info."); }           
        }
        else { if (ollama::use_exceptions) throw ollama::exception("No response returned from server when querying model info: "+httplib::to_string( res.error() ) );}        

        return response;
    }

    bool copy_model(const std::string& source_model, const std::string& dest_model)
    {
        json request;
        request["source"] = source_model;
        request["destination"] = dest_model;

        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;
        
        if (auto res = cli->Post("/api/copy", request_string, "application/json"))
        {
            if (res->status==httplib::StatusCode::OK_200) return true;
            if (res->status==httplib::StatusCode::NotFound_404) { if (ollama::use_exceptions) throw ollama::exception("Source model not found when copying model (Code 404)."); }            
        }
        else { if (ollama::use_exceptions) throw ollama::exception("No response returned from server when copying model: "+httplib::to_string( res.error() ) );}        

        return false;
    }

    bool delete_model(const std::string& model)
    {
        json request;
        request["name"] = model;

        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;
        
        if (auto res = cli->Delete("/api/delete", request_string, "application/json"))
        {
            if (res->status==httplib::StatusCode::OK_200) return true;
            if (res->status==httplib::StatusCode::NotFound_404) { if (ollama::use_exceptions) throw ollama::exception("Model not found when trying to delete (Code 404)."); }            
        }
        else { if (ollama::use_exceptions) throw ollama::exception("No response returned from server when deleting model: "+httplib::to_string( res.error() ) );}        

        return false;
    }

    bool pull_model(const std::string& model, bool allow_insecure = false)
    {
        json request, response;
        request["name"] = model;
        request["insecure"] = allow_insecure;
        request["stream"] = false;

        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;
        
        if (auto res = cli->Post("/api/pull", request_string, "application/json"))
        {
            if (res->status==httplib::StatusCode::OK_200) return true;
            if (res->status==httplib::StatusCode::NotFound_404) { if (ollama::use_exceptions) throw ollama::exception("Model not found when trying to pull (Code 404)."); return false; }

            response = json::parse(res->body);
            if ( response.contains("error") ) { if (ollama::use_exceptions) throw ollama::exception( "Error returned from ollama when pulling model: "+response["error"].get<std::string>() ); return false; }          
        }
        else { if (ollama::use_exceptions) throw ollama::exception("No response returned from server when pulling model: "+httplib::to_string( res.error() ) );}        

        return false;
    }

    bool push_model(const std::string& model, bool allow_insecure = false)
    {
        json request, response;
        request["name"] = model;
        request["insecure"] = allow_insecure;
        request["stream"] = false;

        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;
        
        if (auto res = cli->Post("/api/push", request_string, "application/json"))
        {
            if (res->status==httplib::StatusCode::OK_200) return true;
            if (res->status==httplib::StatusCode::NotFound_404) { if (ollama::use_exceptions) throw ollama::exception("Model not found when trying to push (Code 404)."); return false; }

            response = json::parse(res->body);
            if ( response.contains("error") ) { if (ollama::use_exceptions) throw ollama::exception( "Error returned from ollama when pushing model: "+response["error"].get<std::string>() ); return false; }          
        }
        else { if (ollama::use_exceptions) throw ollama::exception("No response returned from server when pushing model: "+httplib::to_string( res.error() ) );}        

        return false;
    }

    ollama::response generate_embeddings(const std::string& model, const std::string& input, const json& options=nullptr, bool truncate = true, const std::string& keep_alive_duration="5m")
    {
        ollama::request request = ollama::request::from_embedding(model, input, options, truncate, keep_alive_duration);
        return generate_embeddings(request);
    }


    ollama::response generate_embeddings(ollama::request& request)
    {
        ollama::response response;

        std::string request_string = request.dump();
        if (ollama::log_requests) std::cout << request_string << std::endl;
        
        if (auto res = cli->Post("/api/embed", request_string, "application/json"))
        {
            if (ollama::log_replies) std::cout << res->body << std::endl;


            if (res->status==httplib::StatusCode::OK_200) {response = ollama::response(res->body); return response; };
            if (res->status==httplib::StatusCode::NotFound_404) { if (ollama::use_exceptions) throw ollama::exception("Model not found when trying to push (Code 404)."); }

            if ( response.has_error() ) { if (ollama::use_exceptions) throw ollama::exception( "Error returned from ollama when generating embeddings: "+response.get_error() ); }          
        }
        else { if (ollama::use_exceptions) throw ollama::exception("No response returned from server when pushing model: "+httplib::to_string( res.error() ) );}        

        return response;
    }

    std::string get_version()
    {
        std::string version;

        auto res = this->cli->Get("/api/version");

        if (res)
        {
            json response = json::parse(res->body);        
            version = response["version"];
        }
        else
        {
            throw ollama::exception("Error retrieving version: "+res->status);
        }

        return version;

    }

    void setServerURL(const std::string& server_url)
    {
        this->server_url = server_url;
        delete(this->cli);        
        this->cli = new httplib::Client(server_url);
    }

    void setReadTimeout(const int seconds)
    {
        this->cli->set_read_timeout(seconds);
    }

    void setWriteTimeout(const int seconds)
    {
        this->cli->set_write_timeout(seconds);
    }

    private:

/*
    bool send_request(const ollama::request& request, std::function<bool(const ollama::response&)> on_receive_response=nullptr)
    {

        return true;
    }
*/

    std::string server_url;
    httplib::Client *cli;

};

// Functions associated with Ollama singleton
namespace ollama
{    
    // Use directly from the namespace as a singleton
    static Ollama ollama;
    
    inline void setServerURL(const std::string& server_url)
    {
        ollama.setServerURL(server_url);
    }

    inline ollama::response generate(const std::string& model, const std::string& prompt, const json& options=nullptr, const std::vector<std::string>& images=std::vector<std::string>())
    {
        return ollama.generate(model, prompt, options, images);
    }

    inline ollama::response generate(const std::string& model,const std::string& prompt, const ollama::response& context, const json& options=nullptr, const std::vector<std::string>& images=std::vector<std::string>())
    {
        return ollama.generate(model, prompt, context, options, images);
    }

    inline ollama::response generate(ollama::request& request)
    {
        return ollama.generate(request);
    }

    inline bool generate(const std::string& model,const std::string& prompt, std::function<bool(const ollama::response&)> on_receive_response, const json& options=nullptr, const std::vector<std::string>& images=std::vector<std::string>())
    {
        return ollama.generate(model, prompt, on_receive_response, options, images);
    }

    inline bool generate(const std::string& model,const std::string& prompt, ollama::response& context, std::function<bool(const ollama::response&)> on_receive_response, const json& options=nullptr, const std::vector<std::string>& images=std::vector<std::string>())
    {
        return ollama.generate(model, prompt, context, on_receive_response, options, images);
    }

    inline bool generate(ollama::request& request, std::function<bool(const ollama::response&)> on_receive_response)
    {
        return ollama.generate(request, on_receive_response);
    }

    inline ollama::response chat(const std::string& model, const ollama::messages& messages, const json& options=nullptr, const std::string& format="json", const std::string& keep_alive_duration="5m")
    {
        return ollama.chat(model, messages, options, format, keep_alive_duration);
    }

    inline ollama::response chat(ollama::request& request)
    {
        return ollama.chat(request);
    }

    inline bool chat(const std::string& model, const ollama::messages& messages, std::function<bool(const ollama::response&)> on_receive_response, const json& options=nullptr, const std::string& format="json", const std::string& keep_alive_duration="5m")
    {
        return ollama.chat(model, messages, on_receive_response, options, format, keep_alive_duration);
    }

    inline bool chat(ollama::request& request, std::function<bool(const ollama::response&)> on_receive_response)
    {
        return ollama.chat(request, on_receive_response);
    }

    inline bool create(const std::string& modelName, const std::string& modelFile, bool loadFromFile=true)
    {
        return ollama.create_model(modelName, modelFile, loadFromFile);
    }

    inline bool is_running()
    {
        return ollama.is_running();
    }

    inline bool load_model(const std::string& model)
    {
        return ollama.load_model(model);
    }

    inline std::string get_version()
    {
        return ollama.get_version();
    }

    inline std::vector<std::string> list_models()
    {
        return ollama.list_models();
    }

    inline json list_model_json()
    {
        return ollama.list_model_json();
    }

    inline std::vector<std::string> list_running_models()
    {
        return ollama.list_running_models();
    }

    inline json running_model_json()
    {
        return ollama.running_model_json();
    }

    inline bool blob_exists(const std::string& digest)
    {
        return ollama.blob_exists(digest);
    }

    inline bool create_blob(const std::string& digest)
    {
        return ollama.create_blob(digest);
    }

    inline json show_model_info(const std::string& model, bool verbose=false)
    {
        return ollama.show_model_info(model, verbose);
    }

    inline bool copy_model(const std::string& source_model, const std::string& dest_model)
    {
        return ollama.copy_model(source_model, dest_model);
    }

    inline bool delete_model(const std::string& model)
    {
        return ollama.delete_model(model);
    }

    inline bool pull_model(const std::string& model, bool allow_insecure = false)
    {
        return ollama.pull_model(model, allow_insecure);
    }

    inline bool push_model(const std::string& model, bool allow_insecure = false)
    {
        return ollama.push_model(model, allow_insecure);
    }

    inline ollama::response generate_embeddings(const std::string& model, const std::string& input, const json& options=nullptr, bool truncate = true, const std::string& keep_alive_duration="5m")
    {
        return ollama.generate_embeddings(model, input, options, truncate, keep_alive_duration);
    }

    inline ollama::response generate_embeddings(ollama::request& request)
    {
        return ollama.generate_embeddings(request);
    }

    inline void setReadTimeout(const int& seconds)
    {
        ollama.setReadTimeout(seconds);
    }

    inline void setWriteTimeout(const int& seconds)
    {
        ollama.setWriteTimeout(seconds);
    }

}


#endif