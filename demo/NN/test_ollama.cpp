#ifdef _MSC_VER

#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <sstream>
#include "ollama.hpp"

// Blog: https://blog.csdn.net/fengbingchun/article/details/151677357

namespace {

constexpr char model[]{ "qwen3:1.7b" };
std::ostringstream oss_response{};

std::string gbk_to_utf8(const std::string& str)
{
	// gbk to wchar
	auto len = ::MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, nullptr, 0);
	if (len <= 0) return {};
	std::wstring wstr(len, 0);
	::MultiByteToWideChar(CP_ACP, 0, str.c_str(), -1, &wstr[0], len);

	// wchar to utf8
	len = ::WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
	if (len <= 0) return {};
	std::string u8str(len, 0);
	::WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &u8str[0], len, nullptr, nullptr);

	u8str.pop_back(); // remove '\0'
	return u8str;
}

std::string utf8_to_gbk(const std::string& u8str)
{
	// utf8 to wchar
	auto len = ::MultiByteToWideChar(CP_UTF8, 0, u8str.c_str(), -1, nullptr, 0);
	if (len <= 0) return {};
	std::wstring wstr(len, 0);
	::MultiByteToWideChar(CP_UTF8, 0, u8str.c_str(), -1, &wstr[0], len);

	// wchar to gbk
	len = ::WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
	if (len <= 0) return {};
	std::string str(len, 0);
	::WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &str[0], len, nullptr, nullptr);

	str.pop_back(); // remove '\0' 
	return str;
}

bool on_receive_response(const ollama::response& response)
{
	std::cout << utf8_to_gbk(response) << std::flush; // print the token received
	oss_response << response;

	if (response.as_json()["done"] == true) std::cout << std::endl; // the server will set "done" to true for the last response
	
	return true; // return true to continue streaming this response; return false to stop immediately
}

} // namespace

int test_ollama_model_list()
{
	if (auto ret = ollama::is_running(); ret) {
		std::cout << "ollama version: " << ollama::get_version() << std::endl;
		
		std::cout << "model list:" << std::endl;
		for (const auto& str : ollama::list_models()) {
			std::cout << "\t" << str << std::endl;

			auto model_info = ollama::show_model_info(str, false);
			std::cout << "\t\tformat: " << model_info["details"]["format"] << std::endl;
			std::cout << "\t\tfamily: " << model_info["details"]["family"] << std::endl;
			std::cout << "\t\tparameter size: " << model_info["details"]["parameter_size"] << std::endl;
			std::cout << "\t\tquantization level: " << model_info["details"]["quantization_level"] << std::endl;
		}
	}
	else {
		std::cerr << "Error: ollama not running" << std::endl;
		return -1;
	}

	return 0;
}

int test_ollama_chat()
{
	if (auto ret = ollama::is_running(); ret) {
		try {
			ollama::messages messages{};

			while (true) {
				std::cout << "input: ";
				std::string input{};
				std::getline(std::cin, input);
				if (input == "q")
					break;

				ollama::message message_user("user", gbk_to_utf8(input));
				messages.emplace_back(message_user);
				auto response = ollama::chat(model, messages);
				std::cout << "AI: " << utf8_to_gbk(response.as_simple_string()) << std::endl;

				ollama::message message_assistant("assistant", response.as_simple_string());
				messages.emplace_back(message_assistant);
			}
		}
		catch (ollama::exception& e) {
			std::cerr << "Exception Error: " << e.what() << std::endl;
			return -1;
		}
	}
	else {
		std::cerr << "Error: ollama not running" << std::endl;
		return -1;
	}

	return 0;
}

int test_ollama_chat_stream()
{
	if (auto ret = ollama::is_running(); ret) {
		try {
			ollama::messages messages{};

			while (true) {
				std::cout << "input: ";
				std::string input{};
				std::getline(std::cin, input);
				if (input == "q")
					break;

				ollama::message message_user("user", gbk_to_utf8(input));
				messages.emplace_back(message_user);

				std::cout << "AI: ";
				std::function<bool(const ollama::response&)> response_callback = on_receive_response;
				ollama::chat(model, messages, response_callback);

				ollama::message message_assistant("assistant", oss_response.str());
				messages.emplace_back(message_assistant);
				oss_response.str("");
				oss_response.clear();
			}
		}
		catch (ollama::exception& e) {
			std::cerr << "Exception Error: " << e.what() << std::endl;
			return -1;
		}
	}
	else {
		std::cerr << "Error: ollama not running" << std::endl;
		return -1;
	}

	return 0;
}

int test_ollama_generate()
{
	if (auto ret = ollama::is_running(); ret) {
		try {
			std::cout << "please input prompt: ";
			std::string prompt{};
			std::getline(std::cin, prompt);

			auto response = ollama::generate(model, gbk_to_utf8(prompt));
			std::cout << "AI: " << utf8_to_gbk(response.as_simple_string()) << std::endl;
		}
		catch (ollama::exception& e) {
			std::cerr << "Exception Error: " << e.what() << std::endl;
			return -1;
		}
	}
	else {
		std::cerr << "Error: ollama not running" << std::endl;
		return -1;
	}

	return 0;
}

int test_ollama_generate_stream()
{
	if (auto ret = ollama::is_running(); ret) {
		try {
			std::cout << "please input prompt: ";
			std::string prompt{};
			std::getline(std::cin, prompt);

			std::cout << "AI: ";
			std::function<bool(const ollama::response&)> response_callback = on_receive_response;
			ollama::generate(model, gbk_to_utf8(prompt), response_callback);
		}
		catch (ollama::exception& e) {
			std::cerr << "Exception Error: " << e.what() << std::endl;
			return -1;
		}
	}
	else {
		std::cerr << "Error: ollama not running" << std::endl;
		return -1;
	}

	return 0;
}

#endif // _MSC_VER
