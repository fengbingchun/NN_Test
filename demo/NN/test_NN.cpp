#include <iostream>
#include "funset.hpp"
#include "opencv.hpp"
#include "libsvm.hpp"
#include "yolo.hpp"

int main()
{
	if (auto ret = test_ollama_chat_stream(); ret == 0)
		std::cout << "========== test success ==========\n";
	else
		std::cerr << "########## test fail ##########\n";

	return 0;
}
