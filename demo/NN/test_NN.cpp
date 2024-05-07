#include <iostream>
#include "funset.hpp"
#include "opencv.hpp"
#include "libsvm.hpp"

int main()
{
	int ret = test_monocular_ranging_face_triangle_similarity();

	if (ret == 0) std::cout << "========== test success ==========\n";
	else std::cerr << "########## test fail ##########\n";

	return 0;
}

