#include <iostream>
#include "funset.hpp"
#include "opencv.hpp"
#include "libsvm.hpp"

int main()
{
	int ret = test_batch_normalization();
	
	if (ret == 0) fprintf(stdout, "========== test success ==========\n");
	else fprintf(stderr, "########## test fail ##########\n");

	return 0;
}

