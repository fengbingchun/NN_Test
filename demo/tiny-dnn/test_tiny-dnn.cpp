#include <iostream>
#include "funset.hpp"

int main()
{
	int ret = test_dnn_batch_normalization();

	if (ret == 0) fprintf(stdout, "====== test success ======\n");
	else fprintf(stderr, "###### test fail ######\n");

	return 0;
}
