import numpy as np
from pyquaternion import Quaternion

# Blog: https://blog.csdn.net/fengbingchun/article/details/130790531

def calculate_quaternion(R):
	q = Quaternion(matrix=np.array(R))

	return q

def calculate_rotate_matrix(q):
	R = q.rotation_matrix

	return R

def qvec2rotmat(qvec):
	# from instant-ngp/scripts/colmap2nerf.py
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

if __name__ == "__main__":
	R = [[2.044289726145588e-004, -0.2587487517264626, -0.9659446369688031],
		[-0.9993063898830017, -3.602307923217642e-002, 9.438056030485108e-003],
		[-3.723838540803551e-002, 0.9652727185840433, -0.2585766451355823]]
	q = calculate_quaternion(R)
	print(f"quaternion:\n{q}")

	R2 = calculate_rotate_matrix(q)
	print(f"R2:\n{R2}")

	R3 = qvec2rotmat([q[0], q[1], q[2], q[3]])
	print(f"R3:\n{R3}")

	print("test finish")
