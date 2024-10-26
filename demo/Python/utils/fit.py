import numpy as np

def linear_fit(x, y):
	coef = np.polyfit(np.array(x), np.array(y), 1)
	a, b = coef
	print(f"a*x+b: a:{a:.4f}; b:{b:.4f}")

if __name__ == "__main__":
	x = (170,175,183,187)
	y = (26,27,29,30)
	linear_fit(x, y)
	print("====== execution completed ======")
