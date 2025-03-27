import numpy as np
from scipy.optimize import minimize, show_options
from functools import partial
import colorama
import argparse
from pathlib import Path
import csv
from itertools import combinations
import ast
from datetime import datetime
import time

def parse_args():
    parser = argparse.ArgumentParser(description="solve inequalities")
    parser.add_argument("--csv_file", required=True, type=str, help="source csv file")
    parser.add_argument("--size", default=(0,0), help="which groups to perform")

    args = parser.parse_args()
    return args

def objective(x, p1, p2, p3, p4, p5):
    return p1*x[0] + p2*x[1] + p3*x[2] + p4*x[3] + p5*x[4]

def constraint(x, a, b, c, d, e, value, inequality_type):
    if inequality_type == "lower":
        return a * x[0] + b * x[1] + c * x[2] + d * x[3] + e * x[4] - value
    elif inequality_type == "upper":
        return value - (a * x[0] + b * x[1] + c * x[2] + d * x[3] + e * x[4])
    elif inequality_type == "eq":
        return a * x[0] + b * x[1] + c * x[2] + d * x[3] + e * x[4] - value
    else:
        raise ValueError(colorama.Fore.RED + f"invalid inequality_type: {inequality_type}")

def parse_csv(csv_file):
    if not Path(csv_file).exists():
        raise FileNotFoundError(colorama.Fore.RED + f"file doesn't exist: {csv_file}")

    datas = []
    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            datas.append(list(map(float, row))) # str -> float

    if len(datas) != 5: # S, TFe, Al203, -200mu, price
        raise ValueError(colorama.Fore.RED + f"length must be 5: {len(datas)}")

    length = len(datas[0])
    for index in range(1, 5):
        if len(datas[index]) != length:
            raise ValueError(colorama.Fore.RED + f"length mismatch: {length}:{len(datas[index])}")

    return datas, datas[0][0], datas[1][0], datas[2][0], datas[3][0], datas[4][0] # datas, a1, a2, a3, a4, p1

def get_all_groups(datas):
    S = datas[0][1:]
    # print(f"S: {S}")
    TFe = datas[1][1:]
    Al203 = datas[2][1:]
    Mu = datas[3][1:]
    Price = datas[4][1:]

    lists = list(range(len(S)))
    # print(f"lists: {lists}")
    combinations_list = list(combinations(lists, 4))
    # print(f"combinations list: {combinations_list}; length: {len(combinations_list)}")

    groups = []
    for i in combinations_list:
        group = []
        group.append([S[i[0]], S[i[1]], S[i[2]], S[i[3]]])
        group.append([TFe[i[0]], TFe[i[1]], TFe[i[2]], TFe[i[3]]])
        group.append([Al203[i[0]], Al203[i[1]], Al203[i[2]], Al203[i[3]]])
        group.append([Mu[i[0]], Mu[i[1]], Mu[i[2]], Mu[i[3]]])
        group.append([Price[i[0]], Price[i[1]], Price[i[2]], Price[i[3]]])

        groups.append(group)

    return groups

def solve_inequalities(csv_file, size):
    # a1, b1, c1, d1, e1 = 0.033, 0.080, 0.22, 0.58, 0.21 # S
    # a2, b2, c2, d2, e2 = 46.07, 64.58, 64.81, 65.84, 65.10 # TFe
    # a3, b3, c3, d3, e3 = 14.15, 0.63, 0.53, 1.09, 0.72 # Al203
    # a4, b4, c4, d4, e4 = 0.0, 76.9, 86.1, 57.8, 88.5 # -200mu
    a5, b5, c5, d5, e5 = 1.0, 1.0, 1.0, 1.0, 1.0

    datas, a1, a2, a3, a4, p1 = parse_csv(csv_file)
    # print(f"a1:{a1}, a2:{a2}, a3:{a3}, a4:{a4}, p1:{p1}")
    groups = get_all_groups(datas)

    size = ast.literal_eval(size) # str -> tuple
    if len(size) != 2:
        raise ValueError(colorama.Fore.RED + f"length must be 2: {len(size)}")

    start = 0
    end = len(groups)
    if size[1] - size[0] != 0:
        start = size[0]
        end = size[1]

    bounds = [(7.0, 9.0), (1.0, 55.0), (1.0, 55.0), (1.0, 55.0), (1.0, 55.0)] # range: A,B,C,D,E
    iter_num = [2, 3, 3, 3, 3] # A,B,C,D,E
    range_S = (0.0, 49.0)
    range_TFe = (6200.0, 6230.0)
    range_Al203 = (9.9, 782.0)
    range_200mu = (6500.0, 7000.0)
    range_other = 99.0

    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # print(f"time str: {time_str}")
    result_name = time_str + "-result.txt"

    with open(result_name, "w", encoding="utf-8") as file:
        file.write(f"bounds = {bounds}\n")
        file.write(f"iter_num = {iter_num}\n")
        file.write(f"range_S = {range_S}\n")
        file.write(f"range_TFe = {range_TFe}\n")
        file.write(f"range_Al203 = {range_Al203}\n")
        file.write(f"range_200mu = {range_200mu}\n")
        file.write(f"range_other = {range_other}\n")
        total_iter_number = 1
        for num in iter_num:
            total_iter_number *= num
        print(f"total number of iterations per group: {total_iter_number}")
        file.write(f"total number of iterations per group: {total_iter_number}\n")

        print(colorama.Fore.YELLOW + f"groups length: {len(groups)}: start:{start+1}/{len(groups)}; end:{end}/{len(groups)}")
        file.write(f"groups length: {len(groups)}: start:{start+1}/{len(groups)}; end:{end}/{len(groups)}\n")

        samplesA = np.linspace(bounds[0][0], bounds[0][1], num=iter_num[0], endpoint=True, dtype=float)
        samplesB = np.linspace(bounds[1][0], bounds[1][1], num=iter_num[1], endpoint=True, dtype=float)
        samplesC = np.linspace(bounds[2][0], bounds[2][1], num=iter_num[2], endpoint=True, dtype=float)
        samplesD = np.linspace(bounds[3][0], bounds[3][1], num=iter_num[3], endpoint=True, dtype=float)
        samplesE = np.linspace(bounds[4][0], bounds[4][1], num=iter_num[4], endpoint=True, dtype=float)

        count = start + 1
        for group in groups[start:end]:
            start_time = time.time()
            print(colorama.Fore.GREEN + f"index: {count}/{len(groups)}: (S,TFe,Al203,-200mu,unit price) {group}")
            file.write(f"index: {count}/{len(groups)}: (S,TFe,Al203,-200mu,unit price) {group}\n")
            count += 1

            b1 = group[0][0]; c1 = group[0][1]; d1 = group[0][2]; e1 = group[0][3]
            b2 = group[1][0]; c2 = group[1][1]; d2 = group[1][2]; e2 = group[1][3]
            b3 = group[2][0]; c3 = group[2][1]; d3 = group[2][2]; e3 = group[2][3]
            b4 = group[3][0]; c4 = group[3][1]; d4 = group[3][2]; e4 = group[3][3]
            p2 = group[4][0]; p3 = group[4][1]; p4 = group[4][2]; p5 = group[4][3]
            # print(f"S:{a1},{b1},{c1},{d1},{e1}\nTFe:{a2},{b2},{c2},{d2},{e2}\nAl203:{a3},{b3},{c3},{d3},{e3}\n-200mu:{a4},{b4},{c4},{d4},{e4}\nprice:{p1},{p2},{p3},{p4},{p5}")

            cons = [
                {"type": "ineq", "fun": partial(constraint, a=a1, b=b1, c=c1, d=d1, e=e1, value=range_S[0], inequality_type="lower")},
                {"type": "ineq", "fun": partial(constraint, a=a1, b=b1, c=c1, d=d1, e=e1, value=range_S[1], inequality_type="upper")},
                {"type": "ineq", "fun": partial(constraint, a=a2, b=b2, c=c2, d=d2, e=e2, value=range_TFe[0], inequality_type="lower")},
                {"type": "ineq", "fun": partial(constraint, a=a2, b=b2, c=c2, d=d2, e=e2, value=range_TFe[1], inequality_type="upper")},
                {"type": "ineq", "fun": partial(constraint, a=a3, b=b3, c=c3, d=d3, e=e3, value=range_Al203[0], inequality_type="lower")},
                {"type": "ineq", "fun": partial(constraint, a=a3, b=b3, c=c3, d=d3, e=e3, value=range_Al203[1], inequality_type="upper")},
                {"type": "ineq", "fun": partial(constraint, a=a4, b=b4, c=c4, d=d4, e=e4, value=range_200mu[0], inequality_type="lower")},
                {"type": "ineq", "fun": partial(constraint, a=a4, b=b4, c=c4, d=d4, e=e4, value=range_200mu[1], inequality_type="upper")},
                {"type": "eq", "fun": partial(constraint, a=a5, b=b5, c=c5, d=d5, e=e5, value=range_other, inequality_type="eq")},
            ]

            results = set()
            results_count = 0

            for index, valueA in enumerate(samplesA):
                for valueB in samplesB:
                    for valueC in samplesC:
                        for valueD in samplesD:
                            for valueE in samplesE:
                                x0 = [valueA, valueB, valueC, valueD, valueE]

                                solution = minimize(objective, x0, args=(p1, p2, p3, p4, p5), method="SLSQP", bounds=bounds, constraints=cons, options={'maxiter': 100})
                                if solution.success:
                                    A, B, C, D, E = solution.x
                                    if range_S[0] < a1*A+b1*B+c1*C+d1*D+e1*E < range_S[1] and range_TFe[0] < a2*A+b2*B+c2*C+d2*D+e2*E < range_TFe[1] and range_Al203[0] < a3*A+b3*B+c3*C+d3*D+e3*E < range_Al203[1] and range_200mu[0] < a4*A+b4*B+c4*C+d4*D+e4*E < range_200mu[1] and abs(a5*A+b5*B+c5*C+d5*D+e5*E - range_other) < 0.1:
                                        result = (float(round(A, 4)), float(round(B, 4)), float(round(C, 4)), float(round(D, 4)), float(round(E, 4)))
                                        results.add(result)
                                        results_count = len(results)
                print(f"index: {index}, valueA:{valueA}; results_count: {results_count}")
                file.write(f"\tindex: {index}, valueA:{valueA}; results_count: {results_count}\n")

            if len(results) == 0:
                print(f"no solutions: {len(results)}")
                file.write(f"\tno solutions: {len(results)}\n")

                end_time = time.time()
                elapsed_time = end_time - start_time
                if (elapsed_time > 60.0):
                    print(f"elapsed time: {elapsed_time / 60.0:.4f} minutes")
                    file.write(f"\telapsed time: {elapsed_time / 60.0:.4f} minutes\n")
                else:
                    print(f"elapsed time: {elapsed_time:.4f} seconds")
                    file.write(f"\telapsed time: {elapsed_time:.4f} seconds\n")

                continue

            # price = [0.237, 0.863, 0.855, 0.917, 0.886]
            price = [p1, p2, p3, p4, p5]
            min_price = 100000
            idx = 0

            results = list(results)
            values = []
            for result in results:
                value = price[0]*result[0] + price[1]*result[1] + price[2]*result[2] + price[3]*result[3] + price[4]*result[4]
                values.append(value)

            sorted_index = sorted(enumerate(values), key=lambda x: x[1])
            sorted_indices = [index for index, value in sorted_index]

            for i in range(len(sorted_indices)):
                idx = sorted_indices[i]
                print(f"total price: {values[idx]}; unit price: {price}; value: {results[idx]}")
                file.write(f"\ttotal price: {values[idx]}; unit price: {price}; value: {results[idx]}\n")
                if i == 9:
                    break

            end_time = time.time()
            elapsed_time = end_time - start_time
            if (elapsed_time > 60.0):
                print(f"elapsed time: {elapsed_time / 60.0:.4f} minutes")
                file.write(f"\telapsed time: {elapsed_time / 60.0:.4f} minutes\n")
            else:
                print(f"elapsed time: {elapsed_time:.4f} seconds")
                file.write(f"\telapsed time: {elapsed_time:.4f} seconds\n")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    args = parse_args()

    # show_options('minimize', 'SLSQP'); raise
    solve_inequalities(args.csv_file, args.size)

    print(colorama.Fore.GREEN + "====== execution completed ======")
