import csv

# Blog: http://blog.csdn.net/fengbingchun/article/details/78624358
txt_file = r"../../data/database/BacknoteDataset/data_banknote_authentication.txt"
csv_file = r"../../data/database/BacknoteDataset/data_banknote_authentication.csv"

in_txt = csv.reader(open(txt_file, "r"), delimiter = ',')
out_csv = csv.writer(open(csv_file, 'w', encoding = 'utf8'),  lineterminator = '\n')

out_csv.writerows(in_txt)