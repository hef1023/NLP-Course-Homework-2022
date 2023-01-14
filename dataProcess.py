num_data = 5000
num_train = 4500
def read_data(file_path,datalines:'int>0'):
    datalist = []
    with open(file_path,'r',encoding='utf-8') as f:
        for i in range(datalines):
            datalist.append(f.readline().strip())
            # print(datalist[i]) #!
    return datalist

def write_data(file_path,data_list):
    with open(file_path,'w',encoding='utf-8') as f:
        for i in data_list:
            f.write(str(i)+'\n')
    return

datalist_zh = read_data("data/zh-en.zh",num_data)
datalist_en = read_data("data/zh-en.en",num_data)
data_list = []
for i in range(num_data):
    tmp = {}
    tmp["en"] = datalist_en[i]
    tmp["zh"] = datalist_zh[i]
    data_list.append(tmp)
train_data = data_list[:num_train]
val_data = data_list[num_train:]

write_data('data/train.txt',train_data)
write_data('data/val.txt',val_data)

