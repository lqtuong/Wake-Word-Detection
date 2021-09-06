import os, csv, pandas
source = '/home/tuong/PycharmProjects/CNNSound/'

def get_filepaths(directory):
    """
    Load data file paths
    :param directory:
    :return:
    """
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths

def get_label(directory):
    """
    Data label
    :param directory:
    :return:
    """
    labels = {}
    label = pandas.read_csv(directory, header=None)
    for i in range(len(label[0])):
        labels[str(label[0][i])] = label[1][i]
    return labels

def process_filename(path0, path1):

    list0 = get_filepaths(path0)
    list1 = get_filepaths(path1)
    label0 = get_label(source+"label0.csv")
    label1 = get_label(source+"label1.csv")
    lenght = len(list0) + len(list1)
    i = 0
    with open("label.csv","a", newline='') as f:
        writer = csv.writer(f)
        while i<lenght:
            if list0 is not None:
                name = list0.pop()
                os.rename(name, str(i)+".wav")
                writer.writerow([str(i)+".wav", label0[name.split("/")[-1]]])
                i +=1
            if list1 is not None:
                name = list1.pop()
                os.rename(name, str(i) + ".wav")
                writer.writerow([str(i) + ".wav", label1[name.split("/")[-1]]])
                i += 1
            if list0 is not None:
                name = list0.pop()
                os.rename(name, str(i) + ".wav")
                writer.writerow([str(i) + ".wav", label0[name.split("/")[-1]]])
                i +=1
            if list0 is not None:
                name = list0.pop()
                os.rename(name, str(i) + ".wav"),
                writer.writerow([str(i) + ".wav", label0[name.split("/")[-1]]])
                i += 1
            if list1 is not None:
                name = list1.pop()
                os.rename(name, str(i) + ".wav")
                writer.writerow([str(i) + ".wav", label1[name.split("/")[-1]]])
                i += 1

def add_label(path0, path1):
    files0 = get_filepaths(path0)
    files1 = get_filepaths(path1)
    with open("label_va.csv","a",newline='') as f:
        writer = csv.writer(f)
        i = 0
        while len(files0) != 0 :
            if i%2 != 0 or i%7 == 0:
                name = str(i)
                os.rename(files0.pop(), name+'.wav')
                writer.writerow([name, 0])
            i += 1
        i = 0
        while len(files1) != 0:
            if i%2 == 0 and i%7 != 0:
                name = str(i)
                os.rename(files1.pop(), name+'.wav')
                writer.writerow([name, 1])
            i += 2

if __name__ == "__main__":
    #process_filename(source+'ex0', source+'ex1')
    add_label('./va_0','./va_1')

    # files = get_filepaths('./va')
    # label =  get_label('./label_va.csv')
    # for file in files:
    #     if label[file.split('/')[-1].split('.')[0]] == 1:
    #         os.rename(file, file.split('/')[-1])
    # zero = 0
    # one = 0
    # for i in range(872):
    #     if i%2 != 0 or i%7== 0:
    #         print(0)
    #         zero +=1
    #     if i%2 ==0 and i%7 != 0:
    #         print(1)
    #         one +=1
    # print("zero ", zero)
    # print("one ", one)

