
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

label_names = unpickle("datasets/cifar100/meta")

f = open("label_data_names.txt", "w")
f.write("coarse_label_names\n")
for label in label_names[b'coarse_label_names']:
    f.write(label.decode("utf-8") + "\n")

f.write("\n")

f.write("fine_label_names\n")
for label in label_names[b'fine_label_names']:
    f.write(label.decode("utf-8") + "\n")

f.close()
