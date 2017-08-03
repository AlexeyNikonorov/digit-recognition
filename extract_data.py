from sklearn import datasets

def to_string(x, l):
	s = ''
	s += str(int(l))
	for i in x: s += ' ' + str(int(i))
	return s

digits = datasets.load_digits()
n = len(digits.images)

data = digits.images.reshape((n, -1))
target = digits.target

with open('data.txt', 'w') as file:
	for x, l in zip(data, target):
		file.write(to_string(x, l) + '\n')
