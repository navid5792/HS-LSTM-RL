
with open("dataset/data/MS_train.txt") as f:
	data = f.readlines()

data = data[1:]
l1 = []
l2 = []
for i in range(len(data)):
	data[i] = data[i].strip().split("\t")

	l1.append(len(data[i][1].split()))
	l2.append(len(data[i][2].split()))
print(sorted(l1))
print(sorted(l2))

'''
length = 150

f1 = open("dataset/data/AI_train.txt", "r")
f2 = open("dataset/data/AI_actions.txt", "r")
	

data = f1.readlines()
action = f2.readlines()

f1.close()
f2.close()

f1 = open("dataset/data/AI_train_.txt", "w")
f2 = open("dataset/data/AI_actions_.txt", "w")
f1.write(data[0])
f2.write(action[0])

for i in range(1, len(data)):
	data[i] = data[i].strip().split("\t")
	action[i] = action[i].strip().split("\t")
	left = data[i][1].split()
	right = data[i][2].split()
	left_action = action[i][0].split()
	right_action = action[i][1].split()
	if len(left) >= length:
		left = " ".join(left[0:length])		
		left_action = left_action[0:length]
		left_action[-1] = '1'
		left_action = " ".join(left_action)
	else:
		left = " ".join(left)
		left_action = " ".join(left_action)

	if len(right) >= length:	
		right = " ".join(right[0:length])
		right_action = right_action[0:length]
		right_action[-1] = '1'
		right_action = " ".join(right_action)
	else:
		right = " ".join(right)
		right_action = " ".join(right_action)

	f1.write(data[i][0] + "\t" + left + "\t" + right + "\n")
	f2.write(left_action + "\t" + right_action + "\n")

f1.close()
f2.close()
'''