from copy import deepcopy
from collections import Counter

folder  = "sicke"
with open(folder + "/count.txt", "r") as f:
	count = f.readlines() 

with open(folder + "/deleteCount.txt", "r") as f:
	deleteCount = f.readlines()
    
word2count = dict()
word2deleteCount = dict()
for i in range(len(count)):
    x = count[i].strip().split(":")
    if x[0] not in word2count.keys():
        word2count[x[0]] = int(x[1])
        
for i in range(len(deleteCount)):
    x = deleteCount[i].strip().split(":")
    if x[0] not in word2deleteCount.keys():
        word2deleteCount[x[0]] = int(x[1])

del count
del deleteCount

word2freq = dict()
for x in word2deleteCount.keys():
    if x not in word2freq.keys():
        word2freq[x] = (word2deleteCount[x]/word2count[x]) * 100

with open(folder + "/frequency sort.txt", "w") as f:       
    for key, value in sorted(word2freq.items(), key=lambda item: item[1], reverse = True):
        f.write(str(key) + "\t" + str(value) + "\ttotal = " + str(word2count[key]) + "\tdeleted = " + str(word2deleteCount[key]) + "\n")
        

with open(folder + "/count sort.txt", "w") as f:       
    for key, value in sorted(word2deleteCount.items(), key=lambda item: item[1], reverse = True):
        f.write(str(key) + "\t" + str(word2freq[key]) + "\ttotal = " + str(word2count[key]) + "\tdeleted = " + str(value) + "\n")

demo = []        
for x in word2freq.keys():
    demo.append([x, word2freq[x], word2count[x]])
    
for i in range(len(demo)):
    for j in range(i, len(demo)):
        if demo[i][1] < demo[j][1]:
            temp = demo[i]
            demo[i] = demo[j]
            demo[j] = temp
        elif demo[i][1] == demo[j][1]:
            if demo[i][2] < demo[j][2]:
                temp = demo[i]
                demo[i] = demo[j]
                demo[j] = temp
                
freq_count = deepcopy(demo)

for i in range(len(demo)):
    for j in range(i, len(demo)):
        if demo[i][2] < demo[j][2]:
            temp = demo[i]
            demo[i] = demo[j]
            demo[j] = temp
        elif demo[i][2] == demo[j][2]:
            if demo[i][1] < demo[j][1]:
                temp = demo[i]
                demo[i] = demo[j]
                demo[j] = temp

word2hmean = dict()
for x in word2freq.keys():
    word2hmean[x] = (2 * word2freq[x] * word2count[x]) / (word2freq[x] + word2count[x])  


with open(folder + "/harmonic sort.txt", "w") as f:       
    for key, value in sorted(word2hmean.items(), key=lambda item: item[1], reverse = True):
        f.write(str(key) + "\t" + str(value) + "\ttotal = " + str(word2count[key]) + "\tdeleted = " + str(word2deleteCount[key]) + "\tfrequency deleted = "  + str(word2freq[key]) + "\n")

        
plotCount = []
for x in word2freq.keys():
    plotCount.append(round(word2freq[x]))         
plotCount = dict(Counter(plotCount))


with open(folder + "/plotFreq.txt", "w") as f:
    for key, value in sorted(plotCount.items(), key=lambda item: item[0], reverse = True):
        f.write(str(key) + "\t" + str(value) + "\n")
  
'''
with open("temp.txt", "w") as f:
    for i in range(25):
        f.write(str(msrp[i][0]) + " & " + str(msrp[i][1])[0:6] + " & " + str(msrp[i+25][0]) + " & " + str(msrp[i+25][1]) [0:6]+ " & " + str(ai[i][0]) + " & " + str(ai[i][1])[0:6] + " & " + str(ai[i+25][0]) + " & " + str(msrp[i+25][1])[0:6] + " & " + str(sicke[i][0]) + " & " + str(sicke[i][1])[0:6] + " & " + str(sicke[i+25][0]) + " & " + str(sicke[i+25][1])[0:6] + "\\" + '\n')
'''    





