import numpy as np


members = np.load('./member.npy')
listmember = list(members)
file1 = open('./shope1','a')
file2 = open('./shope2','a')
file3 = open('./shope3','a')
count = 1;num = len(listmember)
print num
for li in listmember:
    if count <= 300:
        file1.write(li)
        file1.write('\n')
    if count > 300 and count <=600:
        file2.write(li)
        file2.write('\n')
    if count > 600 and count <=num:
        file3.write(li)
        file3.write('\n')
    count += 1
file1.close()
file2.close()
file3.close()
