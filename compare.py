f1 = open("trainout.txt", "r")
f2 = open("testclass.txt", "r")
i, count=0, 0
while(i<998):
    a = f1.readline()
    b = f2.readline()
    if a==b:
        count= count + 1
    else:
    	fil = str(a) +" "+ str(b)
    	print fil
    i = i+1
print count
