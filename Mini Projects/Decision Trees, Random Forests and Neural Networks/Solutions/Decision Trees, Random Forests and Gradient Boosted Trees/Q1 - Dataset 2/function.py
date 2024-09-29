import pickle
c=1
file = open("experiment.txt","wt")        
def counter(n):
    for i in range(0,n):
        i+=1
        c=i
        print(c)
        file.write(str(c))
        file.write("\n")
    return c

a = counter(400)

# with open('experiment.txt','wb') as f:
#     pickle.dump(a,f) 

