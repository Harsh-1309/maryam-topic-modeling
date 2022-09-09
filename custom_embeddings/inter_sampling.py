from curses import window
from multiprocessing import context


DATASET_FILE ="data/text6"
WINDOW = 3
WINDOW_FLOAT = 3.0
MAX_RATE = 8.0
DIMENSION = 10
# initially dim was 100
SAMPLE_ACCURACY = 4000
WORDS=[]

def counter_update(list,newlist):
    print(newlist)
    for _,x in enumerate(newlist):
        # value for key x same in both list and new list then go into if
        print("x _",x,_)
        if x in list.keys():
            list[x][0] = WINDOW_FLOAT - (list[x][0]+newlist[x][0])/2
            list[x][1] += newlist[x][1]
        else:
            list[x]=newlist[x]
    return list

context={}

def makeContext():
    window={}
    f=open(DATASET_FILE,"r")
    #iterate through each line of the dataset
    for line in f:
        line=line.strip().split(" ")
        doclen=len(line)
        # print(doclen)

        for i,k in enumerate(line):
            window={}
            prev,next=line[max(i - WINDOW, 0):i], line[i + 1:min(1 + WINDOW + i, doclen)]
            print("\nk",k,"\n")
            for mines,j in enumerate([prev,next]):
                pr=0
                nlen=len(j)
                print("j",j)

                for w,x in enumerate(j):
                    print("w,x",w,x)
                    if mines == 0:
                        p = (nlen-w) -pr
                    else:
                        p = (w+1) - pr
                    print("compare x and k here")
                    if x == k:
                        pr +=1
                    else:
                        if x in window.keys():
                            wx=window[x]
                            window[x][0]=WINDOW_FLOAT - ((wx[0]+p)/(wx[1]+1))
                            print("this weird wx",wx)
                        else:
                            window[x]=[p,0,0]
                        window[x][1]+=1
                print("wxxxx",window)

            if k in context.keys():
                print("before counter update",context)
                context[k]=counter_update(context[k],window)
                print("after counter update",context)
            else:
                WORDS.append(k)
                context[k]=window
                print("else wala context print",context)

makeContext()