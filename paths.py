#by Joshua Cook
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import sample

#converts vertex to cell coordinates
def v2c(vertex,size):
    return (vertex//size[0],vertex%size[0])

#converts cell coordinates to vertex    
def c2v(r,c,size):
    return r*size[0]+c

#converts a map to a graph    (list of list of connections)
def make_graph(Map,size,start,end):
    rows=size[0]
    cols=size[1]
    graph=[[] for i in range(rows*cols)] #empty graph
    #shifts=[[-1,0],[0,-1],[1,0],[0,1]]
    shifts=[[-1,0],[0,-1],[1,0],[0,1],[-1,-1],[1,-1],[-1,1],[1,1]] #allows for diagonals                                
    for r in range(rows):
        for c in range(cols):
            #if Map[r][c]==0: #if valid space
            for idx in range(len(shifts)):
                s=shifts[idx]
                if 0 <= r+s[0] < rows and 0 <= c+s[1] < cols : #if not off of the map
                    if Map[r+s[0]][c+s[1]]==0 or c2v(r+s[0],c+s[1],size)==start or c2v(r+s[0],c+s[1],size)==end : #if adjecent cell is valid
                        if idx<4 or Map[r][c+s[1]]==0 or Map[r+s[0]][c]==0:
                            graph[c2v(r,c,size)].append(c2v(r+s[0],c+s[1],size)) #add edge to edge list
                        
    return graph

#huristic function - distance    
def heuristic(startv,stopv,size,type=0):
    x1,y1 = v2c(startv,size)
    x2,y2 = v2c(stopv,size)
    #return 0  #dijkstras
    #return abs(x1-x2)+abs(y1-y2) #manhattan
    if type == 1:
        return ((x1-x2)**2+(y1-y2)**2)**.5 #euclidean
    else:
        return abs(x1-x2)+abs(y1-y2) #manhattan

#finds the front of the priority queue (returns the vertex in oset with the
#lowest cost determined by cost)
def min_cost(oset,cost):
    min_cost=1e10
    min_vtex=-1
    for s in oset:            
        if cost[s]<min_cost:
            min_vtex=s
            min_cost=cost[s]
    return min_vtex

#A* algorithm: finds a path from start to end given graph and the size of the graph        
def a_star(start,end,graph,size):
    oset=[start] #open set
    cset=[]      #closed set
    
    path=[-1 for i in range(len(graph))]
    gcost=[1e9 for i in range(len(graph))] #g cost function
    fcost=[1e9 for i in range(len(graph))] #f cost (g cost + h cost)
    
    #set costs fro starting node                            
    gcost[start]=0
    fcost[start]=heuristic(start,end,size)
    
    #loops until open set is empty and there is nothing more to check                        
    while len(oset) > 0:
        
        #current = front of "priority queue"
        current=min_cost(oset,fcost)
        #print(current,oset)
        #if reached end, reconstruct the optimal path and return it
        if current==end:
            end_path=[(current,gcost[current])]
                                                    
            #walk down the graph, adding the vertices to the path to the end                                        
            while current != start:
                current=path[current]
                end_path.append( (current,gcost[current]) )
            return end_path
            
        #current vtex is no longer in the open set and now in the closed set    
        oset.remove(current)
        cset.append(current)
        
        #loop through the connection
        for vtex in graph[current]:
            
            #if vertex is in closed, its been checked, therefor skip it
            if vtex in cset:
                continue
            
            #if its not in open set, add it to be checked later
            if not vtex in oset:
                oset.append(vtex)
            
            #calculate the relative cost    
            ncost=gcost[current] + heuristic(current,vtex,size,1) #<- 1 is distance between neighbor/edge cost
            
            #if this cost is greater than the vtexs current cost, it is worse so ignore it
            if ncost >= gcost[vtex]:
                continue
            
            #otherwise its an improvement, so update the vertex's fcost and gcost
            gcost[vtex]=ncost
            fcost[vtex]=gcost[vtex]+heuristic(vtex,end,size)
            
            #also save this as part of its best path for later pather reconstruction
            path[vtex]=current
        
    return []
def generate(seed,show=0):
    if seed>=0:
        np.random.seed(seed)
    Map=np.zeros((60,60))
                                        

    #get the size of the map
    size=(len(Map),len(Map[0]))

    def SAMPLE(Map):
        lst=[index for index,x in np.ndenumerate(Map) if x==0]
        return list(sample(lst,1)[0])
    bad=[]
    good=[]
    for i in range(10000):

        x=SAMPLE(Map)
        y=SAMPLE(Map)

        while (abs(x[0]-y[0])+abs(x[1]-y[1]))<3 or x+y in bad or y+x in bad:
            x=SAMPLE(Map)
            y=SAMPLE(Map)

        pt=x+y

            

        #make graph from map
        
        start=c2v(pt[0],pt[1],size)
        end=c2v(pt[2],pt[3],size)
        graph=make_graph(Map,size,start,end)
        #generate path from a to b
        path=a_star(start,end,graph,size)
        if len(path)>0:
            print("path " +str(i)+": ",pt)
            good.append([v2c(vtex,size) for vtex,cost in path])
        else:
            print("path " +str(i)+": Fail")
            bad.append(pt)

        x=[]
        y=[]
        #print out path and path info
        for p in path:
            vtex,cost=p
            r,c=v2c(vtex,size)
            y.append(r)
            x.append(c)
            Map[r][c]=1
        if show:
            plt.plot(x,y,"-g")

    with open("paths/paths"+str(seed)+".pkl", 'wb') as handle:
        pickle.dump(good, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if show:
        plt.show()
        plt.xlim(0,size[0])
        plt.ylim(0,size[0])

import multiprocessing as mp
procs=[]
for i in range(4):
    p=mp.Process(target=generate,args=(i,))
    p.start()
for p in procs:
    p.join()