from math import *
import operator

def load_dataset():
    titleMat = []#导入属性名称列表
    dataMat = []#导入数据
    labelMat = set()#类别列表    
    fr = open('E:/人工智能/python/source/myID3/dataSet.txt','r')
    
    line = fr.readline()
    lineArr = line.strip().split()
    for i in range(len(lineArr)-1):
        titleMat.append(lineArr[i])
        
    for line in fr.readlines():
        lineArr = line.strip().split()
        record=[]
        for i in range(len(lineArr)):
            record.append(lineArr[i])
        dataMat.append(record)
        labelMat.add(lineArr[i])      
    fr.close()    
    return dataMat,labelMat,titleMat

#根据公式1计算出整个元组分类所需要的期望信息（熵）
def calculate_empirical_entropy(dataMat,labelMat):
    labelcount = {}
    #统计整个数组集合去打球和不去打球的次数
    for label in labelMat:
        i = 0
        for record in dataMat:            
            for attribute in record:                
                if label==attribute:
                    i = i+1
                    labelcount[label]=i
    a = len(dataMat)
    info = 0.0
    #公式1
    for label in labelMat:
        info = info-labelcount[label]/a*log2(labelcount[label]/a)    
    return info

#返回信息增益最高的属性位置索引
def calculate_infomation_gain(dataMat,labelMat):        
    attrs = []#将所有记录按属性划分成若干子集，并统计各属性出现的次数
    infoDj = []#返回计算各个属性的信息增益列表
    attributeMat = []#属性列表
    #得出属性列表
    for j in range(len(dataMat[0])):
        a = set()
        for k in dataMat:
            a.add(k[j])
        attributeMat.append(a)
    #得出各个属性在去打球和不去打球中出现的次数   
    for label in labelMat:
        atts = []
        for i in range(len(attributeMat)):
            attr = attributeMat[i]
            a = {x:0 for x in attr}
            for record in dataMat:
                if (record[len(record)-1]==label):                    
                    a[record[i]] = a[record[i]]+1                        
            atts.append(a)
        attrs.append(atts)
    #m为总记录数
    m = 0
    for a in attrs:        
        for k,v in a[(len(a)-1)].items():
            m=m+v
    #根据公式2、3计算各个属性的信息增益        
    for i in range(len(attrs[0])-1):
        l = 0.0
        for a in attributeMat[i]:
            f =0.0
            n =0                      
            for attr in attrs:        
                n = attr[i][a]+n            
            for attr in attrs:
                if attr[i][a]!=0:
                    f = -(attr[i][a]/n)*log2(attr[i][a]/n)+f                
            l = f*(n/m)+l
        infoDj.append((calculate_empirical_entropy(dataMat,labelMat)-l))#公式3
    #返回信息增益最高的属性位置索引
    for i,v in enumerate(infoDj):
        if v==max(infoDj):
            return i           

#当没有多余的feature，但是剩下的样本不完全是一样的类别是，采用出现次数多的那个类别
def classify(classList):  
    ''''' 
    find the most in the set 
    '''  
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys():  
            classCount[vote] = 0  
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)  
    return sortedClassCount[0][0]

#选择完分裂属性后，就行数据集的分裂：
def splitDataSet(dataMat,feat,values):  
    retdataMat = []  
    for record in dataMat:  
        if record[feat] == values:  
            reducedFeatVec = record[:feat]  
            reducedFeatVec.extend(record[feat+1:])  
            retdataMat.append(reducedFeatVec)  
    return retdataMat

def deletefeat(titleMat,feat):
    """ 以下写法会出错，可能是地址引用的原因，还不太清楚，稍后再研究
    del titleMat[feat]
    subtitleMat = titleMat
    """
    subtitleMat = []
    for title in titleMat:
        if title!=titleMat[feat]:
            subtitleMat.append(title)            
    return subtitleMat
        
#构造决策树,采用python字典来递归构造
def create_tree(dataMat,titleMat,labelMat):
    #如果记录结果都一样，直接返回结果
    classList = [example[-1] for example in dataMat]  
    if classList.count(classList[0])==len(classList):  
        return classList[0]  
    if len(dataMat[0])==2:# no more features  
        return classify(classList)

    bestFeat = calculate_infomation_gain(dataMat,labelMat)#bestFeat is the index of best feature  
    bestFeatLabel = titleMat[bestFeat]  
    myTree = {bestFeatLabel:{}}  
    featValues = [example[bestFeat] for example in dataMat]  
    uniqueFeatValues = set(featValues)    
    subtitleMat = deletefeat(titleMat,bestFeat)
    for values in uniqueFeatValues:  
        subdataMat = splitDataSet(dataMat,bestFeat,values)
        print(subtitleMat,values)
        myTree[bestFeatLabel][values] = create_tree(subdataMat,subtitleMat,labelMat)#递归  
    return myTree

#下面根据上面构造的决策树进行数据的分类
def predict(tree,newObject):  
    while isinstance(tree,dict):#isinstance判断实例是否是这个类或者object的变量  
        key = list(tree.keys())[0]  #list(dict)
        tree = tree[key][newObject[key]]  
    return tree

dataMat = []
labelMat = set()       
dataMat,labelMat,titleMat=load_dataset()
tree = create_tree(dataMat,titleMat,labelMat)
print(tree)
print(predict(tree,{'天气':'雨天','温度':'冷','湿度':'高','风力':'强'}))
print(predict(tree,{'天气':'多云','温度':'冷','湿度':'高','风力':'强'}))
print(predict(tree,{'天气':'晴朗','温度':'冷','湿度':'高','风力':'强'}))
print(predict(tree,{'天气':'雨天','温度':'适中','湿度':'高','风力':'弱'}))
print(predict(tree,{'天气':'多云','温度':'冷','湿度':'高','风力':'强'}))
print(predict(tree,{'天气':'晴朗','温度':'冷','湿度':'高','风力':'强'}))
