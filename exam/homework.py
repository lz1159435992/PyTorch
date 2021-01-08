import re
def deleteNote(code):
    skip = False
    for i in range(len(code)):
        if "/*" in code[i]:
            skip = True
            if "*/" in code[i]:
                skip = False
                code[i]=code[i].split("/*")[0]
        elif "*/" in code[i]:
            skip=False
            code[i]=""
        if skip==True:
            code[i]=""
    return code

def getFunctionName(code):
    count_Braces = 0
    functionName = []
    for data in code:
        if count_Braces == 0 and "(" in data:
            name = data.split("(")[0]
            if " " in name:
                name = name.split(" ")[-1]
            functionName.append(name)
        if "{" in data:
            count_Braces += 1
        if "}" in data:
            count_Braces -= 1
    return functionName
    
def getKeywords(code,keywords):
    keywordsInCode = []
    for data in code:
        if data == "":
            continue
        data=re.sub("\t","",data)
        data=re.sub("[^A-Za-z0-9_.]"," ",data)
        splitedData = data.split(" ")
        for i in range(len(splitedData)):
            splitedData[i]=re.sub(" ","",splitedData[i])    #把一句话分割为数个单词
        # print("splitedData:",splitedData)
        for keyword in keywords:
            if keyword in splitedData:
                keywordsInCode.append(keyword)
        keywordsInCode=list(set(keywordsInCode))
        keywordsInCode.sort()
    return keywordsInCode

def getCycComplexity(code,functionName):
    CycComplexity = 1
    functionCode = []
    judgeNode = ["if","case"]
    count_braces = 0
    addCode = False
    meetParentheses = False
    for data in code:
        if addCode == True:
            functionCode.append(data)
            if "{" in data:
                count_braces += 1
                meetParentheses = True
            if "}" in data:
                count_braces -= 1
            if count_braces == 0 and meetParentheses == True:
                break
        elif functionName in data:
            addCode = True
            functionCode.append(data)
    for data in functionCode:
        data=re.sub("\t","",data)
        data=re.sub("[^A-Za-z0-9_.]"," ",data)
        splitedData = data.split(" ")
        for i in range(len(splitedData)):
            splitedData[i]=re.sub(" ","",splitedData[i])    #把一句话分割为数个单词
        for word in judgeNode:
            if word in splitedData:
                CycComplexity += 1
    return CycComplexity

if __name__ == "__main__":
    keywords = ["auto","break","case","char","const","continue","default","do","double","else",
                "enum","extern","float","for","goto","if","int","long","register","return",
                "short","signed","sizeof","static","struct","switch","typedef","union","unsigned",
                "void","volatile","while"]
    f = open("program2.c",mode="r")
    code = f.readlines()
    code = deleteNote(code)
    f.close()
    f = open("function_name_list.txt",mode="w")
    for name in getFunctionName(code):
        f.write(name+"\n")
    f.close()
    f = open("keywords_list.txt",mode="w")
    for keyword in getKeywords(code,keywords):
        f.write(keyword+"\n")
    f.close()
    f = open("Cyclomatic_Complexity.txt",mode="w")
    f.write(str(getCycComplexity(code,"ext4_dax_huge_fault")))
    f.close()
    # print(getCycComplexity(code,"ext4_dax_huge_fault"))
    print("Analyze complete!")