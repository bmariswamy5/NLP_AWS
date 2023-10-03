#package
import re

---------------------------------------------------------

#Q1
with open("/home/ubuntu/NLP_AWS/Email.txt','r',encoding='utf-8") as file:
    text = file.read()

pattern =re.compile(r'\b[A-Za-z][A-Za-z0-9]*@[A-Za-z]+\.[A-Za-z]+\b')
matches=pattern.finditer(text)

for match in matches:
    print(match)


#------------------ end of Q1 --------------------------

#Q2
with open("/home/ubuntu/NLP_AWS/war_and_peace.txt','r',encoding='utf-8") as f:
    text1=f.readlines()

names={}
pattern1=re.complie(r'\b[A-Za-z]*ski\b')
matches=pattern1.finditer(text1)

for match in matches:
    names[match]+=1

sorted_name=sorted(names.items(),key=lambda x:x[0])

for name,count in sorted_name:
    print(f"{name}:{count}")

#--------------------- Q2 end -----------------------------------------------

#Q3
# 1)remove space between digits
text = "12 0 mph is a very high speed in the 6 6 interstate.” to ”120 mph is a veryhigh speed in the 66 interstate.”
pattern2=re.complie()