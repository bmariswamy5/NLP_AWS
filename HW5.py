#package
import re

---------------------------------------------------------

#Q1
with open('/home/ubuntu/NLP_AWS/Email.txt','r',encoding='utf-8') as file:
    text = file.read()

pattern =re.compile(r'\b[A-Za-z][A-Za-z0-9]*@[A-Za-z]+\.[A-Za-z]+\b')
matches=pattern.finditer(text)

for match in matches:
    print(match)


#------------------ end of Q1 --------------------------

#Q2
with open('/home/ubuntu/NLP_AWS/war_and_peace.txt','r',encoding='utf-8') as f:
    text1=f.readlines()

names={}
pattern=r'\b[a-zA-Z]*ski\b'

for line in text1:
    matches=re.findall(pattern,line)

    for match in matches:
        if match in names:
            names[match] +=1
        else:
            names[match]=1


sorted_name=sorted(names.items(),key=lambda x:x[0])

for name,count in sorted_name:
    print(f"{name}:{count}")

#--------------------- Q2 end -----------------------------------------------

#Q3
# 1)remove space between digits
text = "12 0 mph is a very high speed in the 6 6 interstate.” to ”120 mph is a very high speed in the 66 interstate."
pattern2=r'(\d+) (\d+)'
res=re.sub(pattern2,r'\1\2',text)
print(res)

#2)Replace content in parenthesis with (XXXX)
text ="120 mph is a very high speed in the (66 interstate).” to ”120 mph is a (very high speed) in the 66 interstate."
pattern=r'\([^)]+\)'

res1=re.sub(pattern,'(xxxxx)',text)
print(res1)

#3)end with 'ly'
text = "She walked quickly and happily."

pattern=r'\b\w+ly\b'
matches = re.findall(pattern,text)
for match in matches:
    print(match)

#4)to find quotes
text = 'He said, "Hello, world!" She replied, "Hi there!"'
pattern=r'"(.*?)"'
matches=re.findall(pattern,text)

for match in matches:
    print(match)

#5)find 3,4,5 character in text
text = "Words with 3, 4, and 5 characters are here."
pattern = r'\b\w{3,5}\b'
matches = re.findall(pattern,text)
for match in matches:
    print(match)

#6)replace ',' with '-'
text = "Replace all, commas with hyphens, in this, sentence."
res=text.replace(',','-')

print(res)

#7)extract dates
url="https://www.yahoo.com/news/football/wew/2021/09/02/odell–famer-rrrr-on-one-tr-littleball–norman-stupid-author/"
pattern=r'/(\d{4})/(\d{2})/(\d{2})/'

match=re.search(pattern,url)
if match:
    year,month,day=match.groups()
    print(f"Year:{year},Month:{month},Dat:{day}")
else:
    print("date not found in URL")