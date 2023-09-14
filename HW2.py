# HW2 Question 1

S1 = input("Enter the string:\n") # reading the string from user input
print("Number of uppercase letter in string are:",sum(1 for s in S1 if s.isupper())) #generates list of ones for each uppercase char in string and then adds up the ones generated giving the count of uppercase letters.
print("Number of lowercase letter in string are:",sum(1 for s in S1 if s.islower())) #generates list of ones for each lowercase char in string and then adds up the ones generated giving the count of lowercase letters.
print("Number of digits in string are:",sum(1 for s in S1 if s.isdigit())) #generates list of ones for each digit in string and then adds up the ones generated giving the count of digits
print("Number of whitespace in string are:",sum(1 for s in S1 if s.isspace())) #generates list of ones for each whitespace in string and then adds up the ones generated giving the count of whitespace in string.

# _________________ Question 1 end __________________________

# HW2 Question 2

S2 = input("Enter the string to be swapped:") # reading the string from user input
print(S2[1:]+S2[0]) # slice the string from position 1 to end of string and then add the first character to the end of the string

# ----------------------- Question 2 end ----------------------------------------------

# HW2 Question 3

S3 = input("Enter the name:") #Reading the first,middle and last name from user
initial="".join([new[0] for new in S3.split()]) #split the  input string and for each strung after split get the character from position 0
print("The initial are:",initial.upper()) # convert the initial to uppercase and print

# ----------------------- Question 3 end ----------------------------------------------
#HW2 Question 4

while True:
    password=input("Enter the password:") # Enter the string
    if(len(password) >= 8 and #Condition to check if the string is greater than 8 character with atleast one uppercase,lower case and digit to be present
        any(char.isupper() for char in password) and #if the entered string matches the condition the loop breaks else it keeps giving the prompt to enter the right string
        any(char.islower() for char in password) and
        any(char.isdigit() for char in password)):
        print('The entered password is:',password)
        break
    else : print("does not meet requirements")

# ----------------------- Question 4 end ----------------------------------------------
#HW2 Question 5
from collections import Counter
S5=input("Enter the string:")
character_count=Counter(s.lower() for s in S5 if s.isalpha())
count_length={count:chars for chars,count in character_count.items()}

for count,character_count in sorted(count_length.items()):
    print(f"character repeated {count} times:", ','.join(character_count))




from collections import Counter

input_string = input("Enter a string: ").lower()
char_count = Counter(filter(str.isalpha, input_string))

for char, count in char_count.items():
    if count == 1:
        print(char, end=',')
    else:
        print(f"{char}={count}")
# ----------------------- Question 5 end ----------------------------------------------
# HW2 Question 6
from itertools import product
S6 =  input("Enter the input string:")
combine=[''.join(s)  for s in product(*zip(S6.lower(),S6.upper()))]
print(combine)

# ----------------------- Question 6 end ----------------------------------------------

# HW Question 7
from collections import OrderedDict
from collections import Counter
with open('/home/ubuntu/NLP_AWS/test.txt','r') as f:
    print(f.readlines()) #read all the lines from the text file
    f.seek(0) #go to the first line in the file,since previously we read all the line in the file now the cursor is ar the end if the file.
    longest_word=max(f.read().split(),key=len) #read the file from beginning and split the sentence into words and find the length of each word and store the max length of string in variable and prin the longest word
    print("The longest word in the file is:",longest_word)
    f.seek(0)
    print("the number of lines in the file is:",len(f.readlines())) # print the number lines using len funtion
    f.seek(0)
    print(Counter(f.read().split())) # check for word frequency using counter function
    f.close()

# ----------------------- Question 7 end ----------------------------------------------