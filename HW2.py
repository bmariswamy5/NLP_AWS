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
