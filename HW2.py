# HW2 Question 1
S1 = input("Enter the string:\n") # reading the string from user input
print("Number of uppercase letter in string are:",sum(1 for s in S1 if s.isupper())) #generates list of ones for each uppercase char in string and then adds up the ones generated giving the count of uppercase letters.
print("Number of lowercase letter in string are:",sum(1 for s in S1 if s.islower())) #generates list of ones for each lowercase char in string and then adds up the ones generated giving the count of lowercase letters.
print("Number of digits in string are:",sum(1 for s in S1 if s.isdigit())) #generates list of ones for each digit in string and then adds up the ones generated giving the count of digits
print("Number of whitespace in string are:",sum(1 for s in S1 if s.isspace())) #generates list of ones for each whitespace in string and then adds up the ones generated giving the count of whitespace in string.

# _________________ Question 1 end __________________________

# HW2 Question 2

S2 = input("Enter the string:\n") # reading the string from user input
