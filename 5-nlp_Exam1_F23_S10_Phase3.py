#============================Competition=======================================================================================================
# This dataset is about classify sample text to its associated class.
#**********************************
# Import Libraries



#**********************************
# Loading Dataset
train_filename = r"Train.csv"
test_filename = r"Test.csv"



#**********************************


#**********************************




#**********************************
print(20*'-' + "NB" + 20*'-')



#**********************************

print(20*'-' + "Logistic Regression" + 20*'-')




df_test['Target'] = predicted
df_test.to_csv('Test_submission_ajafari.csv')
