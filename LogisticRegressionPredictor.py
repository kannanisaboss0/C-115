#-------------------------------------------------LogisticRegressionPredictor.py-------------------------------------------------#
'''
Importing modules:
-pandas (pd)
-LogisticRegression (LogisticRegression) :-sklearn.linear_model
-matplotlib (plt)
-numpy (np)
-plotly.express (px)
-time (tm)
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression as LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import time as tm

#Defining a function to model the logistic regression values based on the paramaters provided
def FitRegressionModel(x_arg,y_arg):
  lr=LogisticRegression()
  lr.fit(x_arg,y_arg)

  x_test=np.linspace(0,20,40)

  y_value=x_test*lr.coef_+lr.intercept_
  y_sigmod=1/(1+np.exp(-y_value)).ravel()

  return x_test,y_sigmod,lr.coef_,lr.intercept_


#Defining a function to determine the color of the graph and logistic model
def ChooseColorForPlot(color_plot_arg):
  print("Please choose the color for the {}".format(color_plot_arg))

  color_list=["Unusable_Element","black","yellow","orange","red","pink","purple","green","darkgreen","maroon"]
  color_count=0

  for color in color_list[1:]:
    color_count+=1
    print(str(color_count)+":"+color)

  input_param=int(input("Please enter the index of the color:"))
  color_choice=color_list[input_param]

  return color_choice


#Defining a function to ask the user whether to zoom or not
def AskForZooming():
  input_ask_param=input("Improve accuracy by zooming in the graph?(:-Yes or No)")

  #Assessing the user's choice on the option to increase accuracy
  #Case-1 
  if(input_ask_param=="yes" or input_ask_param=="Yes"):
    return "Yes"

  #Case-2
  else:
    return "No"  
 

#Defining a function to predict the y-value with the stipulated x-value
def PredictYValue(slope_arg,intercept_arg,x_arg,y_arg):
    x_value=float(input("Please enter the {} whose {} is to be found:".format(x_arg,y_arg)))

    y_value_pre_determined=x_value*slope_arg+intercept_arg

    y_value_final=1/(1+np.exp(-y_value_pre_determined))
    y_value_final=y_value_final.ravel()[0]

    y_value_final_percentage=y_value_final*100

    print("The y-value in corrrespondence to the x-value is {}".format(round(y_value_final,2)))

    print("The success percentage of the x-value through the derivation (1/(1+e^-x)) is {}%".format(round(y_value_final_percentage,2)))

    continue_further_prediction=input("Continue further prediction?(:-Yes or No)")

    #Assessing the user's choice for continuing the prediction
    #Case-1
    if(continue_further_prediction=="Yes" or continue_further_prediction=="yes" ):
      PredictYValue(slope_arg,intercept_arg,x_arg,y_arg)

    #Case-2
    else:
      print("Request Accepted")  


#Defining a function to display the statisitcs of the data
def DisplayStatsiticData(dict_param_1_arg,dict_param_2_arg,y_arg,x_arg):
  print("The {}(y-value) and {}(x-value) share relationship which can be factored by the derivation '{}=1/(1+e^-{})'(y=1/(1+e^-x))".format(y_arg,x_arg,x_arg,y_arg))
  
  dict_object_param={"x":dict_param_1_arg,"y":dict_param_2_arg}

  correlation_param=np.corrcoef(dict_object_param["x"],dict_object_param["y"])
  correlation_param_range=correlation_param[0,1]

  correlation_param_range_percentage=correlation_param_range*100
  correlation_determination_percentage=(correlation_param_range**2)*100

  #Verifying whether the correlative coefficient of the data is lesser, equal to or greater than 0
  #Case-1
  if(correlation_param_range<0):
    print("The values share a inverse correlation")

  #Case-2
  elif(correlation_param_range>0):
    print("The values share a direct correlation")

  #Case-3
  elif(correlation_param==0):
    print("The values share no profound correlation")

  print("The veracity of the derivation is {}%".format(round(correlation_param_range_percentage,2)))    
  print("{}% of differences in {} can be explained by {}".format(round(correlation_determination_percentage,2),x_arg,y_arg))



#Reading data from the file
df=pd.read_csv("data.csv")

print("Welcome to LogisticGegressionPerdictor.py. We provide data prediction through logistic regression of a particular dataset.")

view_information=input("Do not know what logistic regression is?(:- I Don't Know or I Know)")

#Verifying the user's choice whether they have pre-requisiste knowledge of Logisitic Regression
#Case-1
if(view_information=="I Don't Know" or view_information=="i don't know" or view_information=="I don't know"):
  print("What is logistic regression?")
  tm.sleep(3.0)

  print("Logistic regression is a relationship shared by two values in a linear relationship, which can be mathematically expressed as:y=1/(1+e^-x)")
  tm.sleep(3.1)

  print("The dependent variable y can be predicted using the equation with the help of the indpendent variable x.")
  tm.sleep(2.0)

  print("The dependent variable y is a binary value(can only assume two magnitudes, typically 0 or 1).")
  tm.sleep(1.6)

  print("The higher binary value is the maximum possibility of occurence(or the maximum value attainable).")
  tm.sleep(1.8)

  print("The lower binary value is the minimum possibility of occurence(or the minimum value attainable).")
  tm.sleep(1.7)

  print("What is the difference between logistic and linear regression?")
  tm.sleep(2.5)

  print("Linear regression uses the independent variable to predict the dependent variable continously")
  tm.sleep(2.3)

  print("Logistic regression uses the independent variable to predict the dependent variable on the basis of two values, preferably 0 and 1")
  tm.sleep(2.3)

  print("Where is logistic regression used?")
  tm.sleep(2.4)

  print("Logistic regression is used in analysing data which can only provide two values or answers")
  tm.sleep(2.4)

  print("Such as:")
  tm.sleep(0.5)

  print("1. Predicting the desicions of the average customer using previously recorded data by corporations.")
  tm.sleep(1.9)

  print("2. Estimating periphrastic values to manipulate the result of an equation by the sceintific community.")
  tm.sleep(2.0)

  print("3. Modelling survey results to certain number by demographers, in case of data scarcity.")
  tm.sleep(2.0)

  print("Logistic regression is integrated several more possibilities and fields.")
  print("To know more about logistic regression, visit: 'https://en.wikipedia.org/wiki/Logistic_regression' ")
  tm.sleep(3.8)

print("Loading Data...")
tm.sleep(2.3)

velocity_escape_list=["Unusable_Element","Velocity","Escaped"]
velocity_escape_count=0

for velocity_escape in velocity_escape_list[1:]:
  velocity_escape_count+=1
  print(str(velocity_escape_count)+":"+velocity_escape)

x_input=int(input("Please enter the index of the field preffered to be the x-axis:"))
y_input=int(input("Please enter the index of the field preffered to be the y-axis:"))

x_choice=velocity_escape_list[x_input]
y_choice=velocity_escape_list[y_input]

#Verifying whether the x-axis and y-axis are not the same value
#Case-1
if(x_choice!=y_choice):
  df_x=df[x_choice].tolist()
  df_y=df[y_choice].tolist()

  scatter_plot_color=ChooseColorForPlot("scatter plot")
  logistic_plot_color=ChooseColorForPlot("line of the logistic regression plot")

  x=np.reshape(df_x,(len(df_x),1))
  y=np.reshape(df_y,(len(df_y),1))

  plt.figure()

  #Verifying whether the x-axis is the field 'Velocity' or not
  #Case-1
  if(x_choice=="Velocity"): 
    scatter=plt.scatter(x.ravel(),y,color=scatter_plot_color,zorder=45)

  #Case-2
  elif(x_choice=="Escaped"):
    scatter=plt.scatter(x,y.ravel(),color=scatter_plot_color,zorder=45)

  #Verifying whether the x-axis is the field 'Velocity' or not
  #Case-1
  if(x_choice=="Velocity"):
    x_linear,y_linear,slope_linear,intercept_linear=FitRegressionModel(x,y)

    plot=plt.plot(x_linear,y_linear,color=logistic_plot_color,linewidth=5)

    plt.axhline(y=0,linestyle="-",color="blue")
    plt.axhline(y=0.5,linestyle="--",color="blue")
    plt.axhline(y=1,linestyle="-",color="blue")
    plt.axvline(x=x_linear[22],linestyle="--",color="blue")

    func_zoom=AskForZooming()

    #Assessing the user's choice to increase accuracy by zooming or not
    #Case-1
    if(func_zoom=="Yes"):
      plt.xlim(8.25,13.5)

    #Case-2
    else:
      print("Request Accepted.") 

    print("Statistics:") 
    DisplayStatsiticData(x_linear,y_linear,x_choice,y_choice)

    prediction_ask=input("Predict the value of {}?(:-Yes or No)".format(y_choice))

    #Assessing the user's choice to predict the y-axis value or not
    #Case-1
    if(prediction_ask=="Yes" or prediction_ask=="yes"):
      PredictYValue(slope_linear,intercept_linear,x_choice,y_choice)

  #Case-2
  elif(x_choice=="Escaped"):
    x_linear,y_linear,slope_linear,intercept_linear=FitRegressionModel(y,x)

    plot=plt.plot(y_linear,x_linear,color=logistic_plot_color,linewidth=5)

    plt.axvline(x=0,linestyle="-",color="blue")
    plt.axvline(x=0.5,linestyle="--",color="blue")
    plt.axvline(x=1,linestyle="-",color="blue")
    plt.axhline(y=x_linear[22],linestyle="--",color="blue")

    func_zoom=AskForZooming()

    #Assessing the user's choice to increase accuracy by zooming or not
    #Case-1
    if(func_zoom=="Yes"):
      plt.ylim(8.25,13.5) 

    #Case-2   
    else:
      print("Request Accepted.")  

    print("Stataistics:")  
    DisplayStatsiticData(y_linear,x_linear,y_choice,x_choice)

    prediction_ask=input("Predict the value of {}?(:-Yes or No)".format(x_choice))

    #Assessing the user's choice to predict the y-axis value or not
    #Case-1
    if(prediction_ask=="Yes" or prediction_ask=="yes"):
      PredictYValue(slope_linear,intercept_linear,y_choice,x_choice)

    #Case-2
    else:
      print("Request Accepted.")

      #Prinitng the ending message
      print("Thank You for using LogisticRegressionPredcitor.py.")  

  

  print("Generating Graph...")  
  tm.sleep(3.9)
  print("Graph Generated")
  tm.sleep(1.0)

  plt.title("Velocity Escape Chart")
  plt.xlabel(x_choice)
  plt.ylabel(y_choice)
  plt.show()
    
  #Prinitng the ending message
  print("Thank you for using LogisticRegressionPredictor.py.")  

#Case-2
else:
  print("Request Terminated.")
  print("Please choose different fields for the x and y axes.")

  #Prinitng the ending message
  print("Thank you for using LogisticRegressionPredictor.py.")
#-------------------------------------------------LogisticRegressionPredictor.py-------------------------------------------------#