                    
                    # CASE STUDY 5 -  ECOMMERCE WEB ANALYTICS

# Getting the data

mydata<- read.csv("E:\\Analytixlabs\\Module 6 (Data science using R)\\Case Studies\\Case study 5 - Classification\\train.csv")
newdata<- read.csv("E:\\Analytixlabs\\Module 6 (Data science using R)\\Case Studies\\Case study 5 - Classification\\test.csv")

View(mydata)
str(mydata)
names(mydata)


# Exploratory Data analysis

  #  Getting all the numerical variables

var_con<- c("metric1","metric2","metric6","metric3","metric4","metric5","page1_top",
            "page1_exits","page2_top","page2_exits","page3_top","page3_exits","page4_top",
            "page4_exits","page5_top","page5_exits","page6_top","page6_exits")

  #  Getting all the categorical variables

var_cat<- c("binary_var1","binary_var2","visited_page1","visited_page2","visited_page3",
            "visited_page4","visited_page5","visited_page6","target") 
  
  
# Calculating the descriptives

udf<- function(x){
  nmiss<- sum(x[x==-1])
  mean<- mean(x)
  max<- max(x)
  min<- min(x)
  p1<- quantile(x,0.01)
  p99<- quantile(x,0.99)
  outlier_flag<- max>p99|min<p1
  return(c(nmiss=nmiss,mean=mean,max=max,min=min,P1=p1,P99=p99,Flag=outlier_flag))
}

stats_con<- data.frame(t(apply(mydata[var_con],2,udf)))
View(stats_con)
# All the page exits variables have lot of missing values,almost 95% of their data is missing
# So there is no point using these variables in model building


udf2<- function(x){
  nmiss=sum(x[x==-1])
  return(nmiss=nmiss)
}

stats_cat<- data.frame(apply(mydata[var_cat],2,udf2))
View(stats_cat) 
# No missing values in categorical variables

#  Splitting the data into training and testing datasets
ind<- sample(2,nrow(mydata),replace = TRUE,prob = c(0.7,0.3))
training<- mydata[ind==1,]
testing<- mydata[ind==2,]

#  Data Preparation

prop.table(table(mydata$target))
prop.table(table(training$target))
prop.table(table(testing$target))

#  The distribution of target variable in training and testing datasets is similar
#  to that of original dataset (mydata)

training$binary_var1<- factor(training$binary_var1)
training$binary_var2<- factor(training$binary_var2)
training$visited_page1<- factor(training$visited_page1)
training$visited_page2<- factor(training$visited_page2)
training$visited_page3<- factor(training$visited_page3)
training$visited_page4<- factor(training$visited_page4)
training$visited_page5<- factor(training$visited_page5)
training$visited_page6<- factor(training$visited_page6)
training$target<- factor(training$target)
str(training)

# Since the dataset is very large so if we create our model using this dataset the R will 
# throw us an error: "Cannot allocate a vector of this size"
# So to prevent this from happening we take a sample with less no. of observations to
# create our model

training<- training[1:20000, ]
prop.table(table(training$target)) # We have the same distribution so we can proceed further

#  Model Creation:  We will use Random Forest ML model
library(dplyr)
training<- subset(training,select = -c(unique_id,region,sourceMedium,device,country,
                                       dayHourMinute,page1_exits,page2_exits,page3_exits,
                                       page4_exits,page5_exits,page6_exits))
# We have removed all the variables from training datasets that aren't potential predictors
library(caret)
library(randomForest)

names(training)
set.seed(123)
fit<- randomForest(target ~ ., data = training)
print(fit)
# We can see from the result that ntree=500 and mtry=4
# Since the model that we have created is the basic implementation of Random forest
# Hence we can see that OOB is high,almost 9% and class error is also very high
# so we need fine tuning of rf model and this can be done by changing the number 
# of trees(ntree) and no. of variables to be used at a node for splitting(mtry)

# We will start with error rate
plot(fit)
# We can see that after 300 trees, the error rate is almost constant

  # Fine tuning RF model

tuneRF(training[ ,-21],training[ ,21],
       stepFactor = 0.5,
       plot = TRUE, # This plot obb error as function of mtry(no. of variables to be used for splitting)
       ntreeTry = 300, # default is 500
       trace = TRUE,
       improve = 0.05)
# OBB error is minimum when mtry=4
# Based on our learning about ntree and mtry, we will build a modified  RF model

    #  Modified RF model
set.seed(999)
fit2<- randomForest(target ~ ., data = training,
                    ntree=300,
                    mtry=4,
                    Importance=TRUE,
                    Proximity=TRUE)
print(fit2)
# There is not much improvement in class error but OOB is little reduced this time

#  Confusion matrix and Accuracy:
testing$target<- factor(testing$target)
testing$binary_var1<- factor(testing$binary_var1)
testing$binary_var2<- factor(testing$binary_var2)
testing$visited_page1<- factor(testing$visited_page1)
testing$visited_page2<- factor(testing$visited_page2)
testing$visited_page3<- factor(testing$visited_page3)
testing$visited_page4<- factor(testing$visited_page4)
testing$visited_page5<- factor(testing$visited_page5)
testing$visited_page6<- factor(testing$visited_page6)

pred<- predict(fit,newdata = testing)
head(pred,20)
head(testing$target,20)
# Till the first 20 observations in testing dataset our prediction is correct
confusionMatrix(pred,testing$target,positive = "1")
# Accuracy > No Information rate , so we can accept this model

#  Now we will see how the second model(modified model) performs
pred2<- predict(fit2,newdata = testing)
confusionMatrix(pred2,testing$target,positive = "1")
# Accuracy and sensitivity is bit improved by this model
# But Sensitivity is still very low. This is because of the class imbalance in target variable


# We can also see how important a variable is
varImpPlot(fit2,
           sort = TRUE,
           n.var = 10,
           main = "Top 10 imp variables")
# This plot shows the top 10 important variables

#++++++++++++++++++++++++++++++++ Business Problem +++++++++++++++++++++++++++++++++++++++++#

# Now we have the final model with very good overall accuracy but the problem with the
# model is that accuracy of predicting "1"s ,i.e.,Sensitivity is very poor.

# According to the Business problem we are required to correctly predict the "1"s for 
# each unique ID, i.e., We are more interested in predicting the "1"s instead of focussing 
# much on overall accuracy of the model.

# The problem here is the high class imbalance in our dataset that results in very poor
# sensitivity
prop.table(table(training$target))
prop.table(table(testing$target))
# We can see from these proportion tables that the number of "1"s in target variable in both
# training and testing datasets is only about 10%,which is very low for a large dataset.
# This is the main reason why our model hasn't got perfectly trained to correctly predict
# the "1"s.


#  This problem can be solved by handling the class imbalance in our data.

#+++++++++++++++++++++++++++++ Handling Class Imbalance +++++++++++++++++++++++++++++++++++#

# The method we use for negating the class imbalance problem is a fairly simple one.
# We create random sample of data so that we get equal no. of observations in each class.
# The three sampling techniques that are used for Random sampling are :-
# (i) Oversampling  (ii) Undersampling   (iii) Both Oversampling & Undersampling


#  So in this step we will use all the three sampling techniques to build our model
# and then we will find out which one of them  give us the best sensitivity.

install.packages("ROSE")
library(ROSE)
library(e1071)

names(training)

            # ( i ) Oversampling
over<- ovun.sample(target ~ .,data = training,method = "over",seed = 777,N=36030)$data
table(over$target)
set.seed(345)
rfover<- randomForest(target ~ .,data = over,
                      Importance=TRUE,Proximity=TRUE) 

print(rfover)  

confusionMatrix((predict(rfover,newdata = testing)),testing$target,positive="1")
# Sensitivity = 0.71329
# Here the overall accuracy of the model is reduced but at the expense of accuracy,
# we get a significantly higher value of sensitivity than the previous model(fit2).

       # (ii) Undersampling

under<- ovun.sample(target ~ .,data = training,method = "under",seed = 888,N=3970)$data
table(under$target)

set.seed(678)
rfunder<- randomForest(target ~ .,data = under,
                       Importance=TRUE,Proximity=TRUE) 

print(rfunder)

confusionMatrix((predict(rfunder,newdata = testing)),testing$target,positive="1")
# Sensitivity = 0.81837
# We get a even better sensitivity this time. 


        # (iii) Both oversampling and undersampling

both<- ovun.sample(target ~ .,data = training,method = "both",seed = 999,N=20000)$data
table(both$target)

set.seed(890)
rfboth<- randomForest(target ~ .,data = both,
                      Importance=TRUE,Proximity=TRUE)

print(rfboth)

confusionMatrix((predict(rfboth,newdata = testing)),testing$target,positive="1")
# Sensitivity = 0.77439

# So based on the results obtained from these three models we can clearly see that model
# "rfunder" was able to correctly predict the "1"s more often than others, hence its the 
# best one among the three according to our bussiness need.

# So we will deploy this model on the new dataset(with no class label) 
# for the final prediction

#############++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++########
 
     # ROC(Receiver Operating Characteristics) curve

pred3<- predict(rfunder,newdata = testing,type = "prob")
print(head(pred3))
pred3<- pred3[ ,2] # for getting the probibility of positive class(in this case "1")
print(head(pred3))

library(ROCR)
pred3<- prediction(pred3,testing$target)

ROC<- performance(pred3,"tpr","fpr")
plot(ROC)
abline(a=0,b=1) # This is the benchmark

# We can add some details in the plot
plot(ROC,
     main="ROC Curve",
     ylab="Sensitivity",
     xlab="1-Specificity")
abline(a=0,b=1)

     
         # AUC Score

AUC<- performance(pred3,"auc")
AUC<- unlist(slot(AUC,"y.values"))
print(AUC)  # It prints the AUC score
AUC<- round(AUC,3)
# To print the value of AUC inside the Roc curve
legend("bottomright",0.4,AUC,title = "AUC",cex = 0.8) 


    #  Accuracy and Misclassification rate

pred4<- predict(rfunder,newdata = testing)
tab<-table(pred4,testing$target)


acc<- sum(diag(tab))/sum(tab)
misclass<- 1-acc
print(c(Accuracy=acc,Misclassification_Rate=misclass))

    #  Confusion Matrix
   
confusionMatrix(pred4,testing$target,positive = "1")


    #  F1 Score

precision<- posPredValue(pred4,testing$target,positive = "1")
Recall<- sensitivity(pred4,testing$target,positive = "1")

F1<- (2*precision*Recall)/(precision+Recall)
print(c(F1_score=F1))

#############++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++########

    #  Prediction on New dataset (newdata)

View(newdata)
str(newdata)
str(training)
x<- c("binary_var1","binary_var2","visited_page1","visited_page2","visited_page3",
      "visited_page4","visited_page5","visited_page6")

newdata[x]<- data.frame(apply(newdata[x],2,as.factor))

pred5<- predict(rfunder,newdata)
head(pred5)

newdata<- cbind(newdata,Predict_Final=pred5)
View(newdata)
table(newdata$Predict_Final)
names(newdata)

final_result<- select(newdata,c(unique_id,Predict_Final))
View(final_result)

write.csv(final_result,"Final Result.csv")
#######+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#############












































































