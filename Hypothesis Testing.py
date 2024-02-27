# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:33:19 2024

@author: adity
"""

import pandas as pd
import numpy as np
import scipy 
from scipy import stats 

import statsmodels.stats.descriptivestats as sd
#From statsmodels.stats import Weightstats stests
import statsmodels.stats.weightstats
#1 sample sign test
#for given dataset check whether scores are equal or less than 80
#H0 = scores are either equal or less than 80
#H1 = scores are not equal and greater than 80
#whenever there is single sample and data is not normal
marks = pd.read_csv('C:/6-Hypothesis testing/hypothesis_datasets/Signtest.csv')
marks
#normal QQ plot
import pylab

#Checking whether the data the data is normal or not
stats.probplot(marks.Scores,dist='norm',plot=pylab)
#Data is not normal
#H0- data is normal
#H1- Data is not normal
stats.shapiro(marks.Scores)
#p_value is 0.024 > 0.005, p is high null fly
#Decision: data is not normal
########################################
#Let us check the distribution of the data
marks.Scores.describe()
#1 sample sign test
sd.sign_test(marks.Scores,mu0=marks.Scores.mean())
#p_value is 0.82>0.05 so p is high null fly
#Decision:
#H0 = scores are either equal or less than 80

#####################################################################
#####################1-Sample Z-test###############################
#Importing the data
import pandas as pd
fabric = pd.read_csv('C:/6-Hypothesis testing/hypothesis_datasets/Fabric_data.csv')

#Calculating the normality test
print(stats.shapiro(fabric))
#0.1460 > 0.05 h0 True
#calculating the mean
np.mean(fabric)

#z-test
#parameters in ztest, value is mean of data

ztest, pval = stests.ztest(fabric, x2 =None, value=150)

print(float(pval))

#p-value = 7.156e-06 < 0.05 so p low null go
#######################################################3
######################Mann- Whitney Test#######################################
#Vehicle with and without addictive
#H0: fuel additive does not impact the performance
#H1: fuel addictive does impact the performance
fuel = pd.read_csv("C:/6-Hypothesis testing/hypothesis_datasets/mann_whitney_additive.csv")
fuel

fuel.columns = "Without_additive","With_additive"

#Normality test
#H0: data is normal
print(stats.shapiro(fuel.Without_additive)) #p high null fly
print(stats.shapiro(fuel.With_additive)) #p low null go
#without_additive is normal
#With_additive is not normal
#When twp samples are not normal then mannwhitney test
#non-parameteric test case
#Mann-Whitney test
scipy.stats.mannwhitneyu(fuel.Without_additive, fuel.With_additive)

#p-value = 0.4457 > 0.05 p high null fly
#H0: fuel additive does not impact the performance

#############################  Paires T- test  ##################################

#when two features are normal then paired T-test
#A univariable test that tests for a significant difference between 2 relation
sup = pd.read_csv('C:/6-Hypothesis testing/hypothesis_datasets/paired2.csv')
sup.describe()
#H0: There is no significant difference between means of suppliers of A and B
#Ha: There is significant difference between means of suppliers of A and B
#Normality Test - # Shapiro test
stats.shapiro(sup.SupplierA)
#pvalue= 0.8961  > 0.005 hence data is normal
stats.shapiro(sup.SupplierB)
#pvalue=0.89619 > 0.005 hence data is normal


import seaborn as sns
sns.boxplot(data=sup)

#
#

ttest, pval = stats.ttest_rel(sup['SupplierA'],sup['SupplierB'])
print(pval)

#p-value = 0< 0.005 so p low null go

######################### 2 sample T test ############################3
#Load the data
prom =pd.read_excel('C:/6-Hypothesis testing/hypothesis_datasets/Promotion.xlsx')
prom
#H0: InterestRateWaiver < StandardPromotion
#H1: InterestRateWaiver > StandardPromotion
prom.columns = "InterestRateWaiver","StandardPromotion"

#Normality test
stats.shapiro(prom.InterestRateWaiver) #Shapiro TEst
#pvalue=0.2245 > 0.005
print(stats.shapiro(prom.StandardPromotion))
#pvalue=0.1915 > 0.005


#data is normal

#Variance test
help(scipy.stats.levene)
#H0: Both columns have equal variance
#H1: Both columns have not equal variance
scipy.stats.levene(prom.InterestRateWaiver, prom.StandardPromotion)
#p-value =  0.287 > 0.05 so p high null fly => Equal variance

#2 sample T test
scipy.stats.ttest_ind(prom.InterestRateWaiver, prom.StandardPromotion)
help(scipy.stats.ttest_ind)
#H0: equal means
#H1: unequal means
#p-value = pvalue=0.0242 < 0.05 so p low null high

############################## One-way ANOVA ################################333
'''
Problem Statement: A marketing organization outsources their back-office operations 
to three different suppliers. the contracts are up for renewal and the SMO wants 
to determine whether  they should review contracts with all supliers or any 
specific supplier. CMO want to renew the contract of suppkier with the least 
transaction time. CMO will renew  all contracts if the performance of all 
suppliers is similar.
 '''
 
con_renewal = pd.read_excel("C:/6-Hypothesis testing/hypothesis_datasets/ContractRenewal_Data(unstacked).xlsx")
con_renewal
con_renewal.columns = "SupplierA","SupplierB","SupplierC"
#H0: All the 3 supplier have equal mean transaction time
#H1: All the supplier have not equal mean transaction time
#Normality test
stats.shapiro(con_renewal.SupplierA) #Shapiro Test
#pvalue=0.8961 > 0.005, SupplierA is noermal
stats.shapiro(con_renewal.SupplierB) #shapiro test
#pvalue=0.6483 > 0.005, SupplierB is noermal
stats.shapiro(con_renewal.SupplierC) #Shapiro Test
#pvalue=0.5719 > 0.005, SupplierC is normal
#Variance Test
help(scipy.stats.levene)
#All 3 suppliers are being checked for variance
scipy.stats.levene(con_renewal.SupplierA,con_renewal.SupplierB, con_renewal.SupplierC)
#The Levene test tests the null hypothesis
#that all input samples are form populations with equal variances
#pvalue=0.0777>0.005, p is high null fly
#H0= All inputs samples are from populations with equal variances

#One-Way Anova

F, p = stats.f_oneway(con_renewal.SupplierA, con_renewal.SupplierB, con_renewal.SupplierC)

#p value
p 
#0.10373 > 0.05 P high null fly
#All the 3 suppliers have equal mean transaction time


####################### 2 - Proportion test ##################################
'''
Problem Statement:Hohnnie Talkers soft drinks devision sales manager has been planning
to launch a new sales incentive program for their sales executives. The sales 
executives felt that adults(>40 yrs) wont buy, children will & hence requested sales 
manager not to launch the program. Analyze the data determine there is evidence at
5% significance level to support the hypothesis.
'''
#H0: Proportion A= Proportion B
#H1: Proportion A Not = Proportion B
import numpy as np
two_prop_test = pd.read_excel("C:/6-Hypothesis testing/hypothesis_datasets/JohnyTalkers.xlsx")
from statsmodels.stats.proportion import proportions_ztest
tab1 = two_prop_test.Person.value_counts()
tab1
'''
Children    740
Adults      480
'''
tab2 = two_prop_test.Drinks.value_counts()
tab2
'''
Did Not Purchase    1010
Purchased            210
'''
#Crosstable
pd.crosstab(two_prop_test.Person, two_prop_test.Drinks)
'''
Drinks    Did Not Purchase  Purchased
Person                               
Adults                 422         58
Children  
'''
count=np.array([58, 152])
nobs = np.array([480, 740])

stats, pval = proportions_ztest(count, nobs, alternative = "two-sided")
print(pval) #Pvalue 0.000


stats, pval = proportions_ztest(count, nobs, alternative = "larger")
print(pval) #Pvalue 0.999

####################### Chi-Square Test #############################

Bahaman = pd.read_excel("C:/6-Hypothesis testing/hypothesis_datasets/Bahaman.xlsx")
Bahaman

count = pd.crosstab(Bahaman["Defective"], Bahaman["Country"])
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic','p-value'],
              [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
'''
you use chi2_contingency when you want to test
whether two (or more) groups have the same distribution.
'''

#H0: Null Hypothesis: the two groups have no significant difference
#since p = 0.63 > 0.05 Hence H0 is true.
###########################################################################







