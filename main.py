from __future__ import division
from scipy.stats import pearsonr, spearmanr#, norm
# from scipy.integrate import quad
import numpy as np
import random as rando
import math
import matplotlib.pyplot as plt
import csv
import time
import pylab

# set these parameter values
N = 15                    # number of commercial banks initially
c = [0,0,0]               # expected payouts for each of the actions (vector of length D)
sig = [0.1,0.25,0.50]     # expected volatilities for payouts of each of the actions (vector of length D)
T = 1000                  # number of time steps in simulation
D = len(sig)              # number of possible actions/investment packages
w_ext = 100               # number of iterations for convergence

record_mat = []			  # matrix in which all the relevant records will be noted down

timebefore = time.time()  # time before simulation

sysriskdamage = 0
corrmatrix = np.zeros([N,N])

bankrupt_debts = 0
alpha = 0.2           						# coefficient of systemic risk 
epsilon = 0           						# initial exploration parameter for epsilon-greedy
theta = 0.1           						# softmax probability distribution parameter
eta = 0.05            						# learning rate
llambda1 = 1.0        						# social learning parameter for bailout cases
llambda2 = 0          						# discount parameter for epsilon
NW_init = 1           						# initial net worth of banks
plot_yes = True      						# whether or not to plot net worths and actions

Q = np.zeros([N, D])  						# initialize Q matrices
NW = np.zeros(N) + NW_init 					# net worth of banks
payouts = np.zeros([N,T]) 					# NxT matrix
bailout = np.zeros(N)     					# indicator for whether bank i was just bailed out
bailout_count = 0         					# counter for number of bailouts over the course of the simulation
distress_count = 0							# counter for number of distress cases over the course of the simulation
bhist = [[] for a in range(0,N)]	    	# record of bailout history for all banks
brupthist = [[] for a in range(0,N)]    	# record of bankruptcy history for all banks
bankrupt = np.zeros(N)    					# indicator for whether bank i is bankrupt and out of the simulation
CBM = np.zeros(N)        					# vector of amount spent by CB on each bank

probs_acts = np.zeros(D)  					# probabilities of actions for banks
acts = -1*np.ones([N,T])  					# actions taken by banks

# CB Objective Function
beta0 = -6
beta1 = 4
beta2 = -7
beta3 = 1
def CBObj(C,R_s,H_m):# Cost, Systemic Risk, Moral Hazard
	return 1/(1+math.exp(beta0+beta1*C+beta2*R_s+beta3*H_m))

# start time loop
for t in range(0,T):
	bailout = np.zeros(N)

	# choose actions and determine payouts
	for i in range(0,N):
		if rando.random() > epsilon: # softmax operator
			# find sum of exponential/softmax forms
			sum_exps = 0
			for j in range(0,D):
				sum_exps += math.exp(Q[i][j]/theta)

			# find probability of each action
			for j in range(0,D):
				probs_acts[j] = math.exp(Q[i][j]/theta)/sum_exps

			# select action
			act_determ = rando.random()
			cumul_sum = probs_acts[0]
			counter = 0
			while cumul_sum < act_determ:
				cumul_sum += probs_acts[counter+1]
				counter += 1

			acts[i][t] = counter
		else: # epsilon-greedy
			acts[i][t] = rando.randint(0,D-1)

		# determine payout
		payouts[i][t] = np.random.normal(c[int(acts[i][t])],sig[int(acts[i][t])])

		# determine new net worth
		NW[i] += payouts[i][t]

	# for all banks in bankruptcy
	for i in range(0,N):
		if NW[i] < 0: # CB decision on whether to bail out
			cost = 0.3*NW_init - NW[i] # cost of bailing out

			systemicrisk_hat = 0
			denom = 0
			for i2 in range(0,N):
				if i2 != i:
					if bankrupt[i2] == 0:
						corrmatrix[i][i2] = max(pearsonr(acts[i][0:t+1],acts[i2][0:t+1])[0],0) # pearson coefficient between actions
						if not math.isnan(corrmatrix[i][i2]):
							systemicrisk_hat += corrmatrix[i][i2]
							denom += 1
			systemicrisk_hat = systemicrisk_hat/max(denom,1) # systemic risk proxy

			moral_hazard_hat = bailout_count/max(distress_count,1) # moral hazard proxy

			p_bailout = CBObj(cost, systemicrisk_hat, moral_hazard_hat) # CB Objective

			if rando.random() < p_bailout: # if CB decided to bail out
				bailout[i] += 0.3*NW_init - NW[i] # amount of money awarded in bailout
				NW[i] = 0.3*NW_init # small amount of positive cash given back
				bailout_count += 1 # increase number of bailouts
				distress_count += 1 # increase number of distress cases
				bhist[i].append(t) # make note of this bailout
			else: # if CB decided not to bail out
				brupthist[i].append(t) # make note of this bankruptcy
				distress_count += 1 # increase number of distress cases

	# rewards
	for i in range(0,N):
		Q[i][int(acts[i][t])] = (1-eta)*Q[i][int(acts[i][t])] + eta*(payouts[i][t]+bailout[i])
		if NW[i] < 0 or bailout[i] != 0:
			for i2 in range(0,N):
				Q[i2][int(acts[i][t])] = (1-llambda1*eta)*Q[i2][int(acts[i][t])] + llambda1*eta*(payouts[i][t]+bailout[i])

	# get rid of banks that went bankrupt without bailout	
	for i in range(0,N):
		if NW[i] < 0:
			bankrupt_debts += 1-NW[i] # replace with new bank
			NW[i] = NW_init
			for i2 in range(0,N): # systemic risk impact on other banks
				if i2 != i:
					if not math.isnan(corrmatrix[i][i2]):
						NW[i2] += -alpha*corrmatrix[i][i2]*NW[i2]
						sysriskdamage += alpha*corrmatrix[i][i2]*NW[i2]
				else:
					corrmatrix[i][i2] = 0

	CBM = CBM + bailout # update amount of money spent by CB
	epsilon = epsilon*math.exp(llambda2*t) # epsilon decay

finexps = np.zeros(D) # final expectations of counts for each action
for i in range(0,N):	
	# find sum of exponential/softmax forms
	sum_exps = 0
	for j in range(0,D):
		sum_exps += math.exp(Q[i][j]/theta)

	# find probability of each action
	for j in range(0,D):
		finexps[j] += math.exp(Q[i][j]/theta)/sum_exps


unique, counts = np.unique(acts, return_counts=True)

investriskmetric = (counts[0]*sig[0] + counts[1]*sig[1] + counts[2]*sig[2])/(w_ext*N*T) # investment risk metric

# Final Q Convergence
Q_avg = np.sum(Q,axis=0)/N

timeafter = time.time() # time after simulation

print "**************************************************"
print "Execution took", (timeafter-timebefore)/3600, "hours."

# Plotting
if plot_yes == True:
	for wtr in range(0,N): # plot net worth over time for different banks
		plt.plot(range(0,T),payouts[wtr],'b-',range(0,T),np.zeros(T),'k--')
		plt.title("Payout for bank %s"%wtr)
		plt.xlabel("Time step")
		plt.ylabel("Payout")
		for d in range(0,len(bhist[wtr])):
			plt.plot((bhist[wtr][d],bhist[wtr][d]),(-2,2),'g--')
			plt.annotate('bailout', xy=(bhist[wtr][d], -2), xytext=(bhist[wtr][d], -3), 
				arrowprops=dict(facecolor='black', shrink=0.05),
				)
		for d in range(0,len(brupthist[wtr])):
			plt.plot((brupthist[wtr][d],brupthist[wtr][d]),(-2,2),'r--')
			plt.annotate('bankruptcy', xy=(brupthist[wtr][d], -2), xytext=(brupthist[wtr][d], -3), 
				arrowprops=dict(facecolor='black', shrink=0.05),
				)
		plt.show()

		plt.plot(range(0,T),acts[wtr],'b-') # plot actions over time for different banks
		plt.title("Action of bank %s"%wtr)
		plt.xlabel("Time step")
		plt.ylabel("Action")
		for d in range(0,len(bhist[wtr])):
			plt.plot((bhist[wtr][d],bhist[wtr][d]),(0,D),'g--')
			plt.annotate('bailout', xy=(bhist[wtr][d], 0), xytext=(bhist[wtr][d], -1), 
				arrowprops=dict(facecolor='black', shrink=0.05),
				)
		for d in range(0,len(brupthist[wtr])):
			plt.plot((brupthist[wtr][d],brupthist[wtr][d]),(0,D),'r--')
			plt.annotate('bankruptcy', xy=(brupthist[wtr][d], 0), xytext=(brupthist[wtr][d], -1), 
				arrowprops=dict(facecolor='black', shrink=0.05),
				)
		plt.show()
