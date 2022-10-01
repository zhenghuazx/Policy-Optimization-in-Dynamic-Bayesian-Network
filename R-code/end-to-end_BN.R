library(readr)
library(sgd)
library(deSolve)
library(MASS)
library(readxl)

set.seed(10) # 5 batch, seed=100/70,30,10
###################################################
################# data ###########################
X170517F3 <- read_excel("~/Research/PHD/project/BN-MDP/BN/170517F3.xlsx")
X170706F3 <- read_excel("~/Research/PHD/project/BN-MDP/BN/170706F3.xlsx")
X170517F4 <- read_excel("~/Research/PHD/project/BN-MDP/BN/170517F4.xlsx")
X170601F3 <- read_excel("~/Research/PHD/project/BN-MDP/BN/170601F3.xlsx")
X170803F3 <- read_excel("~/Research/PHD/project/BN-MDP/BN/170803F3.xlsx")
X170721F3 <- read_excel("~/Research/PHD/project/BN-MDP/BN/170721F3.xlsx")
X171003F3 <- read_excel("~/Research/PHD/project/BN-MDP/BN/171003F3.xlsx")
X170506F3 <- read_excel("~/Research/PHD/project/BN-MDP/BN/170506F3.xlsx")
# 
# X170506F3 <- read_csv("170506F3.csv")
# 
# X170517F3 <- read_csv("170517F3.csv")
# 
# X170517F4 <- read_csv("170517F4.csv")
# 
# X170601F3 <- read_csv("170601F3.csv")
# 
# #X170613F3 <- read_csv("170613F3.csv")
# 
# X170706F3 <- read_csv("170706F3.csv")
# 
# X170721F3 <- read_csv("170721F3.csv")
# 
# X170803F3 <- read_csv("170803F3.csv")
# 
# #X170817F3 <- read_csv("170817F3.csv")
# 
# X171003F3 <- read_csv("171003F3.csv")


nodes.name <- names(X170506F3)
# node.at.each.time = nodes.name[c(8,17,7,2,10)] #
#node.at.each.time = nodes.name[c(11,13,2,17,8,7)]

# remove pO2
# node.at.each.time = nodes.name[c(8,17,2,10)]

# CPPs: agitation (8), feed rate (17), pO2 (7), residual Oil (5); initial OD
# no initial variation: agitation, pO2, residual Oil, 
# CQAs: total biomass, CA (16,15)
node.at.each.time = nodes.name[c(25,7,21,10,20,5,27,22)] # if modify --> need to modify beta, CPPs_index
# node.at.each.time = nodes.name[c(21,10,20,5,23,24,22,27)] # if modify --> need to modify beta, CPPs_index

node.at.each.time

ttname = X170506F3$`Ferm. Time (h)` # as.numeric(temp_t$'Ferm..Time..h.')

# CQAs: (residual oil (g/L),yield,productivity,OD600)
# CPPs (FeedRate,Agitation Rate)
# well-controlled CPPs: pH, O2

# # replace initial Feed Rate with initial Oil (for implementation)
# X170506F3$'FeedRate'[1] = X170506F3$'Residual oil (g/L)'[1]
# X170517F3$'FeedRate'[1] = X170517F3$'Residual oil (g/L)'[1]
# X170517F4$'FeedRate'[1] = X170517F4$'Residual oil (g/L)'[1]
# X170601F3$'FeedRate'[1] = X170601F3$'Residual oil (g/L)'[1]
# X170613F3$'FeedRate'[1] = X170613F3$'Residual oil (g/L)'[1]
# X170706F3$'FeedRate'[1] = X170706F3$'Residual oil (g/L)'[1]
# X170721F3$'FeedRate'[1] = X170721F3$'Residual oil (g/L)'[1]
# X170803F3$'FeedRate'[1] = X170803F3$'Residual oil (g/L)'[1]
# X170817F3$'FeedRate'[1] = X170817F3$'Residual oil (g/L)'[1]
# X171003F3$'FeedRate'[1] = X171003F3$'Residual oil (g/L)'[1]


# # replace initial Total Biomass with initial OD600 (for implementation)
# X170506F3$'Biomass total (g)'[1] = X170506F3$'OD600'[1]
# X170517F3$'Biomass total (g)'[1] = X170517F3$'OD600'[1]
# X170517F4$'Biomass total (g)'[1] = X170517F4$'OD600'[1]
# X170601F3$'Biomass total (g)'[1] = X170601F3$'OD600'[1]
# #X170613F3$'Biomass total (g)'[1] = X170613F3$'OD600'[1]
# X170706F3$'Biomass total (g)'[1] = X170706F3$'OD600'[1]
# X170721F3$'Biomass total (g)'[1] = X170721F3$'OD600'[1]
# X170803F3$'Biomass total (g)'[1] = X170803F3$'OD600'[1]
# #X170817F3$'Biomass total (g)'[1] = X170817F3$'OD600'[1]
# X171003F3$'Biomass total (g)'[1] = X171003F3$'OD600'[1]

X170506F3$'Work volume (mL)' = X170506F3$'Work volume (mL)'/1000
X170517F3$'Work volume (mL)' = X170517F3$'Work volume (mL)'/1000
X170517F4$'Work volume (mL)' = X170517F4$'Work volume (mL)'/1000
X170601F3$'Work volume (mL)' = X170601F3$'Work volume (mL)'/1000
#X170613F3$'Work volume (mL)' = X170613F3$'Work volume (mL)'/1000
X170706F3$'Work volume (mL)' = X170706F3$'Work volume (mL)'/1000
X170721F3$'Work volume (mL)' = X170721F3$'Work volume (mL)'/1000
X170803F3$'Work volume (mL)' = X170803F3$'Work volume (mL)'/1000
#X170817F3$'Work volume (mL)' = X170817F3$'Work volume (mL)'/1000
X171003F3$'Work volume (mL)' = X171003F3$'Work volume (mL)'/1000


X170506F3$'Adjusted Volume (mL)' = X170506F3$'Adjusted Volume (mL)'/1000
X170517F3$'Adjusted Volume (mL)' = X170517F3$'Adjusted Volume (mL)'/1000
X170517F4$'Adjusted Volume (mL)' = X170517F4$'Adjusted Volume (mL)'/1000
X170601F3$'Adjusted Volume (mL)' = X170601F3$'Adjusted Volume (mL)'/1000
#X170613F3$'Adjusted Volume (mL)' = X170613F3$'Adjusted Volume (mL)'/1000
X170706F3$'Adjusted Volume (mL)' = X170706F3$'Adjusted Volume (mL)'/1000
X170721F3$'Adjusted Volume (mL)' = X170721F3$'Adjusted Volume (mL)'/1000
X170803F3$'Adjusted Volume (mL)' = X170803F3$'Adjusted Volume (mL)'/1000
#X170817F3$'Adjusted Volume (mL)' = X170817F3$'Adjusted Volume (mL)'/1000
X171003F3$'Adjusted Volume (mL)' = X171003F3$'Adjusted Volume (mL)'/1000

# replace initial Total Biomass with initial OD600 (for implementation)
X170506F3$'F_S' = X170506F3$'F_S'/1000
X170517F3$'F_S' = X170517F3$'F_S'/1000
X170517F4$'F_S' = X170517F4$'F_S'/1000
X170601F3$'F_S' = X170601F3$'F_S'/1000
#X170613F3$'F_S' = X170613F3$'F_S'/1000
X170706F3$'F_S' = X170706F3$'F_S'/1000
X170721F3$'F_S' = X170721F3$'F_S'/1000
X170803F3$'F_S' = X170803F3$'F_S'/1000
#X170817F3$'F_S' = X170817F3$'F_S'/1000
X171003F3$'F_S' = X171003F3$'F_S'/1000

X170506F3$'adjust F_S' = X170506F3$`adjust F_S`/1000
X170517F3$'adjust F_S' = X170517F3$`adjust F_S`/1000
X170517F4$'adjust F_S' = X170517F4$`adjust F_S`/1000
X170601F3$'adjust F_S' = X170601F3$`adjust F_S`/1000
#X170613F3$'F_S' = X170613F3$'F_S'/1000
X170706F3$'adjust F_S' = X170706F3$`adjust F_S`/1000
X170721F3$'adjust F_S' = X170721F3$`adjust F_S`/1000
X170803F3$'adjust F_S' = X170803F3$`adjust F_S`/1000
#X170817F3$'adjust F_S' = X170817F3$'F_S'/1000
X171003F3$'adjust F_S' = X171003F3$`adjust F_S`/1000

# replace initial Total Biomass with initial OD600 (for implementation)
X170506F3$'F_B' = X170506F3$'F_B'/1000
X170517F3$'F_B' = X170517F3$'F_B'/1000
X170517F4$'F_B' = X170517F4$'F_B'/1000
X170601F3$'F_B' = X170601F3$'F_B'/1000
#X170613F3$'F_B' = X170613F3$'F_B'/1000
X170706F3$'F_B' = X170706F3$'F_B'/1000
X170721F3$'F_B' = X170721F3$'F_B'/1000
X170803F3$'F_B' = X170803F3$'F_B'/1000
#X170817F3$'F_B' = X170817F3$'F_B'/1000
X171003F3$'F_B' = X171003F3$'F_B'/1000

X170506F3$`adjust F_B` = X170506F3$`adjust F_B`/1000
X170517F3$`adjust F_B` = X170517F3$`adjust F_B`/1000
X170517F4$`adjust F_B` = X170517F4$`adjust F_B`/1000
# X170601F3$`adjust F_B` = X170601F3$`adjust F_B`/1000
#X170613F3$'F_B' = X170613F3$'F_B'/1000

X170706F3$`adjust F_B` = X170706F3$`adjust F_B`/1000
X170721F3$`adjust F_B` = X170721F3$`adjust F_B`/1000
X170803F3$`adjust F_B` = X170803F3$`adjust F_B`/1000
#X170817F3$'F_B' = X170817F3$'F_B'/1000
# X171003F3$`adjust F_B` = X171003F3$`adjust F_B`/1000
F_B = cbind(X170506F3$`adjust F_B`, X170517F3$`adjust F_B`, X170517F4$`adjust F_B`, X170706F3$`adjust F_B`, X170721F3$`adjust F_B`, X170803F3$`adjust F_B`)
F_B.profile.input = apply(F_B, 1, mean)

X170506F3 = X170506F3[node.at.each.time]
X170517F3 = X170517F3[node.at.each.time]
X170517F4 = X170517F4[node.at.each.time]
X170601F3 = X170601F3[node.at.each.time]
# X170613F3 = X170613F3[node.at.each.time]
X170706F3 = X170706F3[node.at.each.time]
X170721F3 = X170721F3[node.at.each.time]
X170803F3 = X170803F3[node.at.each.time]
# X170817F3 = X170817F3[node.at.each.time]
X171003F3 = X171003F3[node.at.each.time]




data.wide<-data.frame()
data.wide <- rbind(data.wide, c(t(as.matrix(X170506F3))))
data.wide <- rbind(data.wide, c(t(as.matrix(X170517F3))))
data.wide <- rbind(data.wide, c(t(as.matrix(X170517F4))))
data.wide <- rbind(data.wide, c(t(as.matrix(X170601F3))))
# data.wide <- rbind(data.wide, c(t(as.matrix(X170613F3))))
data.wide <- rbind(data.wide, c(t(as.matrix(X170706F3))))
data.wide <- rbind(data.wide, c(t(as.matrix(X170721F3))))
data.wide <- rbind(data.wide, c(t(as.matrix(X170803F3))))
# data.wide <- rbind(data.wide, c(t(as.matrix(X170817F3))))
data.wide <- rbind(data.wide, c(t(as.matrix(X171003F3))))

n_factor = dim(X170506F3)[2]
n_time   = dim(X170506F3)[1]
N = n_factor*n_time
dat <- data.wide[,1:N]

R = dim(data.wide)[1]

X_f = matrix(0,R,n_time)
C = matrix(0,R,n_time)
L = matrix(0,R,n_time)
S = matrix(0,R,n_time)
F_S = matrix(0,R,n_time)
O = matrix(0,R,n_time)
# F_B = matrix(0,R,n_time)
V = matrix(0,R,n_time)
N = matrix(0,R,n_time)

for (t in 1:n_time){
  F_S[,t] = dat[,(t-1)*n_factor+1]
  O[,t] = dat[,(t-1)*n_factor+2]
  X_f[,t] = dat[,(t-1)*n_factor+3]
  C[,t] = dat[,(t-1)*n_factor+4]
  L[,t] = dat[,(t-1)*n_factor+5]
  S[,t] = dat[,(t-1)*n_factor+6]
  N[,t] = dat[,(t-1)*n_factor+7]
  # F_B[,t] = dat[,(t-1)*n_factor+6]
  V[,t] = dat[,(t-1)*n_factor+8]
  
}

#####
product = mean(C[,14])
impurity = mean(X_f[,14])*0.01 + mean(L[,14])+ mean(S[,14]) + 0.7

S.1 = 48
S.2 = 64
product.1 <- purification(S.1, 56.27, 42.0) * product
impurity.1 <- purification(S.1, 53.72, 5.23) * impurity
product.2 <- (1-purification(S.2, 56.27, 42.0))* product.1
impurity.2 <- (1-purification(S.2, 53.72, 5.23)) * impurity.1

plot.mean.feedrate=F


# legend("topright", col = 1:6, lty = 1:6, legend = c('X_f', "C", "L", "S", "N", "V"))
# plot(out[,"P"], out[,"C"], type = "l", lwd = 2, xlab = "producer",
#      ylab = "consumer")


######################################
############## simulation ############
######################################
# simulate data

# alpha_L:      0.12780386 +/- 0.00259961 (2.03%) (init = 0.127275)
# c_max:        123.145600 +/- 1.13222228 (0.92%) (init = 122.9017)
# K_iN:         0.12441510 +/- 0.02568847 (20.65%) (init = 0.122941)
# K_iS:         613.128614 +/- 2.05052747 (0.33%) (init = 612.1781)
# K_iX:         59.9736322 +/- 6.2431e-04 (0.00%) (init = 59.97377)
# K_N:          0.02558868 +/- 0.00493052 (19.27%) (init = 0.02500201)
# K_O:          0.33178243 +/- 0.04832645 (14.57%) (init = 0.3208532)
# K_S:          0.08860285 +/- 0.02351225 (26.54%) (init = 0.0830429)
# K_SL:         0.01807056 +/- 0.00426967 (23.63%) (init = 0.01265744)
# m_s:          0.04335974 +/- 0.00534558 (12.33%) (init = 0.02652332)
# r_L:          0.23977514 +/- 0.03642472 (15.19%) (init = 0.3091781)
# V_evap:       4.2028e-07 +/- 9.4851e-05 (22568.81%) (init = 0.00012)
# Y_cs:         0.86657600 +/- 0.02311103 (2.67%) (init = 0.8902572)
# Y_ls:         0.59066313 +/- 0.00882945 (1.49%) (init = 0.5874429)
# Y_xn:         9.96989169 +/- 0.07485910 (0.75%) (init = 10)
# Y_xs:         0.27520911 +/- 0.00500488 (1.82%) (init = 0.2785559)
# beta_LC_max:  0.16157052 +/- 0.01291308 (7.99%) (init = 0.1955192)
# mu_max:       0.32109468 +/- 0.00576535 (1.80%) (init = 0.3304463)

parameters <- c(
  alpha_L=0.127275,
  c_max=130.901733,
  K_iN=0.12294103,
  K_iS=612.178130,
  K_iX=59.9737695,
  K_N=0.02000201,
  K_O=0.33085322,
  K_S=0.0430429,
  K_SL=0.02165744,
  m_s=0.02252332,
  r_L=0.47917813,
  V_evap=2.6*1e-03,
  Y_cs=0.682572,
  Y_ls=0.3574429,
  Y_xn=10,
  Y_xs=0.2385559,
  beta_LC_max=0.14255192,
  mu_max=0.3844627,
  S_F=917)
out <- ode(y = initial.state, times = tname, func = bioreactor, parms = parameters, method = 'vode')
plot(out)
out.mean.trajector = out

R <- 500  # number of batch data
tname = seq(0,140,5)
tname1 <- seq(0,20,5)
tname2 <- seq(22,36,2)
tname3 <- seq(40,140,5)
tname <- c(tname1,tname2,tname3)
initial.state <- c(X_f = 0.05, C=0, L=0, S = 30, N=5, V=0.6)
feed_profile_init <- apply(F_S, 2, mean)
oxygen_profile <- apply(O, 2, mean)
oxygen.profile.input <- oxygen_profile
# for (i in c(1,2)) {
#   feed_profile_init[i] <- feed_profile_init[i]
# }
# feed_profile_init[3] <- feed_profile_init[3]
# feed_profile_init[4] <- feed_profile_init[4]/2.7
# feed_profile_init[5] <- feed_profile_init[5] * 1.1
# feed_profile_init[6] <- feed_profile_init[6] *1
# for (i in c(7:10)) {
#   feed_profile_init[i] <- feed_profile_init[i] * 1.3
# }
# for (i in c(11:14)) {
#   feed_profile_init[i] <- feed_profile_init[i] * 1.5
# }
# feed_profile_init = feed_profile_init_temp
tname = seq(0,140,4)
# feed.profile.input = approx(ttname,feed.profile.input, tname)$y
# oxygen.profile.input = approx(ttname,oxygen.profile.input, tname)$y


# feed_profile_init_interpolation = approx(ttname, feed_profile_init, tname,'linear')$y
# for (i in 1:4) {
#   feed_profile_init_interpolation[i] <- feed_profile_init_interpolation[i] /i
# }
# feed_profile_init_interpolation[4] <- feed_profile_init_interpolation[4] * 2
# feed_profile_init_interpolation[5] <- feed_profile_init_interpolation[5] * 2.1
# feed_profile_init_interpolation[6] <- feed_profile_init_interpolation[6] * 2.2
# feed_profile_init_interpolation[7] <- feed_profile_init_interpolation[7] * 4.3
# feed_profile_init_interpolation[8] <- feed_profile_init_interpolation[8] * 12.5
# feed_profile_init_interpolation[9] <- feed_profile_init_interpolation[9] * 9.5
# feed_profile_init_interpolation[10] <- feed_profile_init_interpolation[10] * 4.6
# feed_profile_init_interpolation[11] <- feed_profile_init_interpolation[11] * 1.5
# feed_profile_init_interpolation[12] <- feed_profile_init_interpolation[12] * 1.2
# feed_profile_init_interpolation[13] <- feed_profile_init_interpolation[13] * 1.2

# feed_profile_init[3] <- feed_profile_init[3] * 2
# for (i in 14:26) {
#   feed_profile_init_interpolation[i] <- feed_profile_init_interpolation[i] *0.85
# }
# 
# for (i in 27:34) {
#   feed_profile_init_interpolation[i] <- feed_profile_init_interpolation[i]* (1+(i-27)/1.5)
# }

# oxygen_profile_interpolation = approx(ttname, oxygen_profile,tname, 'linear')$y
feed.e <- new.env()
feed.e$feed <- rep(0,36)
feed.e$F_B <- rep(0,36)
bioreactor <- function(t, state, parameters) {
  with(as.list(c(state, parameters)),{
    decision.index = 0
    for (i in 1:length(ttname)){
      if (ttname[i] - t >0) {
        decision.index = i - 1
        break
      }
    }
    # decision.index <- which.min(abs(t - ttname+0.5))
    if (plot.mean.feedrate) {
      F_S <- feed_profile_init[decision.index]
    } else {
      F_S <- feed.profile.input[decision.index]
    }
    
    # print(F_S)
    # print(floor(t/10.01+1))
    # rate of change
    # q.s =  q.s.max*S/(S+Ks)
    # mu <- (q.s-q.m)*Y.em
    # 
    # dX <- (-Feed/V+mu)*X
    # dS <- Feed/V*(Si-S)-q.s*X
    
    # S_F = 917
    
    # print(t)
    # F_S = feed_rate(t, batch_idx)
    O = oxygen.profile.input[decision.index]
    
    beta_LC = K_iN / (K_iN + N) * S / (S + K_S) * K_iS / (K_iS + S) * O / (K_O+O)* K_iX / (K_iX + X_f) * (1 - C/c_max) * beta_LC_max
    q_C = 2 * (1 - r_L) * beta_LC
    beta_L = r_L * beta_LC - K_SL * L / (L + X_f) * O / (O + K_O)
    mu = mu_max * S / (S + K_S) * K_iS / (K_iS + S) * N / (K_N + N) * O / (K_O + O) / (1 + X_f / K_iX)
    q_S = 1 / Y_xs * mu + O / (O + K_O) * S / (K_S + S) * m_s + 1 / Y_cs * q_C + 1 / Y_ls * beta_L
    F_B =   V /1000 * (7.14 / Y_xn * mu * X_f + 1.59 * q_C * X_f)
    D = (F_B + F_S) / V
    
    
    dX_f = mu * X_f - (D - V_evap / V) * X_f
    dC = q_C * X_f - (D - V_evap / V) * C
    dL = (alpha_L * mu + beta_L) * X_f - (D - V_evap / V) * L
    dS = - (q_S * X_f - F_S / V * S_F + (D - V_evap / V) * S)
    dN = - (1 / Y_xn * mu * X_f + (D - V_evap / V) * N)
    dV = F_B + F_S - V_evap
    # if (t<30) {
    #   print(paste("t:", t,"q_S:", q_S,", D:", D, "F_B:",F_B, "F_S:",F_S,"beta_L:", beta_L, "dS:", dS, "term3: ", (D - V_evap / V) * S, "term2:", F_S / V * S_F, "term1:", q_S * X_f ))
    #   print(paste("t:", t,"F_S:", F_S))
    # }
    list(c( dX_f, dC, dL, dS, dN, dV)) # return the rate of change
  })
}


# parameters['F_S'] = 0.002
# parameters['O'] = 70 + rnorm(1, 0, 5)
# feed_profile <- feed_profile_init+ rlnorm(11, 0, 1)
# parameters['Si'] = 780 + rnorm(1, 0, 40)
# initial.state['S'] = 40 + rnorm(1, 0,2)
out <- ode(y = initial.state, times = tname, func = bioreactor, parms = parameters, method = 'vode')
plot(out)
R=400
R_max = 500
n_time = 36 # 36
n_time_purif = 3
N = 8 * 36 + 6 * n_time_purif# 8 * 36


dat.matrix <- matrix(NA,nrow=R_max, ncol=N)
add.noise <- function(x, b) {
  return(max(0,rnorm(1,x,max(0,x)/b)))
} 

purification <- function(s, a, b) {
  return(1/(1+(s/a)^b))
} 


# hybrid generating 
for (k in 1:R_max) {
  time.step <- tname
  feed.profile.input <- c()
  # exploration vs exploitation
  if (runif(1,0,1) > 0.3) {
    for (i in 1:length(ttname)) {
      feed.profile.input[i] <- feed_profile_init[i] + rnorm(1,0,0.00002+ feed_profile_init[i])
      feed.profile.input[i] <- max(0, feed.profile.input[i])
    }
  } else {
    for (i in 1:length(ttname)) {
      feed.profile.input[i] <-  sample(feed_profile_init, 1) + rnorm(1,0, feed_profile_init[i] / 10)
      feed.profile.input[i] <- max(0, feed.profile.input[i])
    }
  }
  
  oxygen.profile.input <- c()
  for (i in 1:length(ttname)) {
    oxygen.profile.input[i] <- oxygen_profile[i] + rnorm(1,0,oxygen_profile[i]/10)
    oxygen.profile.input[i] <- max(0,oxygen.profile.input[i])
    oxygen.profile.input[i] <- min(100,oxygen.profile.input[i])
  }
  initial_state <- initial.state
  for (i in 1:6) {
    initial_state[i] <- initial.state[i] + abs(rnorm(1,0,initial.state[i]/10))
  }
  
  out <- ode(y = initial_state, times = time.step, func = bioreactor, parms = parameters)
  out <- out[,2:7]
  # out <- out[,1:5] * out[,6]
  linear.feed.profile.input = approx(ttname,feed.profile.input, tname)$y
  linear.oxygen.profile.input = approx(ttname,oxygen.profile.input, tname)$y
  biorector.process<- cbind(linear.feed.profile.input, linear.oxygen.profile.input, out)
  output <- c()
  for (i in 1:n_time) {
    output <- c(output, biorector.process[i,])
    output = apply(as.array(output), 1, add.noise, 25)
  }
  
  # ---------------------
  # purification after C1
  # ---------------------
  impurity.0 = (sum(biorector.process[36,c(5,6,7)]) + biorector.process[36,3] * 0.01) * biorector.process[36,8]
  product.0 = biorector.process[36,4] * biorector.process[36,8]
  S_0 <- runif(1,0.35,0.55) * 100
  product.1 <- purification(S_0, 56.27, 42.0) * product.0
  impurity.1 <- purification(S_0, 53.72, 5.23) * impurity.0
  S_1 <- runif(1,0.55,0.7) * 100
  product.2 <- (1-purification(S_1, 56.27, 42.0)) * product.1
  impurity.2 <- (1-purification(S_1, 53.72, 5.23)) * impurity.1
  
  
  dat.matrix[k,] <- c(output, log(S_0), log(product.0), log(impurity.0),0,0,0, log(S_1), log(product.1), log(impurity.1),0,0,0,runif(1,0,0.01),log(product.2), log(impurity.2),0,0,0)
  
}

n_factor = 8


dat.matrix = dat.matrix[apply(is.nan(dat.matrix),1,sum) == 0,]
# dat.matrix = dat.matrix[apply(is.na(dat.matrix),1,sum) == 0,]

dat.matrix = dat.matrix[1:8,]
temp.dat.matrix = dat.matrix
### choose the variable in BN
state4action1 = c(seq(1,n_factor * n_time,n_factor), seq(3,n_factor * n_time,n_factor) , seq(4,n_factor * n_time,n_factor) , seq(6,n_factor * n_time,n_factor) , seq(7,n_factor * n_time,n_factor), seq(8,n_factor * n_time,n_factor), seq(289,306,1))
state4action1 = sort(state4action1)

n_factor = 6
n_time   = 36 # 36
N = n_factor*(n_time + n_time_purif)
### run BN
# dat.matrix= dat.matrix[,c(1,   3,   4,   6,   7,   8,   9,  11,  12,  14,  15,  16)]
dat.matrix = dat.matrix[,state4action1]
# dat.matrix= rbind(dat.matrix1,dat.matrix2,dat.matrix3)

# dat.matrix = dat.matrix1[201:300,]
mu <- apply(dat.matrix, 2, mean)
sd <- apply(dat.matrix, 2, sd)
for (i in c(3)) {
  sd[i] <-abs(rnorm(1,0,(mu[i] + mu[i+n_factor])/ 12+0.0001))
}
sd[which(sd==0)] <- runif(length(which(sd==0)), 0, 0.001)
# normalize data
# dat <- t(t(dat) - mu)
# dat <- t(t(dat)/sd)
# dat[which(is.nan(dat))] <- 0



# add Impellers to X3_t0
# dat[,3] <- c(rep(0,7), rep(1,3))

# only use Rushton impellers data
# dat <- dat[1:7,]
dat <- dat.matrix[,1:N]
dat <- t(t(dat) - mu)
dat <- t(t(dat)/sd)

# v2 <- abs(rnorm(N,0,0.01))
v2 <- apply(dat.matrix,2,sd)+abs(rnorm(N,0,0.001))

# structure of BN:
beta <- matrix(0, N, N)
gamma <- diag(1, N)
TF_mx <- diag(T, N)


# following time periods 5 state and 2 action
# for (i in 1:(n_time-1)){
#   beta[(i-1)*n_factor+1,n_factor*(i+1)] <- 1
#   beta[(i-1)*n_factor+2,(n_factor*(i+1)-4):((n_factor*i)+7)] <- 1
#   beta[(i-1)*n_factor+3,(n_factor*(i+1)-4):((n_factor*i)+7)] <- 1
#   beta[(i-1)*n_factor+4,(n_factor*(i+1)-3):((n_factor*i)+7)] <- 1
#   beta[(i-1)*n_factor+5,(n_factor*(i+1)-2):((n_factor*i)+7)] <- 1
#   beta[(i-1)*n_factor+6,(n_factor*(i+1)-4):((n_factor*i)+7)] <- 1
#   beta[(i-1)*n_factor+7,(n_factor*(i+1)-4):((n_factor*i)+7)] <- 1
#   # beta[(n_factor*i+1:n_factor), (n_factor*(i+1)+4:n_factor)] <- 1
#   # beta[(n_factor*i+n_factor), (n_factor*(i+1)+4)] <- 0
#   # beta[(8*i+1:5), (8*(i+1)+5)] <- 1
#   # beta[(8*i+1:6), (8*(i+1)+6)] <- 1
#   # beta[(8*i+1:7), (8*(i+1)+7)] <- 1
#   # beta[(8*i+c(1:6,8)), (8*(i+1)+8)] <- 1
#   
#   # TF_mx[(n_factor*i+4:n_factor), (n_factor*i+4:n_factor)] <- T
# }

# following time periods 4 state: X_f, C, S, N, V and 1 action

for (i in 1:(n_time-1)){
  beta[(i-1)*n_factor+1,c(n_factor*(i+1)-2,n_factor*(i+1))] <- 1
  # beta[(i-1)*n_factor+2,(n_factor*(i+1)-4):((n_factor*i)+7)] <- 1
  beta[(i-1)*n_factor+2,(n_factor*(i+1)-4):((n_factor*i)+6)] <- 1
  beta[(i-1)*n_factor+3,(n_factor*(i+1)-3):((n_factor*i)+5)] <- 1
  # beta[(i-1)*n_factor+5,(n_factor*(i+1)-2):((n_factor*i)+7)] <- 1
  # beta[(i-1)*n_factor+4,(n_factor*(i+1)-2):((n_factor*i)+5)] <- 1
  beta[(i-1)*n_factor+4,(n_factor*(i+1)-4):((n_factor*i)+5)] <- 1
  beta[(i-1)*n_factor+5,(n_factor*(i+1)-4):((n_factor*i)+5)] <- 1
  beta[(i-1)*n_factor+6,(n_factor*(i+1)-4):((n_factor*i)+6)] <- 1
  # beta[(n_factor*i+1:n_factor), (n_factor*(i+1)+4:n_factor)] <- 1
  # beta[(n_factor*i+n_factor), (n_factor*(i+1)+4)] <- 0
  # beta[(8*i+1:5), (8*(i+1)+5)] <- 1
  # beta[(8*i+1:6), (8*(i+1)+6)] <- 1
  # beta[(8*i+1:7), (8*(i+1)+7)] <- 1
  # beta[(8*i+c(1:6,8)), (8*(i+1)+8)] <- 1
  # TF_mx[(n_factor*i+4:n_factor), (n_factor*i+4:n_factor)] <- T
}
beta[211,219] <- 1 # S => I_0 (extra substrate feeding increases impurity)
beta[c(212, 214,215), 219] <- 1 # X_f, S, N => I1
beta[213, 218] <- 1 # C => P1
beta[217, c(224,225)] <- 1 # S0 => P1, I1
beta[218, 224] <- 1 # P0 => P1
beta[219, 225] <- 1 # I0 => I1
beta[223, c(230, 231)] <- 1 # S1 => P2, I2
beta[224, 230] <- 1 # P1 => P2
beta[225, 231] <- 1 # I1 => I2
# beta[(n_time-1)*6+1,n_factor*n_time] <- 1
# beta[(n_time-1)*6+2,(n_factor*n_time-3):(n_factor*n_time)] <- 1
# TF_mx[(n_factor*(n_time-1)+4:n_factor), (n_factor*(n_time-1)+4:n_factor)] <- T

# qc <- rep(0, n_time-1)
# vc <- rep(0, n_time-1)


# decision variable: FeedRate
# CQAs: biomass, CA, 
'
for (i in 0:((N+3)/6-2)) {
  beta[c((1+6*i):(6+6*i)),c((7+6*i):(9+6*i))] <- 0.9
  gamma[1:(6+6*i), 7+6*i] <- gamma[1:(6+6*i), 1:(6+6*i)]%*%beta[1:(6+6*i), 7+6*i]
  gamma[1:(7+6*i), 8+6*i] <- gamma[1:(7+6*i), 1:(7+6*i)]%*%beta[1:(7+6*i), 8+6*i]
  gamma[1:(8+6*i), 9+6*i] <- gamma[1:(8+6*i), 1:(8+6*i)]%*%beta[1:(8+6*i), 9+6*i]
}
'

##########################################################
# Gibbs Sampling

##########################################################
# Gibbs Sampling

# set prior 
# beta (the same)
the_0 <- 0
tau2_0 <- 10^2

# v inverse gamma (the same)
kap_0 <- 0.001
lam_0 <- 0.001

# mu normal (the same)
mu_0  <- 0
sig2_0 <- 10^2

# initial values
# p_mu <- mu
p_mu <- rep(0,N)
p_v2 <- v2
p_beta <- beta
# p_gamma <- gamma


# generate posterior samples
warmup <-500
hsize <- 10
B <- 500
Total <- warmup + hsize*B
mu_s <- matrix(0, Total, N)
v2_s <- matrix(0, Total, N)
beta_s <- array(0, dim = c(N,N,Total))
# gamma_s <- array(0, dim = c(N,N,Total))

# risk and uncertainty analysis
cov_s 	<- array(0, dim = c(N,N,Total))
shCPP_s <- array(0, dim = c(N,N,Total))
shCQA_s <- array(0, dim = c(N,N,Total))
shCPPp_s <- array(0, dim = c(N,N,Total))
shCQAp_s <- array(0, dim = c(N,N,Total))
marvar_s <- matrix(0, Total, N)  # marginal variance of each node

########################################################
# simulate data
########################################################

# macro <- 10 # number of macro-replication

# mCPPp_ma <- array(0, dim = c(N,N,macro))
# mCQAp_ma <- array(0, dim = c(N,N,macro))
# v2CPPp_ma <- array(0, dim = c(N,N,macro))
# v2CQAp_ma <- array(0, dim = c(N,N,macro))
# marvar_ma <- matrix(0, macro, N)

##########################################################
the_R_mat <- matrix(0, N, N)
tau2_R_mat <- matrix(0, N, N)
kap_R_mat <- rep(0,N)
lam_R_mat <- rep(0,N)
kap_R_mat_s <- matrix(0, Total, N)
lam_R_mat_s <- matrix(0, Total, N)
the_R_mat_s <- array(0, dim = c(N,N,Total))
tau2_R_mat_s <- array(0, dim = c(N,N,Total))

for (b in 1:Total){
  for (i in 1:N){
    
    ##############################
    # beta[, i]
    if (sum(beta[,i]) > 0){  # implies i >= 2
      for (j in 1:N){
        if (beta[j, i] > 0){
          alp <- dat[, j] - p_mu[j]  # R length vector
          p_beta[j, i] <- 0
          m 	<- dat[, i] - p_mu[i] - p_beta[, i]%*%(t(dat) - p_mu)
          the_R <- (tau2_0*sum(alp*m) + p_v2[i]*the_0)/(tau2_0*sum(alp^2) + p_v2[i])
          tau2_R <- (tau2_0*p_v2[i])/(tau2_0*sum(alp^2) + p_v2[i])
          p_beta[j, i] <- rnorm(1, mean = the_R, sd = sqrt(tau2_R))
          the_R_mat[j,i] <- the_R
          tau2_R_mat[j,i] <- tau2_R
        }
      }
    }
    
    ##############################
    # v2[i]
    u <- dat[, i] - p_mu[i] - p_beta[, i]%*%(t(dat) - p_mu)
    kap_R <- kap_0 + R
    lam_R <- lam_0 + sum(u^2)
    p_v2[i] <- 1/rgamma(1, shape = kap_R/2, rate = lam_R/2)
    kap_R_mat[i] <- kap_R
    lam_R_mat[i] <- lam_R
    # ##############################
    # # mu[i]
    # a <- dat[, i] - p_beta[, i]%*%(t(dat) - p_mu)
    # be1 <- p_beta
    # be1[i,] <- 0
    # # cc is R,N matrix, with each column c_ij: R length vector
    # cc <- t(t(matrix(dat[ ,i], R, N))*p_beta[i,]) - t((t(dat) - p_mu)*(p_beta[i,]>0)) + t(t(dat) - p_mu)%*%be1
    # sig2_R <- 1/(1/sig2_0 + R/p_v2[i] + R*sum(p_beta[i,]^2/p_v2))
    # mu_R <- sig2_R*(mu_0/sig2_0 + sum(a)/p_v2[i] + sum(t(cc)*p_beta[i,]/p_v2))
    # p_mu[i] <- rnorm(1, mean = mu_R, sd = sqrt(sig2_R))
    
    ##############################
    # compute gamma[, i]
    # if (i > 1){
    #   p_gamma[1:(i-1), i] <- p_gamma[1:(i-1), 1:(i-1)]%*%p_beta[1:(i-1), i]
    # }
  }
  
  # record b-th sample
  mu_s[b,] <- p_mu
  v2_s[b,] <- p_v2
  beta_s[, , b] <- p_beta
  the_R_mat_s[,,b] <- the_R_mat
  tau2_R_mat_s[,,b] <- tau2_R_mat
  kap_R_mat_s[b,] <- kap_R_mat
  lam_R_mat_s[b,] <- lam_R_mat
  # gamma_s[, , b] <- p_gamma
  
  # shCPP_s[, , b] <- v2_s[b,]*(gamma_s[, , b]^2)
  # marvar_s[b,]   <- apply(shCPP_s[, , b], 2, sum)
  # shCPPp_s[, , b] <- t(t(shCPP_s[, , b])/marvar_s[b,])
  # 
  # Ga_s <- p_gamma*sqrt(p_v2)
  # cov_s[, , b] <- t(Ga_s)%*%Ga_s
  # cov2_s <- cov_s[, , b]*TF_mx
  # for (i in 1:N){
  #   shCQA_s[1:i, i, b] <- (p_gamma[1:i, i]*cov2_s[1:i, 1:i])%*%p_gamma[1:i, i]
  # }
  # shCQAp_s[, , b] <- t(t(shCQA_s[, , b])/marvar_s[b,])
  # # remove the criticality out of boundary [0,1]
  # shCQAp_s[, , b][which(shCQAp_s[, , b] < 0)] <- 0
  # shCQAp_s[, , b][which(shCQAp_s[, , b] > 1)] <- 1
  
  
  if ((b%%100)==0){
    print(b)
  }
}

# posterior index
id <- seq((warmup+1), Total, by=hsize)

mu_ps <- mu_s[id,]
v2_ps <- v2_s[id,]
marvar_ps <- marvar_s[id,]
beta_ps <- beta_s[, , id]
beta_est <- apply(beta_ps,c(1,2),mean)
mu_est <- apply(mu_ps,2,mean)
v2_est <- apply(v2_ps,2,mean)

the_R_mat_ps <- the_R_mat_s[, , id]
tau2_R_mat_ps <- tau2_R_mat_s[, , id]
kap_R_mat_ps <- kap_R_mat_s[id,]
lam_R_mat_ps <- lam_R_mat_s[id,]

the_R_mat_est <- apply(the_R_mat_ps,c(1,2),mean)
tau2_R_mat_est <- apply(tau2_R_mat_ps,c(1,2),mean)
kap_R_mat_est <- apply(kap_R_mat_ps,2,mean)
lam_R_mat_est <- apply(lam_R_mat_ps,2,mean)

write.table(beta_est,sep=",",file="beta_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma25-2operations.txt",row.names=FALSE,col.names = FALSE)
write.table(v2_est,sep=",",file="v2_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma25-2operations.txt",row.names=FALSE,col.names = FALSE)

write.table(the_R_mat_est,sep=",",file="the_R_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma25-2operations.txt",row.names=FALSE,col.names = FALSE)
write.table(tau2_R_mat_est,sep=",",file="tau2_R_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma25-2operations.txt",row.names=FALSE,col.names = FALSE)
write.table(kap_R_mat_est,sep=",",file="kap_R_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma25-2operations.txt",row.names=FALSE,col.names = FALSE)
write.table(lam_R_mat_est,sep=",",file="lam_R_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma25-2operations.txt",row.names=FALSE,col.names = FALSE)
write.table(mu,sep=",",file="mu_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma25-2operations.txt",row.names=FALSE,col.names = FALSE)
write.table(sd,sep=",",file="sd_s5a1-R8-explore0.3-v1-modelrisk--ntime36-sigma25-2operations.txt",row.names=FALSE,col.names = FALSE)


