library(ggplot2)
library(wesanderson)
library(readr)


feeding_profile_R400 <- read.csv("~/Research/PHD/project/BN-MDP/empirical_study/liboptpy/feeding_profile_R400.csv")

p1<- ggplot(feeding_profile_R400, aes(x=time, y=mean, fill=algorithms)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) +
  geom_errorbar(aes(ymin=mean-1.96 * sem, ymax=mean+1.96*sem), width=2, size=0.3,
                position=position_dodge(3.6))

# Finished bar plot
p1 <- p1+labs(x="Time (hr)", y = "feed rate (L/h)")+
  scale_x_continuous(breaks = round(seq(min(feeding_profile_R400$time), max(feeding_profile_R400$time), by = 4),1)) + 
  # scale_fill_manual(wes_palette(name="GrandBudapest"))
  # scale_fill_brewer(palette = 'Set2') +
  scale_y_continuous(limits = c(0, 0.01)) 

p1 <- p1 + theme(legend.background = element_rect(fill="lightgray", 
                                       size=0.5, linetype="solid"), legend.position = c(0.9, 0.88))
print(p1)

feed_profile_init_mean <- apply(F_S, 2, mean)
feed_profile_init_se <- apply(F_S, 2, sd) / sqrt(8)
feeding_profile_human = cbind(ttname, feed_profile_init_mean, feed_profile_init_se)
feeding_profile_human = as.data.frame(feeding_profile_human)

p2<- ggplot(feeding_profile_human, aes(x=ttname, y=feed_profile_init_mean)) + 
  geom_line(size=1.2) +
  geom_errorbar(aes(ymin=feed_profile_init_mean-1.96 * feed_profile_init_se, ymax=feed_profile_init_mean+1.96*feed_profile_init_se), width=2, size=0.5,
                position=position_dodge(3.6))+
  geom_point()

p2 <- p2+labs(x="Time (hr)", y = "feed rate (L/h)")+
  scale_x_continuous(breaks = round(seq(min(feeding_profile_R400$time), max(feeding_profile_R400$time), by = 10),1)) + 
  # scale_fill_manual(wes_palette(name="GrandBudapest"))
  scale_y_continuous(limits = c(0, 0.016)) +
  scale_color_manual(values=c('#E69F00'))
  
print(p2)


X170506F3 <- read_csv("170506F3.csv")

X170517F3 <- read_csv("170517F3.csv")

X170517F4 <- read_csv("170517F4.csv")

X170601F3 <- read_csv("170601F3.csv")

#X170613F3 <- read_csv("170613F3.csv")

X170706F3 <- read_csv("170706F3.csv")

X170721F3 <- read_csv("170721F3.csv")

X170803F3 <- read_csv("170803F3.csv")

#X170817F3 <- read_csv("170817F3.csv")

X171003F3 <- read_csv("171003F3.csv")


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

ttname = X170506F3$'Ferm. Time (h)'

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



