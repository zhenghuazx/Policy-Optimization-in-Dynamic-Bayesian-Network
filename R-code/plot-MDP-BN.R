library(visNetwork)
library(DAAG)
library(bnlearn)

ggplot(ais, aes(x = sport, y = hg, fill = sport)) + geom_boxplot() # + scale_fill_manual(values = colorRampPalette(king.yna)(10))
# set boolean variables
ais$high_hc <- as.factor(ais$hc > median(ais$hc))
ais$high_hg <- as.factor(ais$hg > median(ais$hg))

# create an empty graph
structure <- empty.graph(c("high_hc", "high_hg", "sport"))

# set relationships manually
modelstring(structure) <- "[high_hc][sport][high_hg|sport:high_hc]"

# plot network func
# using the visNetwork package to plot the network because it looks very nice.
plot.network <- function(structure, ht = "600px"){
  nodes.uniq <- unique(c(structure$arcs[,1], structure$arcs[,2]))
  nodes <- data.frame(id = nodes.uniq,
                      label = nodes.uniq,
                      color = "darkturquoise",
                      shadow = TRUE)
  
  edges <- data.frame(from = structure$arcs[,1],
                      to = structure$arcs[,2],
                      arrows = "to",
                      smooth = TRUE,
                      shadow = TRUE,
                      value= c(1,10),
                      color = "black")
  
  return(visNetwork(nodes, edges, height = ht, width = "100%"))
}


# observe structure
policy_theta = matrix(0,nrow=5,ncol=7)
policy_theta[,1] = c(0.13533677,
                     0.0663067 ,
                     0.11072438,
                     0.08723271,
                     0.07232338)
policy_theta[,2] =                  
                    c(0.23461393,
                     0.24332831,
                     0.12023716,
                     0.00779976,
                     0.18281191)
policy_theta[,3] =       
                    c(0.18872643,
                     0.19557626,
                     0.25757733,
                     0.17604082,
                     0.2112505)
policy_theta[,4] = 
                    c(0.1457705 ,
                     0.2053249 ,
                     0.03371047,
                     0.05471739,
                     0.20911444)
policy_theta[,5] = 
                    c(0.11050583,
                     0.12818917,
                     0.11994535,
                     0.19461364,
                     0.12400454)
policy_theta[,6] =  
                    c(0.04908034,
                     0.1525109 ,
                     0.15506139,
                     0.10446156,
                     0.09067574)
policy_theta[,7] = 
                    c(0.06820106,
                     0.01637698,
                     0.22274619,
                     0.09271438,
                     0.29315174)
  
  
  



beta_s5a1.R400.explore0.3 <- read.csv("~/Research/PHD/project/BN-MDP/BN/beta_s5a1-R400-explore0.3-v1.txt", header=FALSE)
v2_s5a1.R400.explore0.3 <- read.table("~/Research/PHD/project/BN-MDP/BN/v2_s5a1-R400-explore0.3-v1.txt", quote="\"", comment.char="")

for (i in 1:35) {
  beta_s5a1.R400.explore0.3[(i-1) * 6 + 3, (i-1) * 6 + 3 + 8] = 0
}

beta_s5a1.R400.explore0.3_matrix = data.matrix(beta_s5a1.R400.explore0.3)
BN.structure<- matrix(0, ncol = 9, nrow = 805)
count_index = 1

end_time = 20 / 4
end_time_index = end_time * 6
for (i in 1:end_time_index) {
  for (j in 1:(end_time_index+6)) {
    if (beta_s5a1.R400.explore0.3_matrix[i,j] != 0) {
      node_index = i%%6
      if (node_index == 1) {
        from_node = paste('feed rate', ' ','(',floor(i/6) * 4,')', sep='')
      } 
      if (node_index == 2) {
        from_node = paste('biomass', ' ','(',floor(i/6) * 4,')', sep='')
      }
      if (node_index == 3) {
        from_node = paste('citrate', ' ','(',floor(i/6) * 4,')', sep='')
      }
      if (node_index == 4) {
        from_node = paste('oil', ' ','(',floor(i/6) * 4,')', sep='')
      }
      if (node_index == 5) {
        from_node = paste('nitrogen', ' ','(',floor(i/6) * 4,')', sep='')
      }
      if (node_index == 0) {
        from_node = paste('volume', ' ','(',floor(i/6-1) * 4,')', sep='')
      }
      r_node_index = j %% 6
      if (r_node_index == 1) {
        to_node =  paste('feed rate', ' ','(',floor(j/6) * 4,')', sep='')
      } 
      if (r_node_index == 2) {
        to_node = paste('biomass', ' ','(',floor(j/6) * 4,')', sep='')
      }
      if (r_node_index == 3) {
        to_node = paste('citrate', ' ','(',floor(j/6) * 4,')', sep='')
      }
      if (r_node_index == 4) {
        to_node =  paste('oil', ' ','(',floor(j/6) * 4,')', sep='')
      }
      if (r_node_index == 5) {
        to_node =  paste('nitrogen', ' ','(',floor(j/6) * 4,')', sep='')
      }
      if (r_node_index == 0) {
        to_node =  paste('volume', ' ','(',floor(j/6-1) * 4,')', sep='')
      }
      
      edge = c(from_node, to_node, "to", T, T, round(beta_s5a1.R400.explore0.3_matrix[i,j],5),round(beta_s5a1.R400.explore0.3_matrix[i,j],3), 'black',FALSE) 
      BN.structure[count_index,] = edge
      count_index = count_index+ 1
    }
  }
}


BN.structure = as.data.frame(BN.structure[1:(count_index-1),])
colnames(BN.structure) <- c('from','to' ,'arrows','smooth' ,'shadow','value', 'label','color','hidden') # 
BN.structure$value = as.numeric(as.character(BN.structure$value))
BN.structure$label = as.numeric(as.character(BN.structure$label))


hidden_MDP = F
MDP.structure <- matrix(0, ncol = 9, nrow = end_time*5)
for (i in 1:end_time) {
  from_node1 = paste('biomass', ' ','(',(i-1) * 4,')', sep='')
  from_node2 = paste('citrate', ' ','(',(i-1) * 4,')', sep='')
  from_node3 = paste('oil', ' ','(',(i-1) * 4,')', sep='')
  from_node4 = paste('nitrogen', ' ','(',(i-1) * 4,')', sep='')
  from_node5 = paste('volume', ' ','(',(i-1) * 4,')', sep='')
  to_node = paste('feed rate', ' ','(',(i-1) * 4,')', sep='')
  edge1 = c(from_node1, to_node, "to", TRUE, TRUE, round(policy_theta[1,i],5),round(policy_theta[1,i],3), 'red', hidden_MDP) 
  edge2 = c(from_node2, to_node, "to", TRUE, TRUE, round(policy_theta[2,i],5),round(policy_theta[2,i],3), 'red', hidden_MDP) 
  edge3 = c(from_node3, to_node, "to", TRUE, TRUE, round(policy_theta[3,i],5),round(policy_theta[3,i],3), 'red', hidden_MDP) 
  edge4 = c(from_node4, to_node, "to", TRUE, TRUE, round(policy_theta[4,i],5),round(policy_theta[4,i],3), 'red', hidden_MDP) 
  edge5 = c(from_node5, to_node, "to", TRUE, TRUE, round(policy_theta[5,i],5),round(policy_theta[5,i],3), 'red', hidden_MDP) 
  MDP.structure[(i-1)*5+1,] = edge1
  MDP.structure[(i-1)*5+2,] = edge2
  MDP.structure[(i-1)*5+3,] = edge3
  MDP.structure[(i-1)*5+4,] = edge4
  MDP.structure[(i-1)*5+5,] = edge5
}
MDP.structure = as.data.frame(MDP.structure)
colnames(MDP.structure) <- c('from','to' ,'arrows','smooth' ,'shadow','value', 'label','color','hidden') # 
MDP.structure$value = as.numeric(as.character(MDP.structure$value))
MDP.structure$label = as.numeric(as.character(MDP.structure$label))
edges = rbind(BN.structure, MDP.structure)
edges$hidden = as.logical(as.character(edges$hidden))




v2_s5a1.R400.explore0.3 = as.matrix(v2_s5a1.R400.explore0.3)
nodes = matrix(NA,nrow = 216, ncol=8)
sparsity = 1
for (i in 1:(end_time_index+6)) {
  node_index = i%%6
  if (node_index == 1) {
    from_node = paste('feed rate', ' ','(',floor(i/6) * 4,')', sep='')
    node = c(from_node, 'feed rate', round(v2_s5a1.R400.explore0.3[i,1],3), "lightgray", FALSE, 'ellipse', 0)
  } 
  if (node_index == 2) {
    from_node = paste('biomass', ' ','(',floor(i/6) * 4,')', sep='')
    node = c(from_node, 'biomass', round(v2_s5a1.R400.explore0.3[i,1],3), "darkturquoise", FALSE, 'circle', 100 * sparsity)
  }
  if (node_index == 3) {
    from_node = paste('citrate', ' ','(',floor(i/6) * 4,')', sep='')
    node = c(from_node, 'citrate', round(v2_s5a1.R400.explore0.3[i,1],3), "orange", FALSE, 'circle', 200 * sparsity)
  }
  if (node_index == 4) {
    from_node = paste('oil', ' ','(',floor(i/6) * 4,')', sep='')
    node = c(from_node, 'oil', round(v2_s5a1.R400.explore0.3[i,1],3), "lightgreen", FALSE, 'circle', 300 * sparsity)
  }
  if (node_index == 5) {
    from_node = paste('nitrogen', ' ','(',floor(i/6) * 4,')', sep='')
    node = c(from_node, 'nitrogen', round(v2_s5a1.R400.explore0.3[i,1],3), "lightblue", FALSE, 'circle',400 * sparsity)
  }
  if (node_index == 0) {
    from_node = paste('volume', ' ','(',floor(i/6-1) * 4,')', sep='')
    node = c(from_node, 'volume', round(v2_s5a1.R400.explore0.3[i,1],3), "yellow", FALSE, 'circle',500 * sparsity)
  }
  # if (node_index == 1) {
  #   node = c(from_node, from_node, round(v2_s5a1.R400.explore0.3[i,1],3), "darkred", FALSE, 'square')
  # } else {
  #   node = c(from_node, from_node, round(v2_s5a1.R400.explore0.3[i,1],3), "darkturquoise", FALSE, 'circle')
  # }
  
  if (node_index == 1) {
    node <- c(node, (ceiling(i/6.00001)+0.5)*300 * sparsity)
  } else {
    node <- c(node, ceiling(i/6.00001)*300* sparsity)
  }
  nodes[i,] = node
}

nodes = as.data.frame(nodes[1:(end_time_index+6),])
nodes <- nodes[which(1:nrow(nodes)  != end_time_index + 1),]
colnames(nodes) <- c('id','label' ,'value','color' ,'shadow', 'shape','y', 'x')
nodes$value = as.numeric(as.character(nodes$value))

net <- graph_from_data_frame(d=edges, vertices=nodes, directed=T)
netm <- as_adjacency_matrix(net, attr="value", sparse=F)
colnames(netm) <- nodes$id # V(net)
rownames(netm) <- nodes$id

## BN + MDP
visNetwork(nodes, edges[,-7], height = '1000px', width = "100%")%>%
  visLayout(randomSeed = 31313) %>%
  visPhysics(solver = "barnesHut", barnesHut = list(gravitationalConstant = -30000000,springConstant=0.02, centralGravity=0)) %>%
  visEdges(scaling = list('max'=7, 'min'=1) ) %>%
  visNodes(font=list(size=25),fixed =TRUE) 

write.csv(edges[c(1,2,6)], 'edge')

# 
# palf <-colorRampPalette(c("yellow2", "dark orange"))#  colorRampPalette(c("red","orange","blue")) # 

# library("RColorBrewer")
# # heatmap.2(data_matrix,col=brewer.pal(11,"RdBu"),scale="row", trace="none"
# heatmap(netm, Rowv = NA, Colv = NA, col = brewer.pal(9,"RdYlBu"),
#         scale="none", margins=c(7,7), xlab='state/action (to)', ylab='state/action (from)')
# legend('topleft', col = brewer.pal(9,"RdYlBu"))
# 
# library(ggplot2)
# arcs_data <- edges[c(1,2,6,7)]
# arcs_data$track_name <- factor(arcs_data$track_name, levels = c())
# 
# ggheatmap <- ggplot(edges[c(1,2,6,7)], aes(to, from, fill = value))+
#   geom_tile(color = "white")+
#   scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
#                        midpoint = 0, limit = c(-1.2,1.2), space = "Lab", 
#                        name="Cofficient") +
#   geom_text(aes(to, from, label = label), color = "black", size = 4) +
#   theme_minimal()+
#   theme(axis.text.x = element_text(angle = 45, vjust = 1, 
#                                    size = 10, hjust = 1), axis.text.y=element_text(size=10))+
#   coord_fixed() 
#   


dat = netm %>% 
  as.data.frame() %>%
  rownames_to_column("from") %>%
  pivot_longer(-c(from), names_to = "to", values_to = "value") %>%
  mutate(to= fct_relevel(to,colnames(netm))) %>%
  mutate(from= fct_relevel(from,colnames(netm)))


ggheatmap <- ggplot(dat, aes(to, from, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-3.2,3), space = "Lab", 
                       name="Cofficient") +
  # geom_text(aes(to, from, label = label), color = "black", size = 4) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 8, hjust = 1), 
        axis.text.y=element_text(size=8),
        legend.justification = c(1, 0),
        legend.position = c(0.5, 0.85),
        legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 10, barheight = 1,
                               title.position = "top", title.hjust = 0.5))
# +
  coord_fixed() 
print(ggheatmap)



