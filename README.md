# A Reinforcement Learning Approach with Masked Agents for Chemical Process Flowsheet Design

This repository contains the code for the two case studies presented in the paper *A reinforcement learning approach with masked agents for chemical process flowsheet design*. 
The work focuses on the generation, design and optimization of chemical process flowsheets using Reinforcement Learning. 

The full paper can be found in <https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.18584?af=R>

## Case Study 1
This case study compares the performance of discrete and hybrid masked PPO agents in generating a chemical process flowsheet for the reaction $A \rightarrow B$. With this illustrative example it was found that for simple examples in which the number of discrete and continuous varibles are reduced, a fully discretized agent outperfroms the hybrid agent, achieving better rewards. 

  
## Case Study 2
In this case study, the performance of discrete and hybrid masked PPO agents is compared for generating a chemical process flowsheet for the reversible reaction:
$$ 2\text{CH}_3\text{OH} \rightleftharpoons \text{DME} + \text{H}_2\text{O}$$
Despite achieving lower overall rewards, the hybrid agent produces more complex and interesting flowsheets. These flowsheets are able to reintegrate more mass into the system, offering potential sustainability benefits. The example was executed using a hybrid Python - ASPEN Plus platform, in which the usage of multiple unit operations was featured.

## Code Access
The full code for both case studies can be found in the repository. 