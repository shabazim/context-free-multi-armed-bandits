This project is for implementing Multi-Armed-Bandit algorithms and testing different decision policies to address the exploration/exploitation dilemma.

## Algorithms ## 

The policies tested so far are:

* ### Epsilon Greedy ###

    * With a small probability epsilon it chooses a random arm and with probability (1-epsilon) it chooses the arm with maximum exprected reward
The epsilon-greedy strategy uses a constant (epsilon) to balance exploration and exploitation. 
  This is not ideal, as the hyperparameter may be hard to tune. 
  Also, the exploration is done in 100% randomly, such that we explore bandits equally (in average), irrespective of how promising they may look.
  As a result, the ε-Greedy bandit will continue to explore at the same rate, even after the best movie has been found, leading to suboptimal performance in the long run. 
  (There are epsilon-Greedy algorithms that adjust the value of epsilon over time to account for this, like decaying epsilon greedy to explore less at the end of the test)

* ### Decaying Epsilon Greedy ###

* ### UCB ###

    * The exploration-exploitation balance is achieved through computing the confidence interval for each expected reward. This is
usually done with respect to the number draws of each arm versus the total number of current draws.


* ### Thompson Sampling ###

    * Takes one sample from each arm's reward distribution, and choose the one with the highest value. So we favor exploration by sampling from the distributions
with high uncertainty and high tails,
and exploit by sampling from the distribution with highest mean.
Despite its simplicity, Thompson Sampling greatly outperforms the other algorithms. 
This bandit quickly identifies the best arm and exploits it more frequently after it has been found, leading to high performance in both the short- and long-term.
The following two methods are actually the varients of Thompson sampling:

* ### UCB-Bayes ###
    * Instead of sampling directly from posterior we sample the quantile function of the posterior of each arm.
    
* ### Upper Credible Choice ###
    * We sample the upper bound of the 90% credible interval of eacha arm's posterior distribution.
    
* ### Non-stationary bandits ###

    * In the case where the reward distribution changes over time, we want to modify the current belief of the bandit to
allow it to adapt itself to the varying situation. Here are some of the algorithms tested:

        * #### Sliding Window UCB ####
            * This algorithm is a variant of the regular UCB algorithm. Each arm keeps a list
            of rewards of fixed length. Each time an arm is pulled, its reward is added to the list. If the list is full, the
            oldest reward is removed to make room for the new one. The estimated expected reward is computed from the
            reward list. The rest of the algorithm is the same as the regular UCB policy. In a non-stationary context, this
            algorithm will adapt itself over time. The length of the list will define how quickly and how accurately it adapts
            to changes. If the list is too short, there will not be enough data to converge to the best action. If the list is
            too long, it will take too much time to adapt to a new situation. This parameter has to be tuned according to
            every particular contexts.
        * #### Sliding Window Thompson Sampling ####
            * we use a list of fixed length containing a boolean value for
            the reward. We can then compute alpha and beta (in case of beta posterior) from that list each time a new reward is added. This will limit
            the maximal probability of being optimal and allow adaptation through time. A short window allows a lot
            of exploration while a large window allows more exploitation. For the same window length, this version of
            thompson sampling appears to be performing better than sliding window UCB.
        * #### Discounted Thompson Sampling ####
            * At each iteration, we modify the parameters alpha and beta of the prior distribution (in case of beta distribution) of each arm by multiplying them with a small constant 
            [0,1) as discount factor. The update rule is the same as Thompson Sampling
            
* ### Exp3 ###
    
        
    * Exp3 stands for Exponential-weight algorithm for Exploration and Exploitation. 
      It works by maintaining a list of weights for each of the arms, 
      using these weights to decide randomly which action to take next, and increasing (decreasing) the relevant weights 
      when a payoff is good (bad). We further introduce an egalitarianism factor gamma in [0,1] which tunes the desire 
      to pick an action uniformly at random. That is, if gamma = 1, the weights have no effect on the choices at any step.


## Files and Folders ## 

The "simulator.py" contains all the classes defined for each policy mentioned in the previous section. To test or to try the models, use the "test" notebook.
The results are already stored in the "Results" folder. Please refer to the plots to capture a quick summary 
