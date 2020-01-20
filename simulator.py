import numpy as np
from tqdm import tqdm
from scipy.stats import beta
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import math

class MABSimulator(object):
    """
    A class for simulating online multi armed bandit algorithms.
    Since the learning process involves stochasticity, to reduce this randomness
    we perform many simulations of many rounds of play and average the results per round across all simulations.
   
    """
    
    def __init__(self,n_simulations,n_rounds,bandit_probs,sliding_window=False,window_length=None, random_seed=1):
        
        np.random.seed(random_seed)
        # number of simulations
        self.n_simulations = n_simulations
        # number of rounds per simulation
        self.n_rounds = n_rounds
        # bandits real probebility of reward
        self.bandit_probs = bandit_probs
        self.n_bandits = len(bandit_probs)
        # cumulative number of rounds
        self.N = 0
        # in case using sliding window for non-stationary bandits
        self.sliding_window = sliding_window
        self.window_length = window_length
        if self.sliding_window:
            self.stored_rewards = [[] for i in range(self.n_bandits)]
        
    def simulate(self):
        """
        returns: a list of dictionaries containing the results for each simulation and each round
        """
        
        results = []
        
        for sim in tqdm(range(0,self.n_simulations)):
            
            # initialize the parameters of each bandit
            self.reset()
            
            total_rewards = 0
            total_regrets = 0
            for round_id in range(self.n_rounds):
                
                # choose arm and pull it
                k = self.select_bandit()
                reward, regret = self.draw(k)
                
                # record infos about this pulling
                self.record_result(k,reward,round_id)
                total_rewards += reward
                total_regrets += regret
                self.N += 1
                result = {}
                result['simulation id'] = sim
                result['round id'] = round_id
                result['reward'] = reward
                result['total_reward'] = total_rewards
                result['cumulative expected reward'] = total_rewards * 1. / (round_id + 1)
                result['cumulative regret'] = total_regrets
                
                results.append(result)
        return results
    
    def plot_MAB_experiment(self, bandit_colors, plot_title):
        """
        plots and animates estimated probability distributions over expected reward (p(\theta|y)) during a single simulation
        """

        # clearing past figures
        plt.close('all')


        # lists for accumulating draws, bandit choices and rewards
        k_list = []
        reward_list = []

        # animation dict for the posteriors
        posterior_anim_dict = {i:[] for i in range(self.n_bandits)}

        # opening figure
        fig = plt.figure(figsize=(9,5), dpi=150)

        # position our plots in a grid, the largest being our plays      
        colspan1 = self.n_bandits
        rowspan1 = self.n_bandits-1
        ax = []
        ax.append(plt.subplot2grid((self.n_bandits+1, self.n_bandits), (0, 0), colspan=colspan1, rowspan=rowspan1)) 
        for i in range(self.n_bandits):
            ax.append(plt.subplot2grid((self.n_bandits+1, self.n_bandits), (rowspan1, i), rowspan=2))
            
        
        self.reset() 

        # loop generating draws
        for round_id in range(self.n_rounds):

            # record information about this draw
            k = self.select_bandit()
            reward, regret = self.draw(k)
            
            # record information about this draw
            self.record_result(k,reward,round_id)
            k_list.append(k)
            reward_list.append(reward)

            # sucesses and failures for our beta distribution
            success_count = self.n_arm_rewards*self.n_arm_samples
            failure_count = self.n_arm_samples - success_count

            # calculating pdfs for each bandit
            for bandit_id in range(self.n_bandits):
                
                X = np.linspace(0,1,1000)
                curve = beta.pdf(X, 1+ success_count[bandit_id], 1+failure_count[bandit_id])


                # appending to posterior animation dict
                posterior_anim_dict[bandit_id].append({'X': X, 'curve': curve})

            # getting list of colors that tells us the bandit
            color_list = [bandit_colors[k] for k in k_list]

            # getting list of facecolors that tells us the reward
            facecolor_list = [['none', bandit_colors[k_list[i]]][r] for i, r in enumerate(reward_list)]

        # fixing properties of the plots
        ax[0].set(xlim=(-1, self.n_rounds), ylim=(-0.5, self.n_bandits-0.5))
        ax[0].set_title(plot_title, fontsize=10)
        ax[0].set_xlabel('Round', fontsize=10); ax[0].set_ylabel('Bandit', fontsize=10)
        ax[0].set_yticks(np.arange(self.n_bandits).tolist())
        ax[0].set_yticklabels(['{}\n($\\theta = {}$)'.format(i, self.bandit_probs[i]) for i in range(self.n_bandits)])
        ax[0].tick_params(labelsize=10)

        # titles of distribution plots
        for i in range(self.n_bandits):
            ax[i+1].set_title('Estimated $\\theta_{}$'.format(i), fontsize=10);
            

        # initializing with first data
        scatter = ax[0].scatter(y=[k_list[0]], x=[list(range(self.n_rounds))[0]], color=[color_list[0]], \
                              linestyle='-', marker='o', s=30, facecolor=[facecolor_list[0]])
        dens = []
        for i in range(self.n_bandits):
            densi = ax[i+1].fill_between(posterior_anim_dict[i][0]['X'], 0, posterior_anim_dict[i][0]['curve'], \
                                     color=bandit_colors[i], alpha=0.7)
            dens.append(densi)
      
     

        # function for updating
        def animate(i):

            # clearing axes
            ax[0].clear()
            for k in range(self.n_bandits):
                ax[k+1].clear()
           
            # updating game rounds
            scatter = ax[0].scatter(y=k_list[:i], x=list(range(self.n_rounds))[:i], color=color_list[:i], 
                                  linestyle='-', marker='o', s=30, facecolor=facecolor_list[:i]);

            # fixing properties of the plot
            ax[0].set(xlim=(-1, self.n_rounds), ylim=(-0.5, self.n_bandits-0.5))
            ax[0].set_title(plot_title, fontsize=10)
            ax[0].set_xlabel('Round', fontsize=10); ax[0].set_ylabel('Bandit', fontsize=10)
            ax[0].set_yticks(np.arange(self.n_bandits).tolist())
            ax[0].set_yticklabels(['{}\n($\\theta = {}$)'.format(i, self.bandit_probs[i]) for i in range(self.n_bandits)])
            ax[0].tick_params(labelsize=10)
            
            
            

            # updating distributions
            for j in range(self.n_bandits):
                dens[j] = ax[j+1].fill_between(posterior_anim_dict[j][i]['X'], 0, posterior_anim_dict[j][i]['curve'], 
                                         color=bandit_colors[j], alpha=0.7)
            
            # titles of distribution plots
            for l in range(self.n_bandits):
                ax[l+1].set_title('Estimated $\\theta_{}$'.format(l), fontsize=10);

            

            # do not need to return 
            return ()

        # function for creating animation
        anim = FuncAnimation(fig, animate, frames=self.n_rounds, interval=100, blit=True)

        # fixing the layout
        fig.tight_layout()

        # showing
        return HTML(anim.to_html5_video())

               
    def select_bandit(self):
        """
        returns: a random arm 
        """
        return np.random.randint(self.n_bandits)
    
    def draw(self,k):
        """
        returns: reward and regret for the selected arm
        """
        return np.random.binomial(1, self.bandit_probs[k]), np.max(self.bandit_probs) - self.bandit_probs[k]
        
    def record_result(self,k,reward,round_id):
        """
        updates selected arm'infos
        """
        self.n_arm_samples[k] += 1
        self.n_arm_rewards[k] += (1./self.n_arm_samples[k]) * (reward - self.n_arm_rewards[k])
        
    def reset(self):
        
        # number of times each arm has been sampled 
        self.n_arm_samples = np.zeros(self.n_bandits)
        
        # estimated expected reward for each arm 
        # (in case of binary reward fraction of times each selected arm has resulted in a non-zero reward)
        self.n_arm_rewards = np.zeros(self.n_bandits)
        
        
        
    
    
class ABTestSimulator(MABSimulator):
    def __init__(self,n_tests, n_simulations, n_rounds, bandit_probs):
        
        super(ABTestSimulator, self).__init__(n_simulations,n_rounds,bandit_probs)
        # number rounds for testing (exploring) phase (number od samples)
        self.n_tests = n_tests
        self.is_testing = True
        self.best_arm = None
        
    def reset(self):
        super(ABTestSimulator, self).reset()
        
        self.is_testing = True
        self.best_arm= None
    
    def select_item(self):
        if self.is_testing:
            return super(ABTestSimulator, self).select_item()
        else:
            return self.best_arm
            
    def record_result(self, k, reward,round_id):
        super(ABTestSimulator, self).record_result(k, reward,round_id)
        
        if (round_id == self.n_tests - 1): # this was the last visit during the testing phase
            
            self.is_testing = False
            self.best_arm = np.argmax(self.n_arm_rewards)
            
class EpsilonGreedySimulator(MABSimulator):
    
    def __init__(self, epsilon, n_simulations,n_rounds,bandit_probs):
        super(EpsilonGreedySimulator, self).__init__(n_simulations,n_rounds,bandit_probs)
        
        self.epsilon = epsilon
        
    def select_bandit(self):
        
        # decide to explore or exploit
        if np.random.uniform() < self.epsilon: # explore
            k = super(EpsilonGreedySimulator, self).select_bandit()
            
        else: # exploit
            k = np.argmax(self.n_arm_rewards)
            
        return k
    
    
    
class ThompsonSamplingSimulator(MABSimulator):
    
    def reset(self):
        super(ThompsonSamplingSimulator, self).reset()
        self.alphas = np.ones(self.n_bandits)
        self.betas = np.ones(self.n_bandits)
        
    def select_bandit(self):
    
        samples = [np.random.beta(a,b) for a,b in zip(self.alphas, self.betas)]
        
        return np.argmax(samples)
    
    def record_result(self, k, reward,round_id):
        super(ThompsonSamplingSimulator, self).record_result(k, reward,round_id)
        ## update value estimate
        if reward == 1:
            self.alphas[k] += 1
        else:
            self.betas[k] += 1
    
class UCBSimulator(MABSimulator):
#     def __init__(self, modifier, n_simulations,n_rounds,n_bandits,bandit_probs):
#         self.modifier = modifier
        
    def select_bandit(self):
        
        sqrt_terms =  np.sqrt(2.0*np.log(np.sum(self.n_arm_samples)) /self.n_arm_samples )
        
        for i in range(self.n_bandits):
            if self.n_arm_samples[i] == 0:
                sqrt_terms[i] = float('inf')
            
                
        
        return np.argmax(self.n_arm_rewards + sqrt_terms)
    
class UCBBayesSimulator(MABSimulator):
    
    def reset(self):
        super(UCBBayesSimulator, self).reset()
        self.alphas = np.ones(self.n_bandits)
        self.betas = np.ones(self.n_bandits)
        self.N = 0
        
    def select_bandit(self):
        
        c = 1 - 1./((self.N+1))
        
        # percent point functions(inverse of cdf)            
        pps = [beta.ppf(c,a,b) for a,b in zip(self.alphas, self.betas)]
        
        return np.argmax(pps)
    
    def record_result(self, k, reward,round_id):
        super(UCBBayesSimulator, self).record_result(k, reward,round_id)
        ## update value estimate
        if reward == 1:
            self.alphas[k] += 1
        else:
            self.betas[k] += 1
    
class UpperCredibleChoiceSimulator(MABSimulator):
    def reset(self):
        super(UpperCredibleChoiceSimulator, self).reset()
        self.alphas = np.ones(self.n_bandits)
        self.betas = np.ones(self.n_bandits)
        
        
    def select_bandit(self):
        # upper bounds
        ubs = [a/(a+b) + 1.65*np.sqrt((a*b)/((a+b)**2*(a+b+1))) for a,b in zip(self.alphas, self.betas)] 
        
        return np.argmax(ubs)
    
    def record_result(self, k, reward,round_id):
        super(UpperCredibleChoiceSimulator, self).record_result(k, reward,round_id)
        ## update value estimate
        if reward == 1:
            self.alphas[k] += 1
        else:
            self.betas[k] += 1
class DiscountedThompsonSamplingSimulator(MABSimulator):
    
    def __init__(self, discount, n_simulations,n_rounds,bandit_probs):
        super(DiscountedThompsonSamplingSimulator, self).__init__(n_simulations,n_rounds,bandit_probs)
        
        self.discount = discount
        
    def reset(self):
        super(DiscountedThompsonSamplingSimulator, self).reset()
        self.alphas = np.ones(self.n_bandits)
        self.betas = np.ones(self.n_bandits)
        
    def select_bandit(self):
        
        samples = [np.random.beta(a,b) for a,b in zip(self.alphas, self.betas)]
        
        return np.argmax(samples)
    
    def record_result(self, k, reward,round_id):
        super(DiscountedThompsonSamplingSimulator, self).record_result(k, reward,round_id)
        self.alphas = self.alphas * self.discount
        self.betas  = self.betas * self.discount
        for a,b in zip(self.alphas, self.betas):
            if a < 1.0:
                a = 1.0
            if b < 1.0:
                b = 1.0
        ## update value estimate of arm k
        if reward == 1:
            self.alphas[k] += 1
        else:
            self.betas[k] += 1
    
class SlidingWidowUCBSimulator(MABSimulator):

    def record_result(self,k,reward,round_id):
        
        self.n_arm_samples[k] += 1
        
        if len(self.stored_rewards[k]) == self.window_length:
            self.stored_rewards[k].pop(0)
        self.stored_rewards[k].append(reward)
        
        if self.n_arm_samples[k] == 0 or len(self.stored_rewards[k]) == 0:
            self.n_arm_rewards[k] = 0
        else:
            estimate = 0
            for r in self.stored_rewards[k]:
                estimate += r
            estimate = estimate / len(self.stored_rewards[k])
            self.n_arm_rewards[k] = estimate
        
        
        
    def select_bandit(self):
        
        sqrt_terms =  np.sqrt(2.0*np.log(np.sum(self.n_arm_samples)) /self.n_arm_samples )
        for i in range(self.n_bandits):
            if self.n_arm_samples[i] == 0:
                sqrt_terms[i] = float('inf')
                
        return np.argmax(self.n_arm_rewards + sqrt_terms)
    
class SlidingWidowThompsonSamplingSimulator(MABSimulator):
    
    def reset(self):
        
        # number of times each arm has been sampled 
        self.n_arm_samples = np.zeros(self.n_bandits)
        
        # estimated expected reward for each arm 
        # (in case of binary reward fraction of times each selected arm has resulted in a non-zero reward)
        self.n_arm_rewards = np.zeros(self.n_bandits)
        
        self.alphas = np.ones(self.n_bandits)
        self.betas = np.ones(self.n_bandits)
        
    def select_bandit(self):
    
        samples = [np.random.beta(a,b) for a,b in zip(self.alphas, self.betas)]
        
        return np.argmax(samples)
    
    def record_result(self, k, reward,round_id):
        
        
        
        self.n_arm_samples[k] += 1
        
        if len(self.stored_rewards[k]) == self.window_length:
            self.stored_rewards[k].pop(0)
        self.stored_rewards[k].append(reward>0)
        
        self.alphas[k] = 1
        self.betas[k] = 1
        
        if self.n_arm_samples[k] != 0 or len(self.stored_rewards[k]) != 0:
            self.alphas[k] = 1
            self.betas[k] = 1
            for r in self.stored_rewards[k]:
                if r:
                    self.alphas[k] += 1
                else:
                    self.betas[k] += 1
        
       
     
    
class DecayingEpsilonGreedySimulator(MABSimulator):
    def __init__(self, epsilon, tau, n_simulations,n_rounds,bandit_probs):
        super(DecayingEpsilonGreedySimulator, self).__init__(n_simulations,n_rounds,bandit_probs)
        
        self.epsilon = epsilon
        self.tau = tau
        
    def select_bandit(self):
        
        e = self.epsilon * np.exp( -1.0 * float(self.N) * self.tau)
        # if not exponential
        #e = epsilon/float(self.N+1)
                
        if e < 0.01:
            e = 0.01
        # decide to explore or exploit
        if np.random.uniform() < e: # explore
            k = super(EpsilonGreedySimulator, self).select_bandit()
            
        else: # exploit
            k = np.argmax(self.n_arm_rewards)
            
        return k
    
    
# https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
class Exp3Simulator(MABSimulator):
    def __init__(self, gamma, n_simulations,n_rounds,bandit_probs):
        super(Exp3Simulator, self).__init__(n_simulations,n_rounds,bandit_probs)
        self.gamma = gamma
    
    def reset(self):
        self.weights = np.ones(self.n_bandits)
        self.probs = (1 - self.gamma) * (self.weights /np.sum(self.weights)) + \
                      self.gamma / float(self.n_bandits) 
        super(Exp3Simulator, self).reset()
    
    def record_result(self, k, reward,round_id):
        
        self.n_arm_rewards[k] = reward / self.probs[k]
        self.weights[k] *= math.exp((self.gamma * self.n_arm_rewards[k]) / (float(self.n_bandits)))
        self.n_arm_samples[k] += 1
    
    def select_bandit(self):
        return np.random.multinomial(1, self.probs.tolist()).tolist().index(1)
