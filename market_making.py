import gym
import gym.spaces
import numpy as np


# Ha senso che la classe environment del problema contenga cose legate allo stato interno dell'agente (tipo inventory) o dovrebbero essere gestite dall'agente stesso?

# Da provare a guardare https://github.com/openai/gym/tree/master/gym/envs/classic_control/assets e prendere ispirtazione da come sono fatti gli environment di esempio

class MarketMakerEnv(gym.Env):

    MAX_ORDER_SIZE = 1e3
    MAX_INVENTORY  = 10000.0
    MAX_VALUE = 1e7

    ACTIONS_0_8 = np.array([
        [1,1],
        [2,2],
        [3,3],
        [4,4],
        [5,5],
        [1,3],
        [3,1],
        [2,5],
        [5,2],
        [0,0] # Action 9
    ])


    def __init__(self, lob_data: np.array, feature_extractor: callable):
        self.lob_depth = lob_data.shape[1]
        self.lob_data = lob_data
        self.feature_extractor = feature_extractor # to do

        # Agent's internal state
        self.inventory = 0

        self.observation_space = gym.spaces.Dict({
            'order_book' : gym.spaces.MultiDiscrete(np.full((self.lob_depth*2,4)*MarketMakerEnv.MAX_VALUE)),
            
        })

        self.action_space = gym.spaces.Dict({
            # Manca order side?5
            # 0: Ask, 1: Bid
            'order_size': gym.spaces.MultiDiscrete([MarketMakerEnv.MAX_ORDER_SIZE, MarketMakerEnv.MAX_ORDER_SIZE]), # Sampling di coppie di interi dall'intervallo [0,999]
            'theta': gym.spaces.Discrete(9), # Sampling di due interi dall'intervallo [0,9] 
        })


    
    """
    Reset the environment to the initial state.
    Returns:
        dict: The initial observation of the environment
    """
    def reset(self):
        self.t = 0
        self.inventory = 0
        return self._get_obs()
    
    
    # Forse dovrebbe estrarre le info non direttamente dal dataset ma da una classe/oggetto "stato" che contiene sia le info del book che le varie features?

    def _get_obs(self):

        lob_obs = self.lob_data[self.t,:]
        
        features_obs = self.feature_extractor(lob_obs)
        
        return {
            'order_book': lob_obs,  
            'features': features_obs,  
            'inventory': np.array([self.inventory], dtype=int)  
        }
    

    def feature_extractor(self, lob_obs):

        feature = []

        # Diverse di queste feature fanno riferimento a quantità che evolvono nel tempo ma sono inerenti a singoli stati
        
        mid_price = (lob_obs[0] + lob_obs[2]) / 2
        market_spread = lob_obs[2] - lob_obs[0]
        mid_price_move = # to be completed
        book_imbalance = (lob_obs[3] - lob_obs[1])/(lob_obs[3] + lob_obs[1]) # (v_bid - v_ask) / (v_bid + v_ask)
        signed_volume = lob_obs[3] - lob_obs[1] # v_bid - v_ask
        volatility = # to be completed
        relative_strength_index = # to be completed





    def AgentCashHolding(self, prev_state, current_state):
        prev_mid_price = prev_state['features'][0]
        current_mid_price = current_state['features'][0]   

        current_inventory = current_state['inventory']
        diff_inventory = current_mid_price - prev_mid_price 

        return (diff_inventory * current_inventory)
    
    
    def PnL_reward(self, prev_state, action, current_state):

        reward = AgentCashHolding(prev_state, current_state)

        # Profit derived from order execution
        # 0.) Ask Price 1: 	Level 1 Ask Price 	(Best Ask)
	    # 1.) Ask Size 1: 	Level 1 Ask Volume 	(Best Ask Volume)
	    # 2.) Bid Price 1: 	Level 1 Bid Price 	(Best Bid)
	    # 3.) Bid Size 1: 	Level 1 Bid Volume 	(Best Bid Volume)

        # ISSUE: dalle formule per psi_a e psi_b abbiamo che Matched_a e Matched_b sono:
        # "Amount of volume matched (executed) against the agentìs orders since the last time t_{i-1}"
        # psi_a = Matched_a(t_i) * ( p_a(t_i) - m(t_i))     
        # psi_b = Matched_b(t_i) * ( m(t_i) - p_b(t_i))
        # Quindi Matched_a e Matched_b NON sono necessariamente uguali a order_size[0] e order_size[1] ?
        

        # Da controllare se book_ask e book_bid sono i valori corretti da scegliere e se servono
        current_mid_price = current_state['feature'][0]
        order_size = action['order_size']
        theta = action['order_side']

        book_a = current_state['order_book'][0]
        book_b = current_state['order_book'][2]

        spread = book_b - book_a
        dist = spread/2 * theta * np.array([1,-1]) # bid has to be under mid-price and ask above)ù

        # current_state['feature'][0] non è gia il mid price?
        # agent_ab = np.mean([book_a, book_b]) + dist
        agent_ab = current_mid_price + dist

        psi_a = order_size[0] * ( agent_ab[0] - current_mid_price )
        psi_b = order_size[1] * ( current_mid_price - agent_ab[1] )
        reward += psi_a + psi_b

        return reward

    def Sym_dampened_PnL_reward(self, prev_state, action, current_state, eta):
        return (PnL_reward(prev_state, action, current_state) - eta * AgentCashHolding(prev_state, current_state))


    def Asym_dampened_PnL_reward(self, prev_state, action, current_state, eta):
        return (PnL_reward(prev_state, action, current_state) - np.max(0,eta * AgentCashHolding(prev_state, current_state)))
    

    # Given a state and an action, return the next state
    def transition(self, current_state, action):
        next_state = current_state.copy()
        order_size = action['order_size']
        theta = action['order_side']
 

        # Place order
        if order_side == 0:
            next_state['order_book'][self.t,1] += order_size[0]
        else:
            next_state['order_book'][self.t,3] += order_size[1]

        # Update time
        self.t += 1
        
        return next_state
    

    def step(self, action):
        # Get action from agent
        order_size = action['order_size']   
        order_side = action['order_side']
        action_type = action['action_type']

        # Compute market features
        current_state = self._get_obs()

        # Perform action
        next_state = self.transition(current_state, action)

        # Compute reward
        reward = self.reward(current_state, action, next_state)

        # Check if episode is over
        done = False
        if self.t == self.lob_data.shape[0]:
            done = True

        return next_state, reward, done, {}
    


    def render(self, mode='human'):
        pass

    def close(self):
        pass


