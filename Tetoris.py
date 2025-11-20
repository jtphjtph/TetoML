#!/usr/bin/env python
# coding: utf-8

# # Teto

# In[ ]:


import time
print("Hello World")
time.sleep(1.5)
teto = """                                          =**++**+****=+==.                                         
                                    .***************#**++++++==-                                    
                                  **#***##*#*++*****+**++++==+======.                               
                               :*#*#*##*****#*#******+++++++++++==+====:                            
                             .*####******************++++*+=+==+=+=======-                          
                            #%%#*****+***#**********+*++++==+=+===+========:                        
                          #%%*#*#*##***********=++**+**+*+++=+++==+====---===                       
                        :#**###***********+*++*=**+*++++***=++====+==========-                      
                       +********+***+***********+****+++++==+++==============-:                     
                .==**#++*#*********************+**+*+++*++++++++===========-=---                    
         .=++*=++***#%+#********+*+***+********+****====+==================-==---==---:.            
    +*++++*++**++**%%****+********+***+****+*++*****++*++=================----==--*+*+=====-=---:   
    ****+==+**++**%%%%************+***++**++==*+**+*++*++==++====+========--==----********+==-==-:: 
    **++=+=++==***%%*********+++++++*++=++=+++++++*+**++===================-=----:+***+===++===--:  
     #*++#**=*+=+*#%*#*****+***+**=**+=++++=+++=+=*+==++=+================-==-----*#*++++=+===+=-.  
   =+*+=++**#%**++#**#******+*+=***+++=+*+=++++++***++++++#================-----=-:*=+====--===-==  
  *+*+++++=**##%@@%******+*****+***+**+++++==#=+********+=-*=+============-----:--:-*+*++===+===--  
 .***+=======**%%@@.******************+**+++=****+*++**+++.:*=============----------==***#=====+=:  
 .****+===-+**#%%@@@+**********+*****+****+:=+*++++*=*+**:..:+*=============------:@%%##*===:==--=  
  +**===+=++***%%@@%*****+***************+::=+*++*+++==++....:=#=======-===-=--=--*@%#*+=+===::==-  
   +++=====+++#@@@@*+******************+=:::=+*=++++====-.. ...-*-======----=--=--:%#**=========--  
      %%%%%@#*#%%%#*#****************++::%#-=+*+++++==+=... .:*+:*====---=--====--::#%*==-=====--   
     +***+++***#%%*********#*********=:::::-=%*=++++==+.....=.....-=-==--=-=====---#%%#*==*+*==:    
     ***==+=****#%%%@*********#*****+:::::-=+*+=++++==.............:*--======---%%#%#*+=++*====     
      **+===****#%@@@**************%%##%%%++%*++**++=.......*.+**#***#%---==---:%%###*+**+*=+=-.    
        *++===*+*%%%@*******+****%%%%-::=%@@%+=+++==........%#*%.  #*%*=---=----**%%##***+=====     
        =++*****#%%%%*******+++*@#%@@.::=@%%*=#++=......... .*#=..:%%-#@..%=#==****+++*#***+==.     
         +====++*##%#%#++***=*=-@-@@@@::@@%%==+=............***%..#%%:*..*-:%##*#*#****+====:       
                      *=:%%%=%=-=-%@@@@%%@%%*:..............*#%*#%%%%...:*-    .:**##***==:         
                          %%**+-:::%%%%%%%#%+::.............*=+*****....%-                          
                          %%#+#--:::%%%%#***:..:.............*==+*#....%=                           
                          .###=*=-::::::=::::.........................%-                            
                            -##=%*=::::::::.::......................:*=.                            
                              :+*##*:-:::::::::..@.:..=#:.........=====--.                          
                           .===----:#+==--::::::::::::::::.::::++==*%-:::+*#%*                      
                        .%%#*::::.:::#%%%%%%##*=-----:=***=*****+:::-:..=**%%%%#@#                  
                     @%%@%##*+-....::*%%%%%##**###******++******+.:::::=**##%**@#@#*@#-             
                 +@%%@%@@#***+=:....:%%%%%%%%##**##%**==*****%%*#=*=-:+**%%%#%@#@**%*%#+*           
             ***#%#%@#********=+...:#@%%%%@@%#%***%*%*#*#%****%##%%@%#%#%%%%%%###@%**%*:.           
            :*=+**************#*:%@@@@@@@%%%#@#**++==+**%**%**@##%%@@%%@%%%%%%%%%#*****.            
             ---=+=+*+******%%%@@@@@@@@%%%%%%%##*+%*%****#****@##@@%%%%%   %%%%%%%##+* .            
             =:::===+***#%:  %@@@%@@@@@%@%%%@###*++++****%%%****@%#%%%%%      .%@%#**..             
              =:::===##-     %@@@%%%@@@@%%%%%%%##*@@+*****#%%%%%@%@@%###=         **.               
                ::=*+        %%@@@@@%%#%%@%%%%%%%*%%#%%@@@%%%*#*****%***#                           
                              @@@%%@@*%%@%@#%%%%%%%@@@@%##*#%%%@%%%%*%%#**+                         
                              %%@@@%%%@@#%%%%@%@@@@%%%%@@@@@@@%%#****#+*%#+*                        
                              +%%@*##%%%@@%@@@@@%%%@@@@@@@@@%%%@@@*@-                               
                              =@@%@@@###%%%%@@@%@%@@@@@@@@@%@@@@@%%@@%                              
                            .#****=++*@%*%%%%@%@@@@@@%@@%%@##***+==+#@                              
                           -#***++====++=:*##@@%%@@@@@@%: %##*++=====+:                             
                           %##**++=====++   %@%%%%@%%%%  #%#**++=====+%                             
                           %%#***+++====+.    +%%%%%-    %%%#**++===++*                             
                           #%##***++++++*                 %%%#**++++++*                             
                            %%##****++*+                  =%%##****+++                              
                             .%%#******                     -%%#*****                               
                                :%%%=                                                               """
print(teto)
time.sleep(1.5)


# ## Environment Definition

# In[ ]:


import gymnasium as gym
import sys
from gymnasium import spaces
import numpy as np

from collections import deque
import time



class TetrisEnv(gym.Env):

    def parse_observations(self):

        # Board Input
        board = input().split()
        board_ints = [int(board[i+1]) for i in range(20)]
        #print(board_ints, flush=True)
        arr = np.zeros((20, 10), dtype=np.float32)
        for y, row_val in enumerate(board_ints):
            bits = np.binary_repr(row_val, width=10)
            arr[y] = np.array(list(bits), dtype=np.float32)
        board_arr = arr.flatten()
        #print(board_arr, flush=True)

        # placing_Board Input
        placing_board = input().split()
        placing_board_ints = [int(placing_board[i+1]) for i in range(20)]
        #print(placing_board_ints, flush=True)
        placing_arr = np.zeros((20, 10), dtype=np.float32)
        for y, row_val in enumerate(placing_board_ints):
            placing_bits = np.binary_repr(row_val, width=10)
            placing_arr[y] = np.array(list(placing_bits), dtype=np.float32)
        placing_board_arr = placing_arr.flatten()
        #print(board_arr, flush=True)

        # Placing Input
        placing = input().split()
        placing_onehot = np.zeros(7, dtype=np.float32)
        match placing[1]:
            case "I": placing_onehot[0] = 1
            case "J": placing_onehot[1] = 1
            case "L": placing_onehot[2] = 1
            case "O": placing_onehot[3] = 1
            case "S": placing_onehot[4] = 1
            case "T": placing_onehot[5] = 1
            case "Z": placing_onehot[6] = 1
        #print(placing_onehot, flush=True)

        # Rotation Input
        rotation = input().split()
        rotation_onehot = np.eye(4)[int(rotation[1])]
        #print(rotation_onehot)

        # Position Input
        position = input().split()
        posx = np.eye(10)[int(position[1])]
        posy = np.eye(25)[min(24,int(position[2]))]
        #print(posx)
        #print(posy)
        pos = np.concatenate([posx, posy])
        #print(pos, flush=True)

        # Held Input
        held = input().split()
        held_onehot = np.zeros(8, dtype=np.float32)
        match held[1]:
            case "I": held_onehot[0] = 1
            case "J": held_onehot[1] = 1
            case "L": held_onehot[2] = 1
            case "O": held_onehot[3] = 1
            case "S": held_onehot[4] = 1
            case "T": held_onehot[5] = 1
            case "Z": held_onehot[6] = 1
            case "0": held_onehot[7] = 1
        #print(held_onehot, flush=True)

        # Has Held Input
        hasheld = input().split()
        hasheld_bit = np.array([1.0 if hasheld[1]=="true" else 0.0], dtype=np.float32)
        #print(hasheld_bit, flush=True)

        # Next Pieces Input
        next_pieces_threehot = np.zeros((3,7), dtype=np.float32)
        nextPieces = input().split()
        for i in range(3):
            match nextPieces[1+i]:
                case "I": next_pieces_threehot[i,0] = 1
                case "J": next_pieces_threehot[i,1] = 1
                case "L": next_pieces_threehot[i,2] = 1
                case "O": next_pieces_threehot[i,3] = 1
                case "S": next_pieces_threehot[i,4] = 1
                case "T": next_pieces_threehot[i,5] = 1
                case "Z": next_pieces_threehot[i,6] = 1
        next_pieces_threehot = next_pieces_threehot.flatten()
        #print(next_pieces_threehot, flush=True)

        # Incoming Severity Input
        incoming = input().split()
        incoming_severity = np.array([min(int(incoming[1]), 20) / 20.0], dtype=np.float32)
        #print(incoming_severity, flush=True)

        self.observation = np.concatenate([
            board_arr,          # 0:199
            placing_board_arr,  # 200:399
            placing_onehot,     # 400:406
            rotation_onehot,
            pos,
            held_onehot,
            hasheld_bit,
            next_pieces_threehot,
            incoming_severity
        ])
        #print("Observation:")
        #print(self.observation, flush=True)

        return True



    # By ChatGPT
    def count_enclosed_regions(self, board):
        """
        Counts the number of empty regions fully enclosed (orthogonally) by 1s 
        and not touching the top row.
        board: 2D numpy array of shape (20, 10), 0 = empty, 1 = filled
        """
        rows, cols = board.shape
        visited = np.zeros_like(board, dtype=bool)
        enclosed_count = 0
    
        # Directions for orthogonal neighbors: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
        for r in range(rows):
            for c in range(cols):
                if board[r, c] == 0 and not visited[r, c]:
                    # Start BFS for this empty region
                    queue = deque()
                    queue.append((r, c))
                    visited[r, c] = True
                    touches_top = (r == 0)
                    enclosed = True
    
                    while queue:
                        cr, cc = queue.popleft()
                        for dr, dc in directions:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if board[nr, nc] == 0 and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                            else:
                                # Reaching outside the board (shouldn't happen in Tetris), still enclosed
                                continue
                        if cr == 0:
                            touches_top = True
    
                    # Only count if region doesn't touch top
                    if not touches_top:
                        enclosed_count += 1
    
        return enclosed_count

    

    def __init__(self):
        super(TetrisEnv, self).__init__()
        #self.action_space = spaces.Discrete(7)
        self.action_space = spaces.Discrete(4)   # Simplified Controls

        # Observation space:
        # Grid: 10 column x 20 row bits, one per cell
        # Placing piece grid, see above
        # Piece Type: 7 bits, 1-hot
        # Rotation: 4 bits, 1-hot
        # Position: x and y position of
        # Held Piece: 8 bits, 1-hot
        # Held this turn: 1 bit
        # Next Pieces: 3 x 7 bits, 1-hot each
        # Incoming: # of rows
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(477,), dtype=np.float32)

        self.reward = 0.0
        self.firstGame = True
        self.terminated = False
        self.truncated = False



    """
    Currently Implemented Reward Factors:
    - Gameover:   -7.5
    - Default:    0
        - Placed:     += 0.5
        - Max Height: -= 0.1*(# of rows)
        - Overhang:   -= max(-0.75, 0.05 - (0.25 * piece_overhang)) # That is, 0.05 minus 0.25 per tile of overhang, minimum of -=0.75
        - Lines:      += 1*(# lines cleared)
        - Sent:       += 1*(# lines sent)
        - B2B:        += 1
        - Combo:      += 0.5*(Combo)
        - Invalids    -= 0.15
        - Repeats     -= 0.2

    Disabled:
    - Hole Creation:  -= 0.25
    """
    def step(self, action):
        print("Stepping!", flush=True)

        strin = input()    # receive move
        if strin == "move": print("Got move order")
        """
        # Full Action Set
        match action:
            case 0: print("move left", flush=True)
            case 1: print("move right", flush=True)
            case 2: print("move hard", flush=True)
            case 3: print("move soft", flush=True)
            case 4: print("move cw", flush=True)
            case 5: print("move ccw", flush=True)
            case 6: print("move hold", flush=True)
        """
        # Simplified Action Set
        match action:
            case 0: print("move left", flush=True)
            case 1: print("move right", flush=True)
            case 2: print("move cw", flush=True)
            case 3: print("move soft", flush=True)

        self.reward = 0

        strin = input()   # receive ack
        strin = input()   # receive "gameover" or first line of report (`lines`)
        if(strin == "gameover"):
            #print("Gameovered. Rip.")
            self.terminated = True
            self.reward -= 7.5                        # penalty for dying
            self.previous_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            self.observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            lines = int(strin.split()[1])
            sent = int(input().split()[1])
            b2b = True if input().split()[1]=="true" else False
            combo = int(input().split()[1])
            invalidmove = True if input().split()[1]=="true" else False
            repeatedmove = True if input().split()[1]=="true" else False
            print("ready", flush=True)
            strin = input()    # receive ack
            print(strin, flush=True)
            self.previous_observation = self.observation
            self.parse_observations()


            # If a piece was placed
            piece_changed = not np.array_equal(self.previous_observation[455:476], self.observation[455:476])
            #print(self.previous_observation[455:476])
            #print(self.observation[455:476])
            if piece_changed:
                prev_board = self.previous_observation[0:200].reshape((20, 10))
                curr_board = self.observation[0:200].reshape((20, 10))
                #print(prev_board, "\n")
                #print(curr_board)

                
                # Decide whether the piece was placed in a position that raises the max board height
                prev_rows_with_ones = np.where(prev_board.any(axis=1))[0]
                if len(prev_rows_with_ones) > 0:
                    prev_bottommost_row = prev_rows_with_ones[-1]
                else: prev_bottommost_row = 0;
                #print(f"prev Bottommost filled row index: {prev_bottommost_row}")

                curr_rows_with_ones = np.where(curr_board.any(axis=1))[0]
                if len(curr_rows_with_ones) > 0:
                    curr_bottommost_row = curr_rows_with_ones[-1]
                else: curr_bottommost_row = 0;
                #print(f"curr Bottommost filled row index: {curr_bottommost_row}")

                self.reward += 0.5    # Base reward for placing a piece (before poor placement penalties)
                
                # Penalize if the placed piece increased row height (don't penalize if no pieces placed yet
                if prev_bottommost_row != 0 and curr_bottommost_row > prev_bottommost_row:
                    penalty = ((curr_bottommost_row - prev_bottommost_row) * 0.1)
                    print("Board height penalty assigned: ", penalty)
                    self.reward -= penalty


                # Decide whether the piece was placed in a position that created more holes in the board
                prev_holes = self.count_enclosed_regions(prev_board)
                curr_holes = self.count_enclosed_regions(curr_board)
                #print("Previous enclosed regions: ", prev_holes)
                #print("Current enclosed regions: ", curr_holes)
                if curr_holes > prev_holes:
                    self.holes_created += 1
                    #print("Assigning hole-creation penalty")
                    #self.reward -= 0.25

                # Decide how much overhang the placed piece caused
                placed_coords = np.logical_xor(prev_board, curr_board).astype(int)
                #print("Placed coords:")
                #print(placed_coords)
                piece_overhang = 0
                for row, col in np.argwhere(placed_coords):
                    col_overhang = 0
                    while row > 0:
                        row -= 1
                        if curr_board[row,col] == 0:
                            piece_overhang += 1
                            col_overhang += 1
                        else: break
                    #print("Overhang in col ", col, "1: ", col_overhang)
                print("Piece Overhang: ", piece_overhang)
                overhang_reward = max(-0.75, 0.05 - (0.25 * piece_overhang))  # Reward overhang 0, neutral == 1, penalize > 1. Max penalty of -1
                print("  rew/pen=", overhang_reward)
                self.reward +=  overhang_reward   
                self.total_overhang += piece_overhang
                
                #time.sleep(1)
                
                

                self.reward += lines                          # +1 reward per line cleared
                self.reward += sent                           # +1 reward per line sent
                self.reward += (1 if b2b else 0)              # +1 reward if b2b'd
                self.reward += combo/2.0                      # +0.5 reward per combo
                self.reward -= (0.15 if invalidmove else 0.0)  # -0.1 reward if invalid move
                self.reward -= (0.2 if repeatedmove else 0.0) # -0.1 reward if repeated move (avoid, but if you need a repeated for some reason it's ok)
    
                #self.reward /= 5   # normalize, maybe
                
                if lines > 0:
                    self.lines_cleared += lines
                if invalidmove:
                    self.invalid_moves += 1
                if repeatedmove:
                    self.repeated_moves += 1
                if piece_changed:
                    self.pieces_placed += 1
            # To implement:
            # + reward for placing blocks without any gaps below them
            # - penalty for placing blocks with overhang (inverse of above)
            # + reward for filling partially-filled lines
            # + small reward for placing a piece (survival)
            #print("Step complete, rewards assigned.", flush=True)

    

        info = {"lines": self.lines_cleared,
                "pieces": self.pieces_placed,
                "repeats": self.repeated_moves,
                "invalids": self.invalid_moves,
                "holes_created": self.holes_created,
                "total_overhang": self.total_overhang}

        """
        # Override test: Sort out O-Pieces rightward
        if self.previous_observation[403]:
            if action == 1: self.reward = 5
            else: self.reward = -5
        else:
            if action == 0: self.reward = 5
            else: self.reward = -5
        """

        return self.observation, self.reward, self.terminated, self.truncated, info



    def reset(self, seed = None):
        print("Resetting!", flush=True)

        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.invalid_moves = 0
        self.repeated_moves = 0
        self.holes_created = 0
        self.total_overhang = 0


        #if not self.firstGame:
        #    print("First game, doing second options pass and waiting for ack", flush=True)
        #    print("options seed 1", flush=True)
        #    strin = input()    # Receive gameover
        #else: self.firstGame = False

        print("options seed 1", flush=True)

        print("ready", flush=True)
        strin = input()    # Receive ack
        #print(strin)

        self.previous_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.parse_observations()

        info = {}
        return self.observation, info


# ###### Environment Verification

# In[ ]:


#from stable_baselines3.common.env_checker import check_env
#env = TetrisEnv()


# In[ ]:


#check_env(env)


# ## Define CNN-MLP Hybrid Feature Extractor
# (Written by ChatGPT)

# In[ ]:


import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TetrisFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the Tetris environment.
    - CNN on 2-channel (20x10x2) board input:
        [0] placed blocks
        [1] currently falling piece
    - MLP on the 1D context vector
    - BatchNorm + Dropout regularization
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # -------------------------------
        # CNN branch: 2-channel 20x10 board
        # -------------------------------
        self.board_cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Flatten()
        )

        # dynamically determine flattened CNN output size
        with th.no_grad():
            dummy_input = th.zeros((1, 2, 20, 10))   # <-- now 2 channels
            cnn_out_size = self.board_cnn(dummy_input).shape[1]

        # -------------------------------
        # Context branch: everything except the 2*200 board cells
        # -------------------------------
        context_size = observation_space.shape[0] - 400
        self.context_mlp = nn.Sequential(
            nn.Linear(context_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # -------------------------------
        # Combine CNN and MLP features
        # -------------------------------
        combined_dim = cnn_out_size + 64
        self.final = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Extract the two 20x10 channels (0:200 = board, 200:400 = piece)
        board_data = observations[:, :400].reshape((-1, 2, 20, 10))
        context = observations[:, 400:]

        board_features = self.board_cnn(board_data)
        context_features = self.context_mlp(context)

        combined = th.cat([board_features, context_features], dim=1)
        return self.final(combined)


# ## Initialize Environment and Set Up Model

# In[ ]:


from stable_baselines3 import PPO
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = Monitor(TetrisEnv())
env = DummyVecEnv([lambda: env])
#env.reset()
print("Reset done!")

policy_kwargs = dict(
    features_extractor_class=TetrisFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[dict(pi=[128, 64], vf=[128, 64])]
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    device = "cuda",
    tensorboard_log=logdir,
    gamma=0.99,
    seed=0
)


# In[ ]:


print(model.policy)


# ## Define Logger Callback

# In[ ]:


from stable_baselines3.common.callbacks import BaseCallback

class LoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            infos = self.locals.get("infos", [])
            for info in infos:
                if "lines" in info:
                    self.logger.record("env/lines", info["lines"])
                if "pieces" in info:
                    self.logger.record("env/pieces", info["pieces"])
                if "invalids" in info:
                    self.logger.record("env/invalids", info["invalids"])
                if "repeats" in info:
                    self.logger.record("env/repeats", info["repeats"])
                if "holes_created" in info:
                    self.logger.record("env/holes_created", info["holes_created"])
                if "total_overhang" in info and info["pieces"] != 0:
                    self.logger.record("env/avg_overhang", info["total_overhang"]/info["pieces"])
        return True


# ## Train

# In[ ]:


import traceback

TIMESTEPS = 1e5;      # Train indefinitely, saving every 100k timesteps
iters = 0
callback = LoggerCallback()

try:
    while True:
        iters+=1
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            callback = callback,
            tb_log_name=f"TetoPPO",
            progress_bar=False)
        model.save(f"{models_dir}/{TIMESTEPS*iters}")
    print("Training complete, saving model.")
    model.save(f"{models_dir}/{TIMESTEPS}")
except Exception:
    print("Python error:")
    print(traceback.format_exc())

