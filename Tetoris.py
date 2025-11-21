#!/usr/bin/env python
# coding: utf-8

# # Teto

# ## Environment Definition

# In[ ]:


import time
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


# In[ ]:


import gymnasium as gym
import sys
from gymnasium import spaces
import numpy as np

from collections import deque
import time



class TetrisEnv(gym.Env):

    def __init__(self):
        super(TetrisEnv, self).__init__()
        #self.action_space = spaces.Discrete(7)
        self.action_space = spaces.Discrete(4)   # Simplified Controls

        # Observation space:
        # Grid:               10 column x 20 row bits, one per cell
        # Placing Piece Grid: identical to Grid above
        # Piece Type:         7 bits, 1-hot
        # Rotation:           4 bits, 1-hot
        # Position:           x and y position of
        # Held Piece:         8 bits, 1-hot
        # Held this turn:     1 bit
        # Next Pieces:        3 x 7 bits, 1-hot each
        # Incoming:           # of rows
        
        #self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(477,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(400,), dtype=np.float32) # Minimal observation space
        
        self.reward = 0.0
        self.firstGame = True
        self.terminated = False
        self.truncated = False

        self.board_arr = None
        self.placing_board_arr = None
        self.placing_onehot = None
        self.rotation_onehot = None
        self.pos = None
        self.held_onehot = None
        self.hasheld_bit = None
        self.next_pieces_threehot = None
        self.incoming_severity = None

        self.prev_board_arr = None
        self.prev_placing_board_arr = None
        self.prev_placing_onehot = None
        self.prev_rotation_onehot = None
        self.prev_pos = None
        self.prev_held_onehot = None
        self.prev_hasheld_bit = None
        self.prev_next_pieces_threehot = None
        self.prev_incoming_severity = None

        self.stepend_timestamp = None
        self.stepbegin_timestamp = None
        
        self.stdioend_timestmap = None
        self.stdiobegin_timestamp = None

        self.renderend_timestamp = None
        self.renderbegin_timestamp = None


    def save_previous_observation_components(self):
        self.prev_board_arr = self.board_arr
        self.prev_placing_board_arr = self.placing_board_arr
        self.prev_placing_onehot = self.placing_onehot
        self.prev_rotation_onehot = self.rotation_onehot
        self.prev_pos = self.pos
        self.prev_held_onehot = self.held_onehot
        self.prev_hasheld_bit = self.hasheld_bit
        self.prev_next_pieces_threehot = self.next_pieces_threehot
        self.prev_incoming_severity = self.incoming_severity
    


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


        

    def parse_observations(self):

        # Board Input
        board = input().split()
        self.renderend_timestamp = time.perf_counter()
        render_time = self.renderend_timestamp - self.renderbegin_timestamp
        print(f"Render took about: {(render_time*1000):.4f} ms")
        
        board_ints = [int(board[i+1]) for i in range(20)]
        #print(board_ints, flush=True)
        arr = np.zeros((20, 10), dtype=np.float32)
        for y, row_val in enumerate(board_ints):
            bits = np.binary_repr(row_val, width=10)
            arr[y] = np.array(list(bits), dtype=np.float32)
        self.board_arr = arr.flatten()
        #print(self.board_arr, flush=True)

        # placing_Board Input
        placing_board = input().split()
        placing_board_ints = [int(placing_board[i+1]) for i in range(20)]
        #print(placing_board_ints, flush=True)
        placing_arr = np.zeros((20, 10), dtype=np.float32)
        for y, row_val in enumerate(placing_board_ints):
            placing_bits = np.binary_repr(row_val, width=10)
            placing_arr[y] = np.array(list(placing_bits), dtype=np.float32)
        self.placing_board_arr = placing_arr.flatten()
        #print(self.board_arr, flush=True)

        # Placing Input
        placing = input().split()
        self.placing_onehot = np.zeros(7, dtype=np.float32)
        match placing[1]:
            case "I": self.placing_onehot[0] = 1
            case "J": self.placing_onehot[1] = 1
            case "L": self.placing_onehot[2] = 1
            case "O": self.placing_onehot[3] = 1
            case "S": self.placing_onehot[4] = 1
            case "T": self.placing_onehot[5] = 1
            case "Z": self.placing_onehot[6] = 1
        #print(self.placing_onehot, flush=True)

        # Rotation Input
        rotation = input().split()
        self.rotation_onehot = np.eye(4)[int(rotation[1])]
        #print(self.rotation_onehot)

        # Position Input
        position = input().split()
        posx = np.eye(10)[int(position[1])]
        posy = np.eye(25)[min(24,int(position[2]))]
        #print(posx)
        #print(posy)
        self.pos = np.concatenate([posx, posy])
        #print(self.pos, flush=True)

        # Held Input
        held = input().split()
        self.held_onehot = np.zeros(8, dtype=np.float32)
        match held[1]:
            case "I": self.held_onehot[0] = 1
            case "J": self.held_onehot[1] = 1
            case "L": self.held_onehot[2] = 1
            case "O": self.held_onehot[3] = 1
            case "S": self.held_onehot[4] = 1
            case "T": self.held_onehot[5] = 1
            case "Z": self.held_onehot[6] = 1
            case "0": self.held_onehot[7] = 1
        #print(self.held_onehot, flush=True)

        # Has Held Input
        hasheld = input().split()
        self.hasheld_bit = np.array([1.0 if hasheld[1]=="true" else 0.0], dtype=np.float32)
        #print(self.hasheld_bit, flush=True)

        # Next Pieces Input
        self.next_pieces_threehot = np.zeros((3,7), dtype=np.float32)
        nextPieces = input().split()
        for i in range(3):
            match nextPieces[1+i]:
                case "I": self.next_pieces_threehot[i,0] = 1
                case "J": self.next_pieces_threehot[i,1] = 1
                case "L": self.next_pieces_threehot[i,2] = 1
                case "O": self.next_pieces_threehot[i,3] = 1
                case "S": self.next_pieces_threehot[i,4] = 1
                case "T": self.next_pieces_threehot[i,5] = 1
                case "Z": self.next_pieces_threehot[i,6] = 1
        self.next_pieces_threehot = self.next_pieces_threehot.flatten()
        #print(self.next_pieces_threehot, flush=True)

        # Incoming Severity Input
        incoming = input().split()
        self.incoming_severity = np.array([min(int(incoming[1]), 20) / 20.0], dtype=np.float32)
        #print(self.incoming_severity, flush=True)

        """
        self.observation = np.concatenate([
            self.board_arr,          # 0:199
            self.placing_board_arr,  # 200:399
            self.placing_onehot,     # 400:406
            self.rotation_onehot,
            self.pos,
            self.held_onehot,
            self.hasheld_bit,
            self.next_pieces_threehot,
            self.incoming_severity
        ])
        """
        #print("Observation:")
        #print(self.observation, flush=True)

        # Simplified Observation Set
        self.observation = np.concatenate([
            self.board_arr,          # 0:199
            self.placing_board_arr  # 200:399
        ])

        return True

    

    



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
        self.stepbegin_timestamp = time.perf_counter()

        if self.stepend_timestamp != None:
            step_interim = self.stepbegin_timestamp - self.stepend_timestamp
            print(f"Next step starting after interim of: {(step_interim*1000):.4f} ms (since previous step ended)")
        
        print("Stepping!", flush=True)

        
        self.stdiobegin_timestamp = time.perf_counter()
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
            
            self.renderbegin_timestamp = time.perf_counter()
            print("ready", flush=True)
            strin = input()    # receive ack
            print(strin, flush=True)
            
            
            self.save_previous_observation_components()
            self.previous_observation = self.observation
            self.parse_observations()

            self.stdioend_timestamp = time.perf_counter()
            stdio_time = self.stdioend_timestamp - self.stdiobegin_timestamp
            print(f"Doing the important stdio things took: {(stdio_time*1000):.4f} ms")
            


            # If a piece was placed
            piece_changed = not np.array_equal(self.next_pieces_threehot, self.prev_next_pieces_threehot)
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

        self.stepend_timestamp = time.perf_counter()
        step_time = self.stepend_timestamp - self.stepbegin_timestamp
        print(f"Full step took: {(step_time*1000):.4f} ms")

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

        self.renderbegin_timestamp = time.perf_counter()
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
# 
# (Overridden to be minimal, CNN-only)

# In[ ]:


import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TetrisFeatureExtractor(BaseFeaturesExtractor):
    """
    Uses ONLY the 2-channel CNN input (20x10x2 board).
    """

    def __init__(self, observation_space, features_dim: int = 256):
        # We will overwrite features_dim later after computing cnn_out_size
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

        # compute flattened CNN output size
        with th.no_grad():
            dummy_input = th.zeros((1, 2, 20, 10))
            cnn_out_size = self.board_cnn(dummy_input).shape[1]

        # Replace the final layer to output desired feature dimension
        self.final = nn.Sequential(
            nn.Linear(cnn_out_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Update features_dim so SB3 knows the output shape
        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Extract only the board channels (2 * 200 = 400 values)
        board_data = observations[:, :400].reshape((-1, 2, 20, 10))

        board_features = self.board_cnn(board_data)
        return self.final(board_features)


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


cfg_path = 'model.cfg'

# If model config file is empty, initialize empty model. Otherwise, load model
if os.path.getsize(cfg_path) == 0:
    print("Initializing untrained model")
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
else:
    try:
        with open(cfg_path, 'r') as file:
            content = file.read()
            content = content.strip()
    except FileNotFoundError:
        print(f"Error: Config file {cfg_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    try: 
        print("Loading trained model from " + content)
        model = PPO.load(content, env=env, device="cuda")
    except FileNotFoundError:
        print(f"Error: Model {content} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


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

