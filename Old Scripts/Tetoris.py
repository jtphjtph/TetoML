import gymnasium as gym
import sys
from gymnasium import spaces
import numpy as np





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
        posy = np.eye(24)[int(position[2])]
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
            board_arr,
            placing_onehot,
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


    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.action_space = spaces.Discrete(7)

        # Observation space:
        # Grid: 10 column x 20 row bits, one per cell
        # Piece Type: 7 bits, 1-hot
        # Rotation: 4 bits, 1-hot
        # Position: x and y position of
        # Held Piece: 8 bits, 1-hot
        # Held this turn: 1 bit
        # Next Pieces: 3 x 7 bits, 1-hot each
        # Incoming: # of rows
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(276,), dtype=np.float32)

        self.reward = 0.0
        self.firstGame = True
        self.terminated = False
        self.truncated = False



    def step(self, action):
        print("Stepping!", flush=True)

        strin = input()    # receive move
        if strin == "move": print("Got move order")
        match action:
            case 0: print("move left", flush=True)
            case 1: print("move right", flush=True)
            case 2: print("move hard", flush=True)
            case 3: print("move soft", flush=True)
            case 4: print("move cw", flush=True)
            case 5: print("move ccw", flush=True)
            case 6: print("move hold", flush=True)

        self.reward = 0

        strin = input()   # receive ack
        strin = input()   # receive "gameover" or first line of report (`lines`)
        if(strin == "gameover"):
            #print("Gameovered. Rip.")
            self.terminated = True
            self.reward -= 10                        # -10 reward for dying
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


            piece_changed = not np.array_equal(self.previous_observation[254:275], self.observation[254:275])
            if piece_changed:
                self.reward += 0.25
                prev_board = self.previous_observation[0:200].reshape((20, 10))
                curr_board = self.observation[0:200].reshape((20, 10))
                #print(prev_board)
                #print(curr_board)

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

                if curr_bottommost_row > prev_bottommost_row:
                    penalty = ((curr_bottommost_row - prev_bottommost_row) * 0.25)
                    #rint("Assigning penalty for increasing row height: ", penalty)
                    self.reward -= penalty


            self.reward += lines                          # +1 reward per line cleared
            self.reward += sent                           # +1 reward per line sent
            self.reward += (1 if b2b else 0)              # +1 reward if b2b'd
            self.reward += combo/2.0                      # +0.5 reward per combo
            self.reward -= (2.5 if invalidmove else 0.0)  # -2.5 reward if invalid move
            self.reward -= (0.5 if repeatedmove else 0.0) # -0.5 reward if repeated move (avoid, but if you need a repeated for some reason it's ok)


            if lines > 0:
                self.lines_cleared += lines
            if invalidmove:
                self.invalid_moves += 1
            if repeatedmove:
                self.repeated_moves += 1
            if piece_changed:
                self.pieces_placed += 1
            # To implement:
            # - reward for creating holes
            # + reward for filling partially-filled lines
            # - reward for increasing board height? (not big)
            # + small reward for placing a piece (survival)
            #print("Step complete, rewards assigned.", flush=True)



        info = {"lines": self.lines_cleared,
                "pieces": self.pieces_placed,
                "repeats": self.repeated_moves,
                "invalids": self.invalid_moves}

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


from stable_baselines3 import PPO
import os
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

#env = DummyVecEnv([lambda: TetrisEnv()])
#env = Monitor(TetrisEnv(), logdir)
env = DummyVecEnv([lambda: TetrisEnv()])
env = VecMonitor(env, logdir)
#env.reset()
#print("Reset done!")

#policy_kwargs = dict(
#    net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]
#)

model = PPO(
    "MlpPolicy",
    env,
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

print(model.policy)


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
        return True


TIMESTEPS = 1e5;
iters = 0
callback = LoggerCallback()

while True: # Train indefinitely
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
