import time
#import fileinput
import sys

if __name__ == "__main__":
    moves = ['left', 'left', 'left', 'left', 'left', 'left', 'left', 'left']

    print("yeetus")
    print("option seed 1")
    print("ready")

    instr = input()



    x = 0
    y = 0
    while True:
        instr = input()
        if instr == "move":
            if x >= 8: print("move hard")
            else: print("move " + moves[x])
            x += 1
            while True:
                instr = input()
                print(instr)
                if("repeatedmove" in instr):
                    time.sleep(1)
                    print("ready")
                    break
                if(instr == "gameover"):
                    if(y<1):
                        print("option seed 1")
                        print("option garbage 1-5 2-6")
                        print("ready")
                        y+=1
                        x=0
                        break
                    else:
                        print("kill")
                        break
