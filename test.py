import os

if __name__ == "__main__":
    x = 
    l = eval(f"[{os.environ['CUDA_VISIBLE_DEVICES']}]")
    print(type(l))
    print(type(l[0]))


"""
CUDA_VISIBLE_DEVICES=0,1 python test.py
"""