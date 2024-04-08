from tqdm import tqdm
import time

with tqdm(total=10*300000) as pbar:
    for i in range(10):
        for j in range(300000):
            time.sleep(0.1)
            pbar.update(1)
