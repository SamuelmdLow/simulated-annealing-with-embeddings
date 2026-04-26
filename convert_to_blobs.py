import os
import numpy as np
from PIL import Image

from main import ScoringSystem, MutationStrategy, MoveBlobsStrategy, SimulatedAnnealing

class IntersectionOverUnion(ScoringSystem):
    def __init__(self, image=None):
        super().__init__()
        if image:
            self.set_target(image)    

    def set_target(self, image):
        self.target = np.asarray(image)
        print(self.target.shape)   

scoringSystem = IntersectionOverUnion()
directory = os.fsencode("data/mpeg-7")
    
for e in os.scandir(directory):
    if e.is_file():
        image = Image.open(e.path, "r")
        
        scoringSystem.set_target(image)
