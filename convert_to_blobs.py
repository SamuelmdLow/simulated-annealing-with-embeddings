import os
import numpy as np
from PIL import Image

from main import ScoringSystem, MutationStrategy, MoveBlobsStrategy, SimulatedAnnealing
from main import white, black

class IntersectionOverUnion(ScoringSystem):
    def __init__(self, image=None):
        super().__init__()
        if image:
            self.set_target(image)    

    def set_target(self, image):
        self.target = np.asarray(image)

    def score_image(self, image):
        image = np.asarray(image)
        reduced_image = np.where(np.all(image == [255, 255, 255], axis=-1), 255, 0)
        overlay = self.target + reduced_image
        
        intersection = np.sum(np.where(overlay > 255, 1, 0))
        union        = np.sum(np.where(overlay > 0, 1, 0))

        if union == 0:
            return 0
        return intersection / union

def blobDistance(blob1, blob2):
    radius = (blob1.radius - blob2.radius) ** 2
    position = (blob1.x - blob2.x) ** 2 + (blob1.y-blob2.y)**2
    return radius + position

def repDistance(rep1, rep2):
    distance = 0
    while len(rep1) > 0 and len(rep2) > 0:
        d = []
        indexes = []
        for blob in rep1:
            distances = map(lambda b: blobDistance(blob, b), rep2)
            d.append(min(distances))
            indexes.append(np.argmin(distances))

        distance += max(d)
        rep1.pop(np.argmax(d))
        rep2.pop(indexes[np.argmax(d)])

    return distance


scoreSystem = IntersectionOverUnion()
directory = os.fsencode("data/mpeg-7")

image_directory = "convert_to_blobs"
os.makedirs(f"images/{image_directory}", exist_ok=True)

names = []
representations = []

for e in os.scandir(directory):
    if e.is_file():
        filename = os.path.basename(e.path)
        name = filename.decode().split("-")[0]
        try:
            image = Image.open(e.path, "r")

            if name in names:
                continue
            
            scoreSystem.set_target(image)

            mutationStrategy = MoveBlobsStrategy(image.width, image.height, 5, white)
        
            localSearch = SimulatedAnnealing(mutationStrategy, scoreSystem)
            localSearch.search(alpha=0.995, satisfying_score=0.9, max_iterations=1)

            localSearch.best_history[-1].save(f"images/{image_directory}/{filename.decode()}", "PNG")

            representations.append(localSearch.best_representation)
            names.append(name)
        except:
            print(f"error: {filename}")

    if len(names) > 5:
        break

distanceMatrix = []
for rep1 in representations:
    distances = []
    for rep2 in representations:
        distances.append(repDistance(rep1, rep2))

    distanceMatrix.append(distances)

print(distanceMatrix)