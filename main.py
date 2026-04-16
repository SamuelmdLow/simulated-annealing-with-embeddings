# https://www.sbert.net/examples/sentence_transformer/applications/image-search/README.html
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageDraw
import sys
import random
import os
import math
import copy
import statistics
import asyncio
import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt

# Colours
white = (255, 255, 255)
black = (0, 0, 0)

# Scoring systems
class ScoringSystem():
    def score_image(self, image):
        return 0

class EmbeddingsScoring():
    def __init__(self, model): 
        # Load CLIP model
        self.model = SentenceTransformer(model)
        self.goal = None
        
    def set_goal_text(self, text):
        # Encode text descriptions
        self.goal = self.model.encode([text])

    def set_goal_image(self, image):
        # Encode image
        self.goal = self.model.encode(image)

    def score_image(self, image):
        # Compare image embedding to goal
        img_emb = self.model.encode(image)

        similarity_score = self.model.similarity(img_emb, self.goal).tolist()[0]

        return similarity_score[0]

    def score_images(self, images):
        # Compare multiple images to goal
        images = self.model.encode(images)
        similarity_score = self.model.similarity(images, self.goal).tolist()

        return similarity_score
    
    def compare_images_to_image(self, images, image):
        image = self.model.encode(image)
        images = self.model.encode(images)
        similarity_score = self.model.similarity(image, images).tolist()

        return similarity_score

    def compute_sample_sim_means(self, samples, sample_size):
        start = time.time()
        flat_list = []
        for sample in samples:
            flat_list.append(sample[0])
            flat_list.extend(sample[1])

        encodings = self.model.encode(flat_list, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        encodingTime = time.time()

        means = []
        setLen = sample_size + 1
        for i in range(len(samples)):
            index = i * setLen

            baseEncoding = encodings[index]
            sampleEncoding = encodings[index + 1: index+setLen]
            means.append(np.mean(self.model.similarity(baseEncoding, sampleEncoding).tolist()[0]))
        simTime = time.time()

        print(f"encoding: {encodingTime-start}, sim: {simTime-encodingTime}")
        
        return means



# Mutation strategies
class MutationStrategy():
    def __init__(self, size):
        pil_image = Image.new(mode="RGB", size=(size, size))
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle([(0, 0), (pil_image.width - 1, pil_image.height - 1)], black)
        self.baseImage = pil_image
        self.size = size

        self.representation = None
        self.best_representation = None

    def mutate_image(self, temp):
        return self.baseImage # Placeholder

    def render_image(self):
        return self.baseImage

class RandomPixelFlipStrategy(MutationStrategy):
    def __init__(self, size):
        super().__init__(size)
        self.representation = self.baseImage
        self.best_representation = self.baseImage

    def mutate_image(self, temp):
        if temp > 0:
            image = self.representation.copy()
            px = image.load()
            
            for i in range(max(1, int(self.size**2 * (0.01+ 0.99*temp* random.random())))):
                x = random.randint(0, image.width - 1)
                y = random.randint(0, image.height - 1)
                self.flip_pixel(px, x, y)
            
            self.representation = image
    
    def flip_pixel(self, px, x, y):
        if px[x,y] == black:
            px[x,y] = white
        else:
            px[x,y] = black

    def render_image(self):
        return self.representation

class Colour():
    def __init__(self):
        self.pallet = []

    def randomFromPallet(self):
        if len(self.pallet) <= 0:
            self.newColour()
            return 0
        return random.randint(0, len(self.pallet)-1)

    def newColour(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        new = (r, g, b)
        self.pallet.append(new)
        return len(self.pallet) - 1
    
    def mergeColours(self, blobs):
        if len(self.pallet) < 2:
            return
        first = random.randint(0, len(self.pallet) - 1)
        second = first
        while first == second:
            second = random.randint(0, len(self.pallet) - 1)
        if first > second:
            temp = first
            first = second
            second = temp

        for blob in blobs:
            if blob.colour == second:
                blob.colour = first
            elif blob.colour == len(self.pallet) - 1:
                blob.colour = second
        self.pallet[second] = self.pallet[len(self.pallet) - 1]
        self.pallet.pop()
        return self.pallet[first]

    def mutateColour(self, colourIndex):
        colour = self.pallet[colourIndex]
        r = min(int(colour[0] * (random.random() + 0.5)), 255)
        g = min(int(colour[1] * (random.random() + 0.5)), 255)
        b = min(int(colour[2] * (random.random() + 0.5)), 255)
        self.pallet[colourIndex] = (r, g, b)
        return colourIndex

    def mutatePallet(self, blobs):
        rand = random.random()
        blob = blobs[random.randint(0, len(blobs)-1)]
        if rand < 0.2 or len(self.pallet) <= 0:
            blob.colour = self.newColour()
        elif rand < 0.4 and len(self.pallet) >=2:
            self.mergeColours(blobs)
        else:
            palletIndex = random.randint(0, len(self.pallet) - 1)
            self.mutateColour(palletIndex)
        print(self.pallet)

class Blob():
    def __init__(self, image_size):
        self.x = image_size // 2
        self.y = image_size // 2
        self.radius = 1
        self.colour = None
        self.tempAdjust = 1
        
        self.image_size = image_size

        self.previous = None

    def __str__(self):
        return f"[{self.x}, {self.y}], radius: {self.radius}"

    def random_mutate(self, temp):
        temp = temp * self.tempAdjust
        step_size = max(1, int((self.image_size/2) * temp))

        if self.x <= 0:
            self.x = self.x + random.randint(0, step_size)
        elif self.x >= self.image_size -1:
            self.x = self.x + random.randint(-step_size, 0)
        else:
            self.x = self.x + random.randint(-step_size, step_size)

        if self.y <= 0:
            self.y = self.y + random.randint(0, step_size)
        elif self.y >= self.image_size -1:
            self.y = self.y + random.randint(-step_size, 0)
        else:
            self.y = self.y + random.randint(-step_size, step_size)

        if self.radius <= 1:
            self.radius = self.radius + random.randint(0, step_size)
        elif self.radius > self.image_size // 4:
            self.radius = self.radius + random.randint(-step_size, 0)
        else:
            self.radius = self.radius + random.randint(-step_size, step_size)

    def rotate(self, rotation):
        opposite = self.x - (self.image_size/2)
        adjacent = self.y - (self.image_size/2)
        
        hypotonuse = math.sqrt(opposite**2 + adjacent**2)

        if adjacent == 0:
            if opposite > 0:
                angle = math.pi/2
            else:
                angle = -math.pi/2
        else:
            angle = math.atan(opposite/adjacent)

        newAngle = angle + rotation

        self.x = hypotonuse * math.sin(newAngle)
        self.y = hypotonuse * math.cos(newAngle)

    def effect(self, x, y):
        dist = (x-self.x)**2 + (y-self.y)**2
        if dist == 0:
            return 1000
        return self.radius**2 / ((x-self.x)**2 + (y-self.y)**2)

    def addBlobToImage(self, image, colours=None):
        px = image.load()
        for px_x in range(image.width):
            for px_y in range(image.height):
                dist = math.pow(px_x - self.x, 2) + math.pow(px_y - self.y, 2)
                if dist < math.pow(self.radius, 2):

                    if colours:
                        px[px_x, px_y] = colours.pallet[self.colour]
                    else:
                        px[px_x, px_y] = black

class MoveBlobsStrategy(MutationStrategy):

    def __init__(self, size, blob_count, colour, recenter=False):
        super().__init__(size)
        self.representation = []
        self.best_representation = []
        self.colour = colour
        self.recenter = recenter
        
        for i in range(blob_count):
            self.add_blob()

    def add_blob(self):
        self.representation.append(Blob(self.size))

    def recenter_blobs(self):
        minX = min([blob.x - blob.radius for blob in self.representation])
        maxX = max([blob.x + blob.radius for blob in self.representation])
        minY = min([blob.y - blob.radius for blob in self.representation])
        maxY = max([blob.y + blob.radius for blob in self.representation])

        #offsetX = (sum(blob.x for blob in self.representation) / len(self.representation)) - (self.size/2)
        #offsetY = (sum(blob.y for blob in self.representation) / len(self.representation)) - (self.size/2)

        width = maxX-minX
        height = maxY-minY
        offsetX = minX + width/2 - (self.size/2)
        offsetY = minY + height/2 - (self.size/2)

        for blob in self.representation:
            blob.x -= offsetX
            blob.y -= offsetY

        length = max(width, height)
        if length > self.size:
            scale = self.size/length

            for blob in self.representation:
                blob.x = self.size/2 + (blob.x - self.size/2) * scale
                blob.y = self.size/2 + (blob.y - self.size/2) * scale
                blob.radius = blob.radius * scale

    def rotate_blobs(self, rotation):
        for blob in self.representation:
            blob.rotate(rotation)

    def flip_blobs_x(self):
        for blob in self.representation:
            blob.x = -1 * blob.x

    def flip_blobs_y(self):
        for blob in self.representation:
            blob.y = -1 * blob.y

    def mutate_image(self, temp):
        if temp > 0:
            for blob in self.representation:
                blob.random_mutate(temp)

            if random.random() < temp:
                choice = random.random()
                if choice < 0.5:
                    rotation = 2*math.pi*(random.random()-0.5)
                    self.rotate_blobs(rotation)
                elif choice < 0.75:
                    self.flip_blobs_x()
                else:
                    self.flip_blobs_y()

            if self.recenter:
                self.recenter_blobs()

    def render_image(self):
        '''
        image = self.baseImage.copy()
        px = image.load()
        
        for px_y in range(image.height):
            for px_x in range(image.width):

                dist = sum(blob.effect(px_x, px_y) for blob in self.representation)

                if dist > 1:
                    px[px_x, px_y] = self.colour

        return image
        '''

        base = np.asarray(self.baseImage)

        xs = np.arange(self.size)
        ys = np.arange(self.size)

        X = xs[:, None]
        Y = ys[None, :]

        effects = np.zeros((self.size, self.size))
        for blob in self.representation:
            dists = (X-blob.x) ** 2 + (Y-blob.y) ** 2
            dists = np.maximum(dists, 1e-4)
        
            effects += blob.radius**2 / dists
        
        cells = np.where(effects[..., None] > 1, self.colour, base).astype(np.uint8)

        img = Image.fromarray(cells)

        return img


# Local search methods
class LocalSearchMethod():
    def __init__(self, mutationStrategy, scoringSystem):
        self.mutationStrategy = mutationStrategy
        self.scoringSystem = scoringSystem

        self.score = 0
        self.best_score = 0

        self.history = []
        self.best_history = []
        self.best_representation = None

    def update_best(self, representation=None, score=None, image=None, image_path=None):
        if score == None:
            score = self.score

        if score >= self.best_score:
            if representation == None:
                representation = copy.deepcopy(self.mutationStrategy.representation)
            if image == None:
                image = self.mutationStrategy.render_image()

            self.best_representation = representation
            self.best_score = score
            self.best_history.append(image)

            if image_path:
                image.save(f"{image_path}/best.png","PNG")

    def save_history(self, path):

        self.history[0].save(f"images/{path}/GIF.gif", 
                save_all = True, append_images = self.history[1:], 
                optimize = False, duration = 10) 
        
        self.best_history[0].save(f"images/{path}/bestGIF.gif", 
                save_all = True, append_images = self.best_history[1:], 
                optimize = False, duration = 10)

    def search(self, image_path=None):
        pass

class SimulatedAnnealing(LocalSearchMethod):
    def __init__(self, mutationStrategy, scoringSystem):
        super().__init__(mutationStrategy, scoringSystem)

    def search(self, image_path=None, alpha=0.95, initial_temp=1):
        MIN_DELTA = 0.005
        MAX_NO_CHANGE = 15
        MIN_TEMP = 0.0000001

        if image_path:
            os.makedirs(image_path, exist_ok=True)        

        temp = initial_temp

        mutationStrategy = self.mutationStrategy
        image = mutationStrategy.render_image()
        self.score = self.scoringSystem.score_image(image)
        self.update_best(image=image, image_path=image_path)

        initial_score = self.best_score

        no_change_count = 0
        i = 0

        gif = []

        #sum(blob.temp for blob in mutationStrategy.representation) > 0.0000001
        while temp > MIN_TEMP:         
            previousRepresentation = copy.deepcopy(mutationStrategy.representation)

            mutationStrategy.mutate_image(temp)

            score = self.scoringSystem.score_image(mutationStrategy.render_image())

            score_dif = 100 * (score - self.score)

            if score_dif < -MIN_DELTA and temp > 0:
                # accept worse random step
                random_prob = math.exp(score_dif/temp)
            else:
                # optimum step
                random_prob = 1
            
            if random.random() < random_prob or no_change_count > MAX_NO_CHANGE :
                # Take step
                self.score = score
                no_change_count = 0
            else:
                # Revert step
                mutationStrategy.representation = previousRepresentation
                no_change_count = no_change_count + 1

            temp = temp * alpha

            i = i + 1

            image = mutationStrategy.render_image()
            self.history.append(image)
            
            if self.score > self.best_score:
                self.update_best(image=image, image_path=image_path)
                print(f"{i} {self.best_score}")
            
            if image_path:
                #image.save(f"{image_path}/{i}.png","PNG")
                gif.append(image)

        mutationStrategy.representation = copy.deepcopy(self.best_representation)
        

        print(f'{round(((self.best_score-initial_score)/initial_score) * 100, 4)}% improvement\n  - {initial_score}\n  - {self.best_score}')

        if image_path:
            gif[0].save(f"{image_path}/GIF.gif", 
                        save_all = True, append_images = gif[1:], 
                        optimize = False, duration = 10) 

def generate_mutation(baseMutationStrategy, temp):
    #start = time.time()
    ms = copy.deepcopy(baseMutationStrategy)
    #copyTime = time.time()
    ms.mutate_image(temp)
    #mutateTime = time.time()
    image = ms.render_image() 
    #imageTime = time.time()
    #print(f"copy: {copyTime - start}, mutate: {mutateTime - copyTime}, image: {imageTime - mutateTime}")
    return image

def generate_sample(mutationStrategy, sample_size):
    start = time.time()
    baseMutationStrategy = copy.deepcopy(mutationStrategy)
    baseMutationStrategy.mutate_image(1)
    baseImage = baseMutationStrategy.render_image()
    
    
    images = [generate_mutation(baseMutationStrategy, temp) for s in range(sample_size)]

    return [baseImage, images]

def sample_distances(mutationStrategy, scoreSystem, samples=1000, sample_size = 100, temp=0.01):
    start = time.time()

    samples = [generate_sample(mutationStrategy, sample_size) for sample in range(samples)]
    sampleTime = time.time()

    means = scoreSystem.compute_sample_sim_means(samples, sample_size)
    distTime = time.time()

    print(f"sample: {sampleTime-start}s, dis: {distTime-sampleTime}s")

    return means


def alternate_blobs_pixels(scoreSystem, imageDir, iterations):
    blobsStrategy = MoveBlobsStrategy(128, 3, white)
    pixelsStrategy = RandomPixelFlipStrategy(128)
    
    localSearch = SimulatedAnnealing(copy.deepcopy(pixelsStrategy), scoreSystem)

    for i in range(iterations):
        localSearch.search(f"images/{imageDir}/{i}/pixels", initial_temp=1/(i+1))

        blobsStrategy.colour = [white,black][i%2]
        blobsStrategy.baseImage = localSearch.best_history[-1]
        localSearch.mutationStrategy = copy.deepcopy(blobsStrategy)

        localSearch.search(f"images/{imageDir}/{i}/influence")

        pixelsStrategy.representation = localSearch.best_history[-1]
        localSearch.mutationStrategy = copy.deepcopy(pixelsStrategy)

        localSearch.save_history(f"{imageDir}/")


if __name__ == '__main__':
    command = sys.argv[1]

    if command == "anneal":
        # Compute similarities

        options = ["alternate", "increasing-blob", "blob", "pixel"]

        if sys.argv[2] in options:
            prompt = sys.argv[3]
            imageDir = prompt

            # Path
            path = os.path.join("images", imageDir)
            
            # Create the directory
            os.makedirs(path, exist_ok=True)
            
            scoreSystem = EmbeddingsScoring("clip-ViT-B-32")
            scoreSystem.set_goal_text(prompt)

            if sys.argv[2] == "alternate":
                alternate_blobs_pixels(scoreSystem, imageDir, 10)
            elif sys.argv[2] == "increasing-blob":
                iterations = int(sys.argv[4])
                groupSize = int(sys.argv[5])
                mutationStrategy = MoveBlobsStrategy(128, groupSize, white, recenter=True)                
                localSearch = SimulatedAnnealing(copy.deepcopy(mutationStrategy), scoreSystem)

                for i in range(iterations):
                    localSearch.search(alpha=1 - 0.1/(i+1), initial_temp=1 - (1/(i+1)**2))
                    localSearch.save_history(f"{imageDir}/")
                    
                    for blob in localSearch.mutationStrategy.representation:
                        blob.tempAdjust = blob.tempAdjust * 0.75

                    for g in range(groupSize):
                        localSearch.mutationStrategy.add_blob()
                    
            else:
                if sys.argv[2] == "blob":
                    blobCount = 3
                    if len(sys.argv) > 4:
                        blobCount = int(sys.argv[4])
                    mutationStrategy = MoveBlobsStrategy(128, blobCount, white, recenter=True)
                elif sys.argv[2] == "pixel":
                    mutationStrategy = RandomPixelFlipStrategy(128)
                    
                localSearch = SimulatedAnnealing(mutationStrategy, scoreSystem)
                localSearch.search(alpha=0.99)

                localSearch.save_history(f"{imageDir}/")
        else:
            print("anneal options are: 'alternate', 'blob', 'pixel'")

    elif command == "measure_steps":
        scoreSystem = EmbeddingsScoring("clip-ViT-B-32")
        
        mutationStrategy = RandomPixelFlipStrategy(128)
        
        if sys.argv[2] == "blob":
            mutationStrategy = MoveBlobsStrategy(128, 3, white)

        means = []
        temp = 1

        xpoints = np.arange(21) * 0.05
        for temp in xpoints:
            
            sample = sample_distances(mutationStrategy, scoreSystem, samples=64, sample_size=1, temp=temp)
            means.append(statistics.mean(sample))
            print(f"{temp}) mean: {statistics.mean(sample)}, stdev: {statistics.stdev(sample)}")

        plt.plot(xpoints, means)
        plt.title(f"{sys.argv[2]} mutation strategy")
        plt.xlabel("Temperature")
        plt.ylabel("CLIP similarity")
        
        plt.savefig(f"{sys.argv[2]}_plot.png")
        plt.show()

        print(means)

    else:
        print("Options:\n   * anneal {alternate/increasing-blob/blob/pixel} {prompt}\n   * measure_steps {pixel/blob}")
