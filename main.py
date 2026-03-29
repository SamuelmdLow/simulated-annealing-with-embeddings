# https://www.sbert.net/examples/sentence_transformer/applications/image-search/README.html
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageDraw
import sys
import random
import os
import math
import copy

# Colours
white = (255, 255, 255)
black = (0, 0, 0)

# Scoring systems
class ScoringSystem():
    def score_image(self, image):
        return 0

class EmbeddingsScoring():
    def __init__(self, model, text): 
        # Load CLIP model
        self.model = SentenceTransformer(model)
        
        # Encode text descriptions
        self.text_emb = self.model.encode([text])

    def score_image(self, image):
        img_emb = self.model.encode(image)

        similarity_score = self.model.similarity(img_emb, self.text_emb).tolist()[0]

        return similarity_score[0]

# Mutation strategies
class MutationStrategy():
    def __init__(self, size, scoring):
        pil_image = Image.new(mode="RGB", size=(size, size))
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle([(0, 0), (pil_image.width - 1, pil_image.height - 1)], black)
        self.baseImage = pil_image
        self.size = size

        self.representation = []
        self.best_representation = []

    def mutate_image(self, temp):
        return self.baseImage # Placeholder

    def render_image(self):
        return self.baseImage

    def cool():
        # for simulated annealing
        pass

class RandomPixelFlipStrategy(MutationStrategy):
    def mutate_image(self, temp):
        image = self.baseImage.copy()
        px = self.image.load()
        
        for i in range(self.size**2 * temp):
            x = random.randint(0, image.width - 1)
            y = random.randint(0, image.height - 1)
            self.flip_pixel(px, x, y)
        
        return image
    
    def flip_pixel(self, px, x, y):
        if px[x,y] == black:
            px[x,y] = white
        else:
            px[x,y] = black

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
    def __init__(self, image_size, temp):
        self.x = image_size // 2
        self.y = image_size // 2
        self.radius = 1
        self.colour = None

        self.temp = temp
        
        self.image_size = image_size

        self.previous = None

    def __str__(self):
        return f"[{self.x}, {self.y}], radius: {self.radius}"

    def random_mutate(self, temp):
        step_size = max(1, int((self.image_size/2) * (0.3 + 0.7 * temp)))

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

    def cool(self):
        self.temp = self.temp * 0.95

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

    def add_blob(self):
        self.representation.append(Blob(self.size, 1))

    def recenter_blobs(self):
        offsetX = (sum(blob.x for blob in self.representation) / len(self.representation)) - (self.size/2)
        offsetY = (sum(blob.y for blob in self.representation) / len(self.representation)) - (self.size/2)

        for blob in self.representation:
            blob.x -= offsetX
            blob.y -= offsetY

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

        self.recenter_blobs()

    def cool(self):
        for blob in mutationStrategy.representation:
            blob.cool()

    def render_image(self):
        image = self.baseImage.copy()
        px = image.load()
        
        for px_y in range(image.height):
            for px_x in range(image.width):

                dist = sum(blob.effect(px_x, px_y) for blob in self.representation)

                if dist > 1:
                    px[px_x, px_y] = white

        return image


# Local search methods
class LocalSearchMethod():
    def __init__(self, mutationStrategy, scoringSystem):
        self.mutationStrategy = mutationStrategy
        self.scoringSystem = scoringSystem

        self.score = 0
        self.best_score = 0

        self.history = []
        self.best_history = []

    def search(self, image_path=None):
        pass

class SimulatedAnnealing(LocalSearchMethod):
    def search(self, image_path=None):
        MIN_DELTA = 0.005
        MAX_NO_CHANGE = 15
        MIN_TEMP = 0.0000001

        temp = 1
        alpha = 0.95

        mutationStrategy = self.mutationStrategy
        mutationStrategy.best_representation = copy.deepcopy(mutationStrategy.representation)
        initial_score = self.score = self.scoringSystem.score_image(mutationStrategy.render_image())
        self.best_score = self.score

        no_change_count = 0
        i = 0

        gif = []
        if image_path:
            os.makedirs(image_path, exist_ok=True)        

        #sum(blob.temp for blob in mutationStrategy.representation) > 0.0000001
        while temp > MIN_TEMP:
            self.score = self.scoringSystem.score_image(mutationStrategy.render_image())
            
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

            mutationStrategy.cool()
            temp = temp * alpha

            i = i + 1

            image = mutationStrategy.render_image()
            self.history.append(image)
            
            if self.score > self.best_score:
                self.best_representation = copy.deepcopy(mutationStrategy.representation)
                self.best_score = self.score
                self.best_history.append(image)
                print(f"{i} {self.best_score}")
            
            if image_path:
                image.save(f"{image_path}/{i}.png","PNG")
                gif.append(image)

        mutationStrategy.representation = copy.deepcopy(mutationStrategy.best_representation)
        end_score = self.score = self.scoringSystem.score_image(mutationStrategy.render_image())

        print(f'{round(((end_score-initial_score)/initial_score) * 100, 4)}% improvement\n  - {initial_score}\n  - {end_score}')

        if image_path:
            gif[0].save(f"{image_path}/GIF.gif", 
                        save_all = True, append_images = gif[1:], 
                        optimize = False, duration = 10) 


if __name__ == '__main__':
    # Compute similarities

    prompt = sys.argv[1]
    imageDir = sys.argv[1]

    mutationStrategy = MoveBlobsStrategy(128, 200)
    scoreSystem = EmbeddingsScoring("clip-ViT-B-32", prompt)
    
    localSearch = SimulatedAnnealing(mutationStrategy, scoreSystem)

    comparisons = sys.argv[1:]

    # Path
    path = os.path.join("images", imageDir)
    
    # Create the directory
    os.makedirs(path, exist_ok=True)

    MIN_BLOBS = 1
    MAX_BLOBS = 5

    for i in range(MIN_BLOBS):
        mutationStrategy.add_blob()

    for i in range(MAX_BLOBS - MIN_BLOBS):
        mutationStrategy.add_blob()
        #mutationStrategy.anneal(f"images/{imageDir}/{i}")        
        localSearch.search(f"images/{imageDir}/{i}")

    localSearch.history[0].save(f"images/{imageDir}/GIF.gif", 
            save_all = True, append_images = localSearch.history[1:], 
            optimize = False, duration = 10) 
    
    localSearch.best_history[0].save(f"images/{imageDir}/bestGIF.gif", 
            save_all = True, append_images = localSearch.best_history[1:], 
            optimize = False, duration = 10) 