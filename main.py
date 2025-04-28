# https://www.sbert.net/examples/sentence_transformer/applications/image-search/README.html
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageDraw
import sys
import random
import os
import math

# Load CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# Colours
white = (255, 255, 255)
black = (0, 0, 0)

class MutationStrategy():
    def __init__(self, size, pallet):
        pil_image = Image.new(mode="RGB", size=(size, size))
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle([(0, 0), (pil_image.width - 1, pil_image.height - 1)], white)

        self.baseImage = pil_image
        self.pallet = pallet

    def mutate_image(self, image):
        return image # Placeholder
    
    def convert_pil_image(self, image):
        return self.baseImage

class RandomPixelFlipStrategy(MutationStrategy):
    def mutate_image(self, image):
        px = image.load()
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
    def __init__(self, x, y, radius, colour):
        self.x = x
        self.y = y
        self.radius = radius
        self.colour = colour

    def __str__(self):
        return f"[{self.x}, {self.y}], radius: {self.radius}"

    def addBlobToImage(self, image, colour):
        px = image.load()
        for px_x in range(image.width):
            for px_y in range(image.height):
                dist = math.pow(px_x - self.x, 2) + math.pow(px_y - self.y, 2)
                if dist < math.pow(self.radius, 2):
                    px[px_x, px_y] = colour.pallet[self.colour]

class MoveBlobsStrategy(MutationStrategy):
    
    def mutate_image(self, image):
        mutateProb = 1
        if random.random() < mutateProb:
            rand = random.random()
            if rand < 0.2 or len(image) <= 0:
                # Add blob
                x = random.randint(0, self.baseImage.width - 1)
                y = random.randint(0, self.baseImage.height - 1)
                radius = random.randint(0, self.baseImage.width/4)
                
                image.append(Blob(x, y, radius, self.pallet.randomFromPallet()))
            elif rand < 0.4:
                # Remove blob, weighted by radius size
                radiuses = sum(list(map(lambda blob: blob.radius, image)))
                rand = random.randint(0, radiuses)
                
                radi = 0
                i = 0
                for blob in image:
                    radi = radi + blob.radius
                    if rand < radi:
                        image.pop(i)
                        return image
                    i = i + 1
            elif rand < 0.6:
                # Alter pallet
                self.pallet.mutatePallet(image)
            else:
                rand = random.randint(0, len(image) - 1)
                blob = image[rand]
                if rand < 0.8:
                    # Move blob
                    blob.x = blob.x + random.randint(-1, 1)
                    blob.y = blob.y + random.randint(-1, 1)
                else:
                    # Change blob radius
                    blob.radius = max(1, int(blob.radius * (random.random() + 0.5)))

            mutateProb = mutateProb / 2
        return image

    def convert_pil_image(self, image):
        pil_image = self.baseImage.copy()       
        for blob in image:
            blob.addBlobToImage(pil_image, self.pallet)
        
        return pil_image

def anneal(comparisons, imageDir, mutationStrategy):
    # Encode text descriptions
    text_emb = model.encode(
        comparisons
    )

    # Path
    path = os.path.join("images", imageDir)
    
    # Create the directory 'ihritik'
    os.makedirs(path, exist_ok=True)

    iterations = 0
    
    blobs = mutationStrategy.mutate_image([])
    image = mutationStrategy.convert_pil_image(blobs)

    gif = []    
    image.save(f"{path}/{str(iterations)}.png","PNG")
    gif.append(image)

    prev_similarity_score = None
    temp = 1000
    best_score = None
    bests = []
    while temp > 0:
        # Encode an image:
        mutatedBlobs = mutationStrategy.mutate_image(blobs)
        image = mutationStrategy.convert_pil_image(blobs)
        img_emb = model.encode(image)
        similarity_score = compareScore(0, text_emb, img_emb)

        if prev_similarity_score == None:
            prev_similarity_score = similarity_score

        score_dif = 500000 * (similarity_score - prev_similarity_score)
        if score_dif < 0:
            random_prob = math.exp(score_dif/temp)
        else:
            random_prob = 0
        print(random_prob)
        random_step = random_prob > random.random()

        if score_dif > 0 or random_step:
            if random_step:
                print(f" - random step {iterations}, {similarity_score}, {random_prob}")
            else:
                print(f" - new best {iterations}, {similarity_score}")    
            prev_similarity_score = similarity_score

            if best_score == None:
                best_score = similarity_score
            if similarity_score >= best_score:
                best_score = similarity_score
                bests.append(image)

            image.save(f"{path}/{str(iterations)}.png","PNG")
            gif.append(image)
            blobs = mutatedBlobs
            for blob in blobs:
                print(f" - {str(blob)}")

        iterations = iterations + 1
        temp = temp - 1

    
    gif[0].save(f"{path}/GIF.gif", 
                save_all = True, append_images = gif[1:], 
                optimize = False, duration = 10) 
    
    bests[0].save(f"{path}/BESTS.gif", 
            save_all = True, append_images = bests[1:], 
            optimize = False, duration = 10) 
    
    compare(comparisons, text_emb, model.encode(gif[-1]))
    compare(comparisons, text_emb, model.encode(bests[-1]))


def compare(texts, text_emb, img_emb):
    similarity_score = model.similarity(img_emb, text_emb).tolist()[0]
    bestIndex = max(enumerate(similarity_score),key=lambda x: x[1])[0]

    print(f"{texts[bestIndex]} {similarity_score[bestIndex]}")
    return texts[bestIndex]

def compareScore(expectedIndex, text_emb, img_emb):
    similarity_score = model.similarity(img_emb, text_emb).tolist()[0]
    bestIndexOrder = list(map(lambda e: e[0], sorted(enumerate(similarity_score),key=lambda x: x[1])))
    if bestIndexOrder[0] == expectedIndex:
        return similarity_score[expectedIndex] - similarity_score[bestIndexOrder[1]]
    else:
        return similarity_score[expectedIndex] - similarity_score[bestIndexOrder[0]]

if __name__ == '__main__':
    # Compute similarities

    prompt = sys.argv[1]
    imageDir = sys.argv[1]
    mutationStrategy = MoveBlobsStrategy(32, Colour())
    
    comparisons = sys.argv[1:]
    anneal(comparisons, imageDir, mutationStrategy)
