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
black = (255, 0, 0)

class MutationStrategy():
    def __init__(self):
        pass

    def mutate_image(self, image):
        return image # Placeholder

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

def anneal(prompt, imageDir, mutationStrategy):
    # Encode text descriptions
    text_emb = model.encode(
        [prompt]
    )

    # Path
    path = os.path.join("images", imageDir)
    
    # Create the directory 'ihritik'
    os.makedirs(path, exist_ok=True)

    iterations = 0
    size = 8
    image = Image.new(mode="RGB", size=(size, size))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (image.width - 1, image.height - 1)], white)

    image.save(f"{path}/{str(iterations)}.png","PNG")
    prev_similarity_score = 0
    temp = 1000

    while temp > 0:
        # Encode an image:
        mutatedImage = mutationStrategy.mutate_image(image)
        #image.save(f"{path}/{str(iterations)}.png","PNG")
        img_emb = model.encode(mutatedImage)
        similarity_score = model.similarity(img_emb, text_emb)[0]

        score_dif = 10000 * ((similarity_score - prev_similarity_score)/prev_similarity_score)
        random_prob = math.exp(score_dif/temp)
        random_step = random_prob > random.random()
        # print(f"{iterations}, {similarity_score}, {random_prob}") 
        if score_dif > 1 or random_step:
            if score_dif > 0:
                print(f" - new best {iterations}, {similarity_score}")    
            else:
                print(f" - random step {iterations}, {similarity_score}, {random_prob}")
            prev_similarity_score = similarity_score
            mutatedImage.save(f"{path}/{str(iterations)}.png","PNG")
            
            image = mutatedImage

        iterations = iterations + 1
        temp = temp - 1

def compare(texts, text_emb, imgfile):
    img_emb = model.encode(Image.open(f"images/{imgfile}"))
    similarity_score = model.similarity(img_emb, text_emb).tolist()[0]
    bestIndex = max(enumerate(similarity_score),key=lambda x: x[1])[0]

    print(f"{texts[bestIndex]} {similarity_score[bestIndex]}")
    return texts[bestIndex]

if __name__ == '__main__':
    # Compute similarities

    prompt = sys.argv[1]
    imageDir = sys.argv[1]
    mutationStrategy = RandomPixelFlipStrategy()
    anneal(prompt, imageDir, mutationStrategy)

    #texts = ["Building", "Earth", "Hand", "Pope", "Canadian flag"]
    #text_emb = model.encode(texts)
    #compare(texts, text_emb, "building.webp")
    #compare(texts, text_emb, "canada.png")
    #compare(texts, text_emb, "pope.jpg")