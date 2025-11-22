import numpy as np
from sentence_transformers import SentenceTransformer
import torch

"""
Author: Trevor Robbins
File Purpose: Embedding of sentences using 'SentenceTransformer" to vectorize data in backend

Goals: Understand basic linear algebraeic concepts such as dot products, euclidean normality, and cosine similarity
       of two different vectors 
       

"""

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text) -> torch.tensor:
    return model.encode(text, convert_to_tensor=True)




sentences = [
    "A neural network can learn patterns.",
    "A neural network is capable of learning patterns."
]



"""testVAL1 = np.random.rand(1024)
compare1 = np.random.rand(1024)

testVAL2 = np.random.rand(512)
compare2 = np.random.rand(512)

testVAL3 = np.random.rand(256)
compare3 = np.random.rand(256)

testVAL4 = np.random.rand(128)
compare4 = np.random.rand(128)

x = cosine_sim(testVAL1, compare1)
y = cosine_sim(testVAL2, compare2)
z= cosine_sim(testVAL3, compare3)
g = cosine_sim(testVAL4, compare4)

print(x)
print(y)
print(z)
print(g)
"""




#def vectorCompare(x, y):
    
""""
def dot(x, y):
    return np.dot(x, y)
"""

def eucilidean_norm(x):
    """
    Variables: x[vector of n-elements]

    Mathematical formula in pseudocode:
    
    sumSquare = 0 
    for i = 0 to n:
        sumSquare = sumSquare + (x[i]*x[i]) 
    
    return np.sqrt(sumSquare)
        
        
    """
    return np.linalg.norm(x)


def cosine_sim(x, y):
    """
    z = (a * b)/
        |a|*|b|

    """
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))



#print("Shape:", embs.shape)
#print("First vector (first 5 dims):", embs[0][:])
print(torch.cosine_similarity(embed(sentences[0]), embed(sentences[1]), dim = 0))

#print(torch.cuda.is_available())  # True = using GPU