import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


'Utility Functions'

def softmax(logits):
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def temperature_scale(logits, T):
    if T == 0.0:
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs
    return softmax(logits / T)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


'Similarity Visualization'

def similarity_visualization():
    print("Generating Similarity Matrix...")
    logits = np.random.randn(10)
    temps = [0.01, 0.3, 0.7, 1.0]  
    distributions = [temperature_scale(logits, t) for t in temps]

    similarity_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            similarity_matrix[i, j] = cosine_similarity(distributions[i], distributions[j])

    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label="Similarity Score")
    plt.title("Cosine Similarity Across Temperatures")
    plt.xticks(range(4), temps)
    plt.yticks(range(4), temps)
    plt.xlabel("Temperature")
    plt.ylabel("Temperature")

    
    plt.savefig("similarity_matrix.png")
    plt.show() 

'Length Distribution Analysis'

def simulate_length(T, base_len=50):
   
    return int(base_len + T * np.random.randint(5, 50))


def length_distribution_analysis():
    print("Generating Length Distribution...")
    temps = [0.0, 0.3, 0.7, 1.0]

    plt.figure(figsize=(10, 6))
    for t in temps:
        lengths = [simulate_length(t) for _ in range(30)]
        plt.hist(lengths, bins=15, alpha=0.5, label=f"T={t}")

    plt.legend()
    plt.title("Response Length Distribution by Temperature")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")

   
    plt.savefig("length_distribution.png")
    plt.show() 


'Execution'

if __name__ == "__main__":
    
    similarity_visualization()
    length_distribution_analysis()
    print("Done! Check your project folder for the .png files.")
