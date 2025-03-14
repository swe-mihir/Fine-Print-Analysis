import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


words = ['Negative1', 'Negative2', 'Negative3', 'Negative4', 'Negative5', 'Positive6', 'Positive7', 'Positive8', 'Positive9', 'Positive10']
vectors = np.array([
    [0.1155, -0.1939, 0.0231, -0.0199, -0.2317],
    [-0.1141, -0.1502, -0.0441, 0.1240, 0.0716],
    [0.0159, -0.3768, -0.0215, -0.0711, 0.1071],
    [0.1049, -0.4109, -0.0008, 0.0873, 0.4608],
    [0.1731, -0.3499, -0.2199, 0.0166, 0.2790],
    [0.1180, -0.5523, -0.3900, -0.0831,  0.2512],
    [0.3821, -0.6564, -0.4231,  0.0266,  0.3836],
    [-0.0200, -0.1477, -0.1843,  0.0234, -0.2232],
    [0.3191, -0.2046, -0.2825, -0.1490,  0.0968],
    [0.0974, -0.4132, -0.1385,  0.0281, -0.0122]
])


pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)


plt.figure(figsize=(8, 6))
for i, words in enumerate(words):
    x, y = reduced_vectors[i]
    plt.scatter(x, y, marker='o', color='blue')
    plt.text(x + 0.01, y + 0.01, words, fontsize=12)
plt.title("2D Visualization of 5D Word Embeddings (PCA)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True)
plt.show()
