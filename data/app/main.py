import numpy as np
from sklearn.linear_model import SGDClassifier

# Exemplo de modelo incremental (online learning)
model = SGDClassifier()

# Dados fictícios
X = np.random.rand(10, 4)
y = np.random.randint(0, 2, size=10)

# Treina incrementalmente
model.partial_fit(X, y, classes=[0, 1])
print("Modelo treinado com batch inicial!")

# Simula nova rodada
X_new = np.random.rand(2, 4)
y_new = np.random.randint(0, 2, size=2)
model.partial_fit(X_new, y_new)

print("Nova rodada treinada. Predições:", model.predict(X_new))
