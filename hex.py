import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Carregamento da base de dados
dataset = pd.read_csv('personagens.csv')

# Exibição de informações iniciais (opcional)
print("Shape do dataset:", dataset.shape)
print("Primeiras linhas:")
print(dataset.head())
print("Últimas linhas:")
print(dataset.tail())

# Visualização da distribuição das classes
sns.countplot(x='classe', data=dataset)
plt.title("Distribuição das Classes")
plt.xlabel("Classe")
plt.ylabel("Contagem")
plt.show()

# Separação entre atributos (X) e rótulos (y)
X = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 6].values

# Transformando a classificação para binário (apenas 'Bart' será True)
y = (y == 'Bart')

# Divisão da base em treino e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.2)

# Verificando os shapes
print("Shape do X_treinamento:", X_treinamento.shape)
print("Shape do y_treinamento:", y_treinamento.shape)
print("Shape do X_teste:", X_teste.shape)
print("Shape do y_teste:", y_teste.shape)

# Construção e treinamento da rede neural
rede_neural = tf.keras.models.Sequential()
rede_neural.add(tf.keras.layers.Dense(units=4, activation='relu', input_shape=(6,)))
rede_neural.add(tf.keras.layers.Dense(units=4, activation='relu'))
rede_neural.add(tf.keras.layers.Dense(units=4, activation='relu'))
rede_neural.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Exibe o resumo da arquitetura
rede_neural.summary()

# Compilação do modelo
rede_neural.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
historico = rede_neural.fit(X_treinamento, y_treinamento, epochs=1000, validation_split=0.1)

# Visualização da evolução da perda e acurácia
plt.plot(historico.history['val_loss'], label='Val Loss')
plt.plot(historico.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Histórico de Treinamento")
plt.xlabel("Época")
plt.ylabel("Valor")
plt.show()

# Previsões
previsoes = rede_neural.predict(X_teste)
previsoes = (previsoes > 0.5)

# Avaliação
acc = accuracy_score(previsoes, y_teste)
print("Acurácia no teste:", acc)

# Matriz de confusão
cm = confusion_matrix(y_teste, previsoes)
print("Matriz de confusão:")
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()
