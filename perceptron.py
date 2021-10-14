#Rede perceptron simples

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def sum(amostra, weights, bias):
    #multiplica a entrada pelos pesos e soma o bias
    dp = np.dot(amostra, weights) + bias
    
    return dp

# A função de ativação utilizada foi a degrau ou limiar
def ativacao(valor):
    return np.where(valor >= 0.0, 1, -1)

# Função de previsão
def previsao(amostra, weights, bias):
    juncao = sum(amostra, weights, bias)
    result = ativacao(juncao)
    
    return result

# Vamos começar com os pesos como um array numpy aleatório, já que não temos pistas sobre quais são os pesos corretos.
def treinamento(amostras, classes, taxa_aprendizado=0.01, max_iteracoes=50):
    bias = np.random.uniform(-1, 1)
    weights = np.random.uniform(-1, 1, (amostras.shape[1]))
    errors = []
    
    # Iterando até o número de max_iteracoes...
    for iteracao in range(max_iteracoes):
        error = 0
        
        print("Iteração {}\n- weights: {}\n- errors: {}".format(iteracao, weights, errors))
        
        for amostra, classe in zip(amostras, classes):
            print("Aprendendo amostra = {} (classe {})".format(amostra, classe))
            
            saida = previsao(amostra, weights, bias)

            print("Previsão ({}) => Classe ({})".format(saida,classe))
            atualizacao = taxa_aprendizado * (classe - saida)
            
            bias += atualizacao
            
            weights += atualizacao * amostra
            
            error += int(atualizacao != 0.0)
            
        errors.append(error)

    return bias, weights, errors
    
# Validação do método (Verificando acurácia)

def validacao(amostras, classes, bias, weights):
    errors = 0
    for amostra, classe in zip(amostras, classes):
            print("Classificando amostra = {} (classe {})".format(amostra, classe))
            saida = previsao(amostra, weights, bias)
            print("Previsão ({}) => Classe ({})".format(saida,classe))
            
            if(saida != classe): errors += 1
    print('{} erros na classificação!'.format(errors))
    acuracia = (amostras.shape[0] - errors) / amostras.shape[0]

    return acuracia

# Vamos criar 2 blobs com 200 samples e 4 features (2 dimensões) e 2 centros.
blobs = make_blobs(n_samples=200, n_features=4, centers=2)
print(blobs)
# Em seguida, vamos plotar os dados gerados para visualização.
plt.scatter(blobs[0][:,0], blobs[0][:,1], c=blobs[1])
plt.show()

amostras_treinamento = blobs[0][:int(70/100*blobs[0].shape[0])]
amostras_validacao = blobs[0][int(70/100*blobs[0].shape[0]):]

classes_treinamento = blobs[1][:int(70/100*blobs[1].shape[0])]
classes_validacao = blobs[1][int(70/100*blobs[1].shape[0]):]

classes_treinamento[classes_treinamento == 0] = -1
classes_validacao[classes_validacao == 0] = -1

bias, weights, errors = treinamento(amostras_treinamento, classes_treinamento, taxa_aprendizado=0.01, max_iteracoes=50)

plt.plot(errors)
plt.show()

acuracia = validacao(amostras_validacao, classes_validacao, bias, weights)
print("Acurácia de {}%".format(acuracia*100))