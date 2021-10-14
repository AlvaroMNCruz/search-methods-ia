#Rede com 3 perceptrons (Exemplo IRIS) 
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sys

def sum(amostra, weights, bias):
    #multiplica a entrada pelos pesos e soma o bias
    sum_perceptrons = np.zeros(weights.shape[0])
    for i in range(weights.shape[0]):
        sum_perceptron = np.dot(amostra, weights[i]) + bias[i]
        sum_perceptrons[i] = sum_perceptron

    # print('Sum perceptrons: ', sum_perceptrons)
    return sum_perceptrons

# A função de ativação utilizada foi a softmax
def softmax(valor):
    probs = np.exp(valor) / np.sum(np.exp(valor))
    # print('Softmax: ', probs)
    return probs

# Função de previsão
def previsao(amostra, weights, bias):
    juncao = sum(amostra, weights, bias)
    result = softmax(juncao)
    return result

# Vamos começar com os pesos como um array numpy aleatório, já que não temos pistas sobre quais são os pesos corretos.
def treinamento(amostras, classes, taxa_aprendizado, max_iteracoes):
    numero_perceptrons = len(np.unique(classes))
    bias = np.random.uniform(-1, 1, numero_perceptrons)
    weights = np.random.uniform(-1, 1, (numero_perceptrons,amostras.shape[1]))
    print(weights)

    quad_errors = []
    
    # Iterando até o número de max_iteracoes...
    for iteracao in range(max_iteracoes):
        quad_error = 0
        
        print("Iteração {}\n- weights: {}\n- errors: {}".format(iteracao, weights, quad_errors))
        
        for amostra, classe in zip(amostras, classes):
            print("Aprendendo amostra = {} (classe {})".format(amostra, classe))              
            
            saida = previsao(amostra, weights, bias)
            print("Previsão ({}) => Classe ({})".format(saida.argmax(axis=0),classe))
            
            error = np.zeros(len(saida))

            for i in range(len(saida)):
                if(i == classe):
                    error[i] =  1.0 - saida[i]
                    if math.isnan(error[i]): error[i] = 0

                else:
                    error[i] = 0.0 - saida[i]
                    if math.isnan(error[i]): error[i] = 0
                    
                bias[i] += taxa_aprendizado * error[i]

                for j in range(weights.shape[1]):
                    weights[i][j] += taxa_aprendizado * error[i] * amostra[j]

            quad_error += np.sum(error**2)
            print('quad_error', quad_error)
        
        quad_errors.append(quad_error)
        
        if(quad_error == 0):
            break    
        

    return bias, weights, quad_errors
    
# Validação do método (Verificando acurácia)

def validacao(amostras, classes, bias, weights):
    errors = 0
    for amostra, classe in zip(amostras, classes):
            # print("Classificando amostra = {} (classe {})".format(amostra, classe))
            saida = previsao(amostra, weights, bias)
            # print("Previsão ({}) => Classe ({})".format(saida.argmax(axis=0),classe))
            
            if(saida.argmax(axis=0) != classe): errors += 1
    print('{} erros na classificação!'.format(errors))
    acuracia = (amostras.shape[0] - errors) / amostras.shape[0]

    return acuracia

# Importando base de dados da Iris
iris = datasets.load_iris()
amostras = iris.data[:,:]
classes = iris.target

print(classes)

print('Número de amostras: ', len(amostras))
print('Número de classes: ', len(classes))

amostras_treinamento = amostras[:int(70/100*amostras.shape[0])]
amostras_validacao = amostras[int(70/100*amostras.shape[0]):]

classes_treinamento = classes[:int(70/100*classes.shape[0])]
classes_validacao = classes[int(70/100*classes.shape[0]):]

print('Número de amostras treinamento: ', len(amostras_treinamento))
print('Número de classes treinamento: ', len(classes_treinamento))
print('Número de amostras validação: ', len(amostras_validacao))
print('Número de classes validação: ', len(classes_validacao))

bias, weights, quad_errors = treinamento(amostras_treinamento, classes_treinamento, taxa_aprendizado=0.01, max_iteracoes=50)

print('Matriz de pesos final: ', weights)
print('Bias final: ', bias)
plt.plot(quad_errors)
plt.show()

acuracia = validacao(amostras_validacao, classes_validacao, bias, weights)

print("Acurácia de {}%".format(acuracia*100))