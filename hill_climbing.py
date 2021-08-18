# hill climbing search of a one-dimensional objective function
from numpy.random import randn
import random
from numpy.random import seed
import math
from matplotlib import pyplot

# Função objetivo - aqui podemos colocar qualquer função que queremos resolver no espaço unidimensional e com apenas uma variável.
def objetivo(x):
    y = 2 ** (-2 * (x - 0.1 / 0.9) ** 2) * (math.sin(5 * math.pi * x))** 6
    return y

def hill_climbing(objetivo, dominio, n_iter, passo):
    # Gera ponto inicial para o algoritmo
    solucao = round(random.uniform(dominio[0], dominio[1]),2)
    print("Solução inicial: ", solucao)
    # avaliando solução inicial
    solucao_eval = objetivo(solucao)
    #lista para armazenar todos as melhorias
    pontos = list()
    # iniciando hill climbing
    for i in range(n_iter):
        # dando um passo
        pertubacao = randn()
        candidato_1 = solucao + (pertubacao * passo)
        candidato_2 = solucao - (pertubacao * passo)
        # avaliando ponto candidato_1 e 2
        candidato_eval_1 = objetivo(candidato_1)
        candidato_eval_2 = objetivo(candidato_2)
        melhor_candidato = 0
        melhor_candidato_eval = 0
        if(candidato_eval_1 >= candidato_eval_2): 
            melhor_candidato = candidato_1
            melhor_candidato_eval = candidato_eval_1
        else: 
            melhor_candidato = candidato_2
            melhor_candidato_eval = candidato_eval_2
        # verificando se nova solução é melhor
        if melhor_candidato_eval >= solucao_eval:
            # Armazena o melhor ponto e sua imagem
            solucao, solucao_eval = melhor_candidato, melhor_candidato_eval
            pontos.append(solucao_eval)
            # Imprimindo melhor solução até o momento
            print('>%d f(%s) = %.5f' % (i, solucao, solucao_eval))
    return [solucao, solucao_eval, pontos]

# seed the pseudorandom number generator
seed(10)
# definimos o domínio da função
dominio = [-5, 5]
# define o total de iterações
n_iter = 10000
# define o tamanho do passo a ser dado pelo algoritmo
passo = 0.1
# chama a função hill_climbing
melhor, imagem, pontos = hill_climbing(objetivo, dominio, n_iter, passo)
print('Pronto!')
print('f(%s) = %f' % (melhor, imagem))
# line plot of best scores
pyplot.plot(pontos, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()