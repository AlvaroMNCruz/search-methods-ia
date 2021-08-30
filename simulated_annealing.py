# simulated annealing aplicado a um problema de MINIMIZAÇÃO!
from numpy import asarray
from numpy import exp
from numpy.random import randn
import random 
import math
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
 
# Função objetivo - aqui podemos colocar qualquer função que queremos resolver no espaço unidimensional e com apenas uma variável.
def objetivo(x):
    y = 2 ** (-2 * (x - 0.1 / 0.9) ** 2) * (math.sin(5 * math.pi * x))** 6
    return y
 
# algoritmo simulated annealing 
def simulated_annealing(objetivo, dominio, n_iter, passo, temp, alpha):
    # Gera ponto inicial para o algoritmo
    solucao = round(random.uniform(dominio[0], dominio[1]),2)
    # avalia ponto inicial
    solucao_eval = objetivo(solucao)
    print('Solução inicial: f({}) = {} '.format(solucao,solucao_eval))
    # solucao atual
    solucao_atual, solucao_atual_eval = solucao, solucao_eval
    pontos = list()
    aceitos = list()
    t=temp
    # rodando algoritmo
    while(t>0.000001):
        for i in range(n_iter):
            # definindo tamanho da perturbação
            pertubacao = random.randrange(-2,2)
            candidato = solucao_atual + (pertubacao * passo)
            # avaliando candidato (vizinho)
            candidato_eval = objetivo(candidato)
            # verificando se solução encontrada é a melhor até o momento
            if (candidato_eval < solucao_eval):
                # armazena o melhor ponto encontrado e sua avaliação
                solucao, solucao_eval = candidato, candidato_eval
                # armazena o ponto para desenhar o gráfico de melhorias
                pontos.append(solucao_eval)
                # imprime a evolução do algoritmo
                print('>Iteração(%d) f(%s) = %.5f' % (i, solucao, solucao_eval))
            # cálculo da variação entre a solução atual e a do candidato
            delta = candidato_eval - solucao_atual_eval
            # calcula a temperatura para iteração atual. Poderíamos utilizar outros métodos, como o de fator (alpha) ou fator de resfriamento.
            t = t * alpha
            # cálculo de metropolis, i.e, aceitação de mudança
            metropolis = exp(-delta / t)
            # verifica se aceitaremos uma resposta, mesmo que pior
            if (delta < 0 or rand() < metropolis):
                # store the new current point
                solucao_atual, solucao_atual_eval = candidato, candidato_eval
                print('>Iteração({}) Aceitando f({}) = {}'.format(i,round(solucao_atual,4), round(solucao_atual_eval,4)))
                aceitos.append(candidato_eval)
    return [solucao, solucao_eval, pontos, aceitos]
 
# seed the pseudorandom number generator
seed(1)
# define domínio da função
dominio = [-5, 5]
# define o número máximo de iterações
n_iter = 1000
# define o tamanho do passo
passo = 0.1
# temperatura inicial
temp = 10
#fator de resfriamento
alpha = 0.9
# executa o simulated annealing
solucao, solucao_eval, pontos, aceitos = simulated_annealing(objetivo, dominio, n_iter, passo, temp, alpha)
print('Pronto!')
print('f(%s) = %f' % (solucao, solucao_eval))

# pĺota o gráfico de melhorias encontradas
pyplot.subplot(1, 2, 1) # row 1, col 2 index 1
pyplot.plot(pontos, '.-')
pyplot.title('Melhorias')
pyplot.xlabel('Número da melhoria')
pyplot.ylabel('Avaliação de f(x)')

pyplot.subplot(1, 2, 2) # index 2
pyplot.plot(aceitos, '.-')
pyplot.title('Aceitações')
pyplot.xlabel('Número da aceitação')
pyplot.ylabel('Avaliação de f(x)')

pyplot.show()