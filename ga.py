import numpy as np
from numpy.random.mtrand import rand, randint, random, seed

def inicia_populacao(tamanho_populacao, tamanho_individuo):
    populacao = []
    populacao = np.random.randint(2,size=(tamanho_populacao,tamanho_individuo))
    return populacao

def aptidao(populacao,padrao):
    lista_aptidao = []
    for i in populacao:
        aptidao_i = np.count_nonzero(i != padrao)
        aptidao_i = len(padrao) - aptidao_i
        lista_aptidao.append(aptidao_i)
    return lista_aptidao

def roleta(populacao, aptidao):
    selecionados = []
    aptidao_relativa = []
    aptidao_total = sum(aptidao)
    ## Define proporções da roleta proporcional as aptidões de cada indivíduo    
    for index,i in enumerate(aptidao):
        if(index>0):
            porcao = (360*i/aptidao_total) + aptidao_relativa[index-1]
        else:
            porcao = (360*i/aptidao_total)
        aptidao_relativa.append(porcao)

    for i in range(len(populacao)):
        numero_sorteado = randint(0,360)
        for index,porcao_roleta in enumerate(aptidao_relativa):
            if(numero_sorteado<porcao_roleta):
                selecionados.append(populacao[index])
                break

    return np.array(selecionados)

def reproducao(populacao, taxa_crossover, taxa_mutacao):
    pares = []
    nova_populacao = []
    
    ##formando pares em uma lista de pares
    for i in range(0,len(populacao),2):
        pares.append([populacao[i], populacao[i+1]])

    probabilidade_crossover = np.random.rand(1,len(pares))

    for index,i in np.ndenumerate(probabilidade_crossover):
        if(i >= taxa_crossover):
            ##definindo ponto de crossover aleatoriamente
            cp = randint(0,12)
            filho_1 = np.concatenate((pares[index[1]][0][:cp],pares[index[1]][1][cp:]))
            filho_2 = np.concatenate((pares[index[1]][1][:cp],pares[index[1]][0][cp:]))
            nova_populacao.append(filho_1)
            nova_populacao.append(filho_2)
        else:
            nova_populacao.append(pares[index[1]][0])
            nova_populacao.append(pares[index[1]][1])

    ##Aplicando mutação caso exista

    for index,i in enumerate(populacao):
        for index_gene,gene in np.ndenumerate(i):
            if(random(1) <= taxa_mutacao):
                if(nova_populacao[index][index_gene] ==  0): nova_populacao[index][index_gene] = 1
                else: nova_populacao[index][index_gene] = 0
  
    return np.array(nova_populacao)

##Definindo parâmetros do algoritmo genético
tamanho_populacao = 8
tamanho_individuo = 12
taxa_crossover = 0.6
taxa_mutacao = 0.02
max_geracao = 50000

padrao = np.array([0,1,0,0,1,0,0,1,0,0,1,0])

print("Padrão a ser encontrado:\n", padrao)

print("Gerando população inicial...")
populacao = inicia_populacao(tamanho_populacao, tamanho_individuo)
print(populacao)

for geracao in range(max_geracao):
    print("Calculando aptidão da população (geração {})".format(geracao+1))
    lista_aptidao = aptidao(populacao, padrao)
    print(lista_aptidao)

    ## Verificando se algum indivíduo atende o critério de parada: ao menos um indivíduo com aptidão igual a 12
    if(tamanho_individuo in lista_aptidao):
        print("Critério de parada atendido!");
        print("Geração número {}".format(geracao+1));
        print("População final:\n", populacao)
        break

    ## Seleção de indivíduos por roleta
    print("Selecionando indivíduos da população (geração {})".format(geracao+1))
    selecionados = roleta(populacao,lista_aptidao)
    print("População selecionada:\n", selecionados)

    populacao = reproducao(selecionados, taxa_crossover, taxa_mutacao)
    print("Nova população:\n", populacao)
