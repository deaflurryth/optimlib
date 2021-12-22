import numpy as np
from numpy import array, zeros, diag, diagflat, dot
import random
import math
import time
from datetime import datetime

def CompactACO():
    import random as rd
    def ACO():
        np.seterr(divide= 'ignore', invalid= 'ignore')
        def lengthCal(antPath,distmat):         # Рассчитать расстояние
            length= []
            dis= 0
            for i in range(len(antPath)):
                for j in range(len(antPath[i])- 1):
                    dis+= distmat[antPath[i][j]][antPath[i][j+ 1]]
                dis+= distmat[antPath[i][-1]][antPath[i][0]]
                length.append(dis)
                dis= 0
            return length
        print('Алгоритм муравьиной колонии')
        print('_______________')
        sizer= int(input('Введите кол-во городов: '))
        distmat= np.array(np.random.randint(100, size= (sizer, sizer)))
        print('α= 0.6 | β= 0.65 | p= 0.3')
        antNum= sizer                   # Кол-во муравьев

        alpha= 0.6                      # Фактор важности феромона
        beta= 0.65                      # Фактор важности эвристической функции
        pheEvaRate= 0.3              # Скорость испарения феромона
        cityNum= distmat.shape[0]
        pheromone= np.ones((cityNum,cityNum))                   # Феромоновая матрица
        heuristic= 1/ (np.eye(cityNum)+ distmat)- np.eye(cityNum)       # Матрица эвристической информации, дубль 1 / дизмат
        iter, itermax= float(input('Введите феромон: ')), 100                       # Итерации

        while iter < itermax:
            antPath= np.zeros((antNum, cityNum)).astype(int) - 1   # Путь муравья
            firstCity= [i for i in range(sizer)]
            rd.shuffle(firstCity)          # Случайно назначьте начальный город для каждого муравья
            unvisted= []
            p= []
            pAccum= 0
            for i in range(len(antPath)):
                antPath[i][0]= firstCity[i]
            for i in range(len(antPath[0]) - 1):       # Постепенно обновляйте следующий город, в который собирается каждый муравей
                for j in range(len(antPath)):
                    for k in range(cityNum):
                        if k not in antPath[j]:
                            unvisted.append(k)
                    for m in unvisted:
                        pAccum+= pheromone[antPath[j][i]][m]** alpha* heuristic[antPath[j][i]][m] ** beta
                    for n in unvisted:
                        p.append(pheromone[antPath[j][i]][n]** alpha* heuristic[antPath[j][i]][n]** beta/ pAccum)
                    roulette= np.array(p).cumsum()               # Создать рулетку
                    r= rd.uniform(min(roulette), max(roulette))
                    for x in range(len(roulette)):
                        if roulette[x]>= r:                      # Используйте метод рулетки, чтобы выбрать следующий город
                            antPath[j][i + 1]= unvisted[x]
                            break
                    unvisted= []
                    p = []
                    pAccum= 0
            pheromone = (1 - pheEvaRate)* pheromone            # Феромон летучий
            length= lengthCal(antPath,distmat)
            for i in range(len(antPath)):
                for j in range(len(antPath[i])- 1):
                    pheromone[antPath[i][j]][antPath[i][j+ 1]]+= 1 / length[i]     # Обновление феромона
                pheromone[antPath[i][-1]][antPath[i][0]]+= 1 / length[i]
            iter += 1
        print('-------------------------')
        print('Результат: ', antPath[length.index(min(length))])
        print('Стоимость: ', min(length))
    def ACO2():
        np.seterr(divide= 'ignore', invalid= 'ignore')
        def lengthCal(antPath,distmat):         # Рассчитать расстояние
            length= []
            dis= 0
            for i in range(len(antPath)):
                for j in range(len(antPath[i])- 1):
                    dis+= distmat[antPath[i][j]][antPath[i][j+ 1]]
                dis+= distmat[antPath[i][-1]][antPath[i][0]]
                length.append(dis)
                dis= 0
            return length
        print('Алгоритм муравьиной колонии')
        print('_______________')
        print('Кол-во городов = 5')
        print('Феромон(начальный) = 0.1')
        print('α= 0.6 | β= 0.65 | p= 0.3')
        sizer= 5
        distmat= np.array(np.random.randint(100, size= (sizer, sizer)))
        print('Кол-во муравьев = кол-ву городов')
        antNum= sizer                   # Кол-во муравьев

        alpha= 0.6                     # Фактор важности феромона
        beta= 0.65                      # Фактор важности эвристической функции
        pheEvaRate= 0.3              # Скорость испарения феромона
        cityNum= distmat.shape[0]
        pheromone= np.ones((cityNum,cityNum))                   # Феромоновая матрица
        heuristic= 1/ (np.eye(cityNum)+ distmat)- np.eye(cityNum)       # Матрица эвристической информации, дубль 1 / дизмат
        iter, itermax= 0.1, 100                       # Итерации

        while iter < itermax:
            antPath= np.zeros((antNum, cityNum)).astype(int) - 1   # Путь муравья
            firstCity= [i for i in range(sizer)]
            rd.shuffle(firstCity)          # Случайно назначьте начальный город для каждого муравья
            unvisted= []
            p= []
            pAccum= 0
            for i in range(len(antPath)):
                antPath[i][0]= firstCity[i]
            for i in range(len(antPath[0]) - 1):       # Постепенно обновляйте следующий город, в который собирается каждый муравей
                for j in range(len(antPath)):
                    for k in range(cityNum):
                        if k not in antPath[j]:
                            unvisted.append(k)
                    for m in unvisted:
                        pAccum+= pheromone[antPath[j][i]][m]** alpha* heuristic[antPath[j][i]][m] ** beta
                    for n in unvisted:
                        p.append(pheromone[antPath[j][i]][n]** alpha* heuristic[antPath[j][i]][n]** beta/ pAccum)
                    roulette= np.array(p).cumsum()               # Создать рулетку
                    r= rd.uniform(min(roulette), max(roulette))
                    for x in range(len(roulette)):
                        if roulette[x]>= r:                      # Используйте метод рулетки, чтобы выбрать следующий город
                            antPath[j][i + 1]= unvisted[x]
                            break
                    unvisted= []
                    p = []
                    pAccum= 0
            pheromone = (1 - pheEvaRate)* pheromone            # Феромон летучий
            length= lengthCal(antPath,distmat)
            for i in range(len(antPath)):
                for j in range(len(antPath[i])- 1):
                    pheromone[antPath[i][j]][antPath[i][j+ 1]]+= 1 / length[i]     # Обновление феромона
                pheromone[antPath[i][-1]][antPath[i][0]]+= 1 / length[i]
            iter += 1
        print('-------------------------')
        print('Результат: ', antPath[length.index(min(length))])
        print('Стоимость: ', min(length))
    picker2= int(input('Будете вводить сами?: [1] - да, [2] - нет: '))
    if picker2== 1:
        start_time = datetime.now()
        ACO()
        end_time = datetime.now()
        print('Время выполнения: {}'.format(end_time - start_time))
    elif picker2== 2:
        start_time = datetime.now()
        ACO2()
        end_time = datetime.now()
        print('Время выполнения: {}'.format(end_time - start_time))
#CompactACO()