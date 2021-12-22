import numpy as np
from numpy import array, zeros, diag, diagflat, dot
import random
import math
import time
from datetime import datetime

def CompactAnn():
    def Simulated():
        print('ИМИТАЦИЯ ОТЖИГА')
        print('_______________')
        v= int(input('Введите кол-во вершин: '))
        iteraaa= int(input('Введите кол-во итераций: '))
        t= float(input('Введите температуру: '))
        a= float(input('Введите альфу: '))
        number= v
        asker= int(input('Показывать шаги ? 1 - да, 2 - нет: '))
        
        def generateRandomState(n):
            return list(range(n)) 
        def costFunction(solution):
            cost= 0
            for position in range(0, len(solution)):
                for next_position in range(position+ 1, len(solution)):
                    if (solution[position]== solution[next_position]) or abs(position- next_position)== abs(solution[position]- solution[next_position]):
                        cost= cost+ 1
            return cost
        def generateNextState(state):
            state= state[:]
            i, j= random.sample(range(len(state)), 2)
            state[i], state[j]= state[j], state[i]
            return state
        
        def simulatedAnnealing(size, temperature, alpha, number, itera):
            max_iter= itera
            current_state= generateRandomState(size)
            current_cost= costFunction(current_state)
            counterhero= 0
            for iteration in range(max_iter):
                if asker== 1:
                    print('шаг: ',counterhero, '|', current_state, current_cost) 
                temperature= temperature* alpha
                counterhero+= 1
                next_state= generateNextState(current_state)
                next_cost= costFunction(next_state)
                delta_E= next_cost - current_cost
                if delta_E<0 or math.exp(-delta_E/ temperature)> random.uniform(0,1):
                   current_state= next_state
                   current_cost= next_cost
                   if current_cost== 0:
                      return current_state
            print('------------------------')
            print('Результат: ', current_state)
            print('Стоимость: ', current_cost)
            return None
        simulatedAnnealing(v, t, a, number, iteraaa)
        print('_______________')
        return
    
    
    def Simulated2():
        print('ИМИТАЦИЯ ОТЖИГА')
        print('_______________')
        print('Количество вершин = 5')
        print('Колчество итераций = 500')
        print('Стартовая температура = 0.1')
        print('Альфа = 5')
        asker= int(input('Показывать шаги ? 1 - да, 2 - нет: '))
        v= 5
        iteraaa= 100
        t= 0.1
        a= 5
        number= v
        def generateRandomState2(n):
            return list(range(n)) 
        def costFunction2(solution):
            cost= 0
            for position in range(0, len(solution)):
                for next_position in range(position+ 1, len(solution)):
                    if (solution[position]== solution[next_position]) or abs(position- next_position)== abs(solution[position]- solution[next_position]):
                        cost= cost+ 1
            return cost

        def generateNextState2(state):
            state = state[:] 
            state[random.randint(0, len(state)-1)] = random.randint(0,len(state)-1)
            return state

        def simulatedAnnealing2(size, temperature, alpha, number, itera):
            max_iter= itera
            current_state= generateRandomState2(size)
            current_cost= costFunction2(current_state)
            counterhero2= 0
            for iteration in range(max_iter):
                if asker== 1:
                    print('шаг: ', counterhero2, '|', current_state, current_cost)
                temperature= temperature* alpha
                counterhero2+= 1
                next_state= generateNextState2(current_state)
                next_cost= costFunction2(next_state)
                delta_E= next_cost- current_cost
                if delta_E<0 or math.exp(-delta_E/ temperature)> random.uniform(0,1):
                   counterhero=+ 1
                   current_state= next_state
                   current_cost= next_cost
                   if current_cost== 0:
                      counterhero=+ 1
                      return current_state
            print('------------------------')
            print('Результат: ', current_state)
            print('Стоимость: ', current_cost)
            return None
        simulatedAnnealing2(v, t, a, number, iteraaa)
        return
    picker1= int(input('Будете вводить сами?: [1] - да, [2] - нет: '))
    if picker1== 1:
        start_time= datetime.now()
        Simulated()
        end_time= datetime.now()
        print('Время выполнения: {}'.format(end_time- start_time))
    elif picker1== 2:
        start_time= datetime.now()
        Simulated2()
        end_time= datetime.now()
        print('Время выполнения: {}'.format(end_time- start_time))
#CompactAnn()
