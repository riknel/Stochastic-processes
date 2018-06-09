import numpy as np

def create_page_rank_markov_chain(links, damping_factor=0.15):
    ''' По веб-графу со списком ребер links строит матрицу 
    переходных вероятностей соответствующей марковской цепи.
    
        links --- список (list) пар вершин (tuple), 
                может быть передан в виде numpy.array, shape=(|E|, 2);
        damping_factor --- вероятность перехода не по ссылке (float);
        
        Возвращает prob_matrix --- numpy.matrix, shape=(|V|, |V|).
    '''

    links = np.array(links)
    N = links.max() + 1  # Число веб-страниц
    
    
    #Строим матрицу смежности

    #матрица смежности
    Matrix_links = np.zeros((N, N))
    #вспомогательный массив, в котором храним первый столбец links
    pages_with_link = links[:, 0]
    
    for i in range(N):
        #найдем массив страниц, на которые ссылается i-ая страница
        links_i = links[:,1][pages_with_link==i]
        
        Matrix_links[i, links_i] = 1
    
    
    #Построим массив N_i
    #Заметим, что N_i для определенного i это сумма чисел в i-ой строке матрице смежности
    N_i = np.sum(Matrix_links, axis=1)
    
    #массив вероятнотей перехода по ссылке для каждой страницы
    p_links = (1 - damping_factor) / N_i
    p_links[N_i == 0] = 0  #уберем бесконечности, которые получились когда N_i = 0
    
    
    #сначала добавим вероятности перехода по ссылке
    #для этого, каждую i-ую строку матрицы смежности нужно умножить на число p_links[i]
    #а для этого воспользуемся тем, что * - поэлементное умножение матриц и функцией np.tile
    prob_matrix = Matrix_links * (np.tile(p_links, (N, 1))).T
    
    #массив вероятнотей перехода не по ссылке для каждой страницы
    p_damping = np.ones((N,)) * (damping_factor / N)
    p_damping[N_i == 0] = 1.0 / N
    
    #добавим для каждой страницы свои вероятности перехода по телепортации
    prob_matrix = prob_matrix + (np.tile(p_damping, (N, 1))).T
    
    return prob_matrix
            
    
def page_rank(links, start_distribution, damping_factor=0.15, 
              tolerance=10 ** (-7), return_trace=False):
    ''' Вычисляет веса PageRank для веб-графа со списком ребер links 
    степенным методом, начиная с начального распределения start_distribution, 
    доводя до сходимости с точностью tolerance.
    
        links --- список (list) пар вершин (tuple), 
                может быть передан в виде numpy.array, shape=(|E|, 2);
        start_distribution --- вектор размерности |V| в формате numpy.array;
        damping_factor --- вероятность перехода не по ссылке (float);
        tolerance --- точность вычисления предельного распределения;
        return_trace --- если указана, то возвращает список распределений во 
                            все моменты времени до сходимости
    
        Возвращает:
        1). если return_trace == False, то возвращает distribution --- 
        приближение предельного распределения цепи,
        которое соответствует весам PageRank.
        Имеет тип numpy.array размерности |V|.
        2). если return_trace == True, то возвращает также trace ---
        список распределений во все моменты времени до сходимости. 
        Имеет тип numpy.array размерности 
        (количество итераций) на |V|.
    '''
    
    prob_matrix = create_page_rank_markov_chain(links, 
                                                damping_factor=damping_factor)
    distribution_old = np.matrix(start_distribution)
    
    distribution = np.dot(distribution_old, prob_matrix)
    trace = [distribution_old, distribution]
    while np.linalg.norm(distribution - distribution_old) > tolerance:
        distribution_old = distribution
        distribution = np.dot(distribution, prob_matrix)
        trace.append(distribution)
    
    if return_trace:
        return np.array(distribution).ravel(), np.array(trace)
    else:
        return np.array(distribution).ravel()
