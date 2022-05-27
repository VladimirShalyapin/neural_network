# python notebook for Make Your Own Neural Network
# (c) Tariq Rashid, 2016
# license is GPLv2
# https://github.com/makeyourownneuralnetwork/

################################################################
########### license is GPLv2 2022 Vladimir Shalyapin ###########
###########               Python   v 3.8             ###########
################################################################

# Импортируем библиотеки
import numpy
import scipy.special
 
# Определяем класс нейронной сети
class neuralNetwork:
    
    # Инициализируем нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # Создаем матрицы весовых коэффициентов, wih и who
        # Веса внутри массивов равны w_i_j, где ссылка идет от узла i к узлу j на следующем уровне
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
 
        # Скорость обучения
        self.lr = learningrate
        
        # Функция активации - сигмойда
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
 
    # Обучаем нейронную сеть
    def train(self, inputs_list, targets_list):
        # Преобразовываем список входных данных в 2d массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # Вычисляем сигналы в скрытом слое
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Вычисляем сигналы, исходящие от скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Вычисление сигналов в конечном выходном слое
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Вычисление сигналов, из конечного выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        # Ошибка выходного уровня – это (целевой - фактический)
        output_errors = targets - final_outputs
        # Ошибка скрытого слоя – это output_errors, разделенные по весам, рекомбинированные в скрытых узлах 
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # Обновляем веса для связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # Обновляем веса для связей между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
 
    
    # Запрос к нейронной сети
    def query(self, inputs_list):
        # Преобразование списка входных данных в 2d массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # Вычисление сигналов в скрытом слое
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Вычисление сигналов, исходящих от скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Вычисление сигналов в конечном выходном слое
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Вычисление сигналов, поступающие из конечного выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
 
# Количество входных, скрытых и выходных узлов
input_nodes = 192
hidden_nodes = 90
output_nodes = 9
 
# Скорость обучения
learning_rate = 0.3
 
# Создаем экземпляр нейронной сети
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# Подгружаем тренировочные данные
training_data_file = open("name_file_training.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
 
# Обучаем нейронную сеть
# Epochs – это количество раз, когда набор обучающих данных используется для обучения
epochs = 100
 
for e in range(epochs):
    # Просмотреть все записи в наборе обучающих данных
    for record in training_data_list:
        # Разделяем записи запятыми ','
        all_values = record.split(',')
        # Масштабирование и сдвиг входных данных
        inputs = list(map(float, all_values[1:]))
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] является целевой меткой для этой записи
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
 
# Подгружаем тестовые данные
t_data_file = open("name_file_test.csv ", 'r')
t_data_list = t_data_file.readlines()
t_data_file.close()
 
# Передаем номер интересующей записи
all_values = t_data_list[0].split(',')

# Используем при желании просмотра тестируемых данных
# print(all_values)
res = n.query(numpy.asfarray(all_values[1:]))
label = numpy.argmax(res)

# Используем при желании просмотра значения на выходных узлах
# print(res)
print(label)
