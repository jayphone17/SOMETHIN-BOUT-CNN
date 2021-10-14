# SOMETHIN-BOUT-CNN

一、network.py

（1）import mnist_loader

（2）training_data,validation_data,test_data = \
...mnist_loader.load_data_wrapper()

（3）import network

（4）net = network.Network([784,30,10])

（5）net.SGD(training_data,30,10,3.0,test_data = test_data)


二、network2.py

（1）import mnist_loader

（2）training_data, validation_data, test_data = \
...mnist_loader.load_data_wrapper()

（3）import network2

（4）net = network2.Network([784,30,10],cost = network2.CrossEntropyCost)

（5）net.large_weight_initializer()

（6）net.SGD(training_data, 30,10,0.5, evaluation_data = test_data, monitor_evaluation_accuracy = True)
