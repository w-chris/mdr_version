from py4j.java_gateway import JavaGateway, GatewayParameters
import json
import os

class SendRequiredJsonConfigFile():

    def __init__(self):
        data = self.prepare_data()


    def prepare_data(self):
        directory_path =  './data/signal_config/'
        for load_file in os.listdir(directory_path):
            file = open(directory_path + load_file, 'r')
            data = json.load(file)
            file.close()
            self.send_data(data)


    def send_data(self, data):
        gateway = JavaGateway(
            gateway_parameters=GatewayParameters(address="10.5.2.10", port=44444))
        gateway.entry_point.receiveConfig(str(data))


SendRequiredJsonConfigFile().__init__()