from py4j.java_gateway import JavaGateway, CallbackServerParameters
import json

from run_model import Model


class Reveiver():

    model = None
    lastTimeStamp = 0

    def __init__(self):
        print("Hello")
        self.model = Model()
        # directory_path = 'data/model_infos/kbins_uniform_max20Bins_and_features_102/'
        # self.model.load_pickle_data(directory_path  + 'dicit.pickle')
        # self.receive_data()
        # self.prepare_data()

    class Java:
        implements = ["com.bmw.mdr.anomaliedetecionbase.transmit.data.PythonReceiverInterface"]


    def receive_data(self, jsonData):
        #print("Hallo, Java ist da.")
        jsonDataList = json.loads(jsonData)
        for json_row in jsonDataList:
            json_row['delta_time_diff'] = json_row['timestamp'] - self.lastTimeStamp
        self.lastTimeStamp = json_row['timestamp']

        # TODO: Die deltaTime muss noch normiert werden
        self.model.prepare_data(jsonDataList)
        # self.model.prepare_data(json_row)


if __name__ == "__main__":
    receiver = Reveiver();
    gateway = JavaGateway(
        python_server_entry_point=receiver,
        python_proxy_port=44445,
        callback_server_parameters=CallbackServerParameters(address='10.5.2.42', port=44445))

