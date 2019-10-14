import os
import json
import csv
from os import path


class TraceLabConverter():

    lastTimeStamp = 0

    def read_converted_json_traces(self, directory_path):
        result_file_path = "traclab_rawData/output/"
        if not os.path.exists(result_file_path):
            os.makedirs(result_file_path)

        for load_file in os.listdir(directory_path):
            if not os.path.isfile(directory_path + load_file):
                continue
            file = open(directory_path + load_file, 'r')
            data = json.load(file)
            i = 0
            while i < len(data):
                self.receive_data(data[i], result_file_path + load_file.split('.')[0] + 'converted.csv', i)
                i = i + 1

    def receive_data(self, jsonData, file_name, count):
        #print("Hallo, Java ist da.")
        jsonDataList = json.loads(jsonData)

        with open(file_name, 'a+', newline='') as f:
            w = csv.DictWriter(f, jsonDataList[0].keys())
            if (count == 0):
                w.writeheader()
            for json_row in jsonDataList:
                w.writerow(json_row)

if __name__ == "__main__":
    source_directory_path = './traclab_rawData/'

    converter = TraceLabConverter();
    # Generate Test Data for Anomalie-Detection
    converter.read_converted_json_traces(source_directory_path)