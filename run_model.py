import pandas as pd
import pickle as pickle
import tensorflow as tf

class Model():

    last_timestamp = 0
    last_delta_timestamp = 0
    last_signal_short_name = None
    last_interpreted_signal_double = 0.0
    last_combined_signal = 0

    last_signal_short_name_list = None
    last_combined_signal_list = None

    sess = None

    feature_discretized_info = None
    signal_feature_discretized_rule = None
    signal_feature_discretization_name_order = None
    max_delta_timestamp = 0


    def load_Model(self, directory_path):
        sess = tf.Session()
        new_saver = tf.train.import_meta_graph(directory_path + 'model.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(directory_path))
        return sess


    def evaluate_event_with_trained_model(self, x_, tx_, y_, ty_, s_):
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        tx = graph.get_tensor_by_name("tx:0")
        y = graph.get_tensor_by_name("y:0")
        ty = graph.get_tensor_by_name("ty:0")
        s = graph.get_tensor_by_name("sequence-length:0")

        feed_dict = {x: [[x_], [x_]], tx: [[tx_], [tx_]], y: [[y_], [y_]] , ty: [[ty_], [ty_]], s: s_}
        test = self.sess.run(['py:0'], feed_dict)
        sum = 0.0
        for i in range(0, len(test[0])):
            sum += test[0][i]
        print(sum)


    def sort_value_to_bin(self, bins, signal_value):
        # Was passiert, sobald ein neuer Wert auftritt  -> Eigentlich gleich Aussreißer, da dieser Wert in den Trainingsdaten nicht bekannt war
        # - unendlich < x < bins[0]             -> neuer Wert:  -1
        # bins[len(bins) - 1] < x < unendlich   -> neuer Wert:  len(bins) + 1
        for i in range(0, len(bins)):
            if (bins[i] <= signal_value) and (bins[i + 1] >= signal_value):
                return float(i)


    def load_pickle_data(self, path):
        fileObject = open(path, 'rb')
        return pickle.load(fileObject)


    def load_model_and_discretized_rule(self, directory_path):
        self.sess = self.load_Model(directory_path)
        self.feature_discretized_info = self.load_pickle_data(directory_path + 'dicit.pickle')
        self.signal_feature_discretized_rule = self.feature_discretized_info['feature_discretization_rule']
        self.signal_feature_discretization_name_order = self.feature_discretized_info['feature_discretization_name_order']
        self.max_delta_timestamp = self.feature_discretized_info['max_delta_timestamp']


    def normalize_delta_time(self, delta_time):
        return delta_time / self.max_delta_timestamp

    def prepare_data(self, jsonDataList, count):
        if self.feature_discretized_info == None:
            directory_path = 'data/model_infos/kbins_uniform_max20Bins_and_features_102/'
            self.load_model_and_discretized_rule(directory_path)

        current_timestamp = jsonDataList[0]['timestamp']
        current_delta_time_normalized = self.normalize_delta_time(jsonDataList[0]['delta_time'])
        for jsonData in jsonDataList:
            current_signal_short_name = jsonData['signalShortName']
            current_interpreted_signal_double = jsonData['interpretedSignalDouble']

            if (self.last_timestamp == 0 and self.last_interpreted_signal_double == 0.0
                    and self.last_signal_short_name == None):
                # Beim ersten Signal könnne wir keine Vorhersage machen, da wir das vorrangestellte Signal nicht kennen
                # aus diesem Grund wird der Vorgang abgebrochen
                self.last_timestamp = current_timestamp
                self.last_signal_short_name = current_signal_short_name
                self.last_interpreted_signal_double = current_interpreted_signal_double
                self.last_delta_timestamp = 0
                return

            df = pd.DataFrame()

            # current_delta_timestamp = current_timestamp - self.last_timestamp
            # df['delta_time_diff'] = current_delta_timestamp
            df['delta_time_diff'] = current_delta_time_normalized
            df['interpreted_signal_double'] = current_interpreted_signal_double
            df['signal_short_name'] = current_signal_short_name

            # Es wird überprüft, ob das Signal beim Training bereits bekannt war
            if (current_signal_short_name in self.signal_feature_discretized_rule):
                bins = self.signal_feature_discretized_rule[current_signal_short_name]
                feature_discretized = self.sort_value_to_bin(bins, current_interpreted_signal_double)
                df['feature_discretized'] = feature_discretized
                combined = self.signal_feature_discretization_name_order[
                    '{}{}'.format(feature_discretized, current_signal_short_name)]
                df['combined'] = combined
                # rows_with_feature_discretized = rows_with_feature_discretized.append(row)

                # self.plot_data_before_and_after_kbins_discretizer(data, rows_with_feature_discretized, signal)
                # data_feature_discretized = data_feature_discretized.append(rows_with_feature_discretized)
                # df['combined'] = data_feature_discretized['combined']

            self.evaluate_event_with_trained_model(self.last_combined_signal, self.last_delta_timestamp,
                                                   combined, current_delta_time_normalized, [2, 2])

            self.last_combined_signal = combined

    def prepare_data(self, jsonData):
        if self.feature_discretized_info == None:
            directory_path = 'data/model_infos/kbins_uniform_max20Bins_and_features_102/'
            self.load_model_and_discretized_rule(directory_path)

        current_timestamp = jsonData['timestamp']
        current_normalized_delta_time = self.normalize_delta_time(jsonData['delta_time'])
        current_signal_short_name = jsonData['signalShortName']
        current_interpreted_signal_double = jsonData['interpretedSignalDouble']

        if (self.last_timestamp == 0 and self.last_interpreted_signal_double == 0.0
                and self.last_signal_short_name == None):
            # Beim ersten Signal könnne wir keine Vorhersage machen, da wir das vorrangestellte Signal nicht kennen
            # aus diesem Grund wird der Vorgang abgebrochen
            self.last_timestamp = current_timestamp
            self.last_signal_short_name = current_signal_short_name
            self.last_interpreted_signal_double = current_interpreted_signal_double
            self.last_delta_timestamp = 0
            return

        df = pd.DataFrame()

        #current_delta_timestamp = current_timestamp - self.last_timestamp
        #df['delta_time_diff'] = current_delta_timestamp
        df['delta_time_diff'] = current_normalized_delta_time
        df['interpreted_signal_double'] = current_interpreted_signal_double
        df['signal_short_name'] = current_signal_short_name

        # Es wird überprüft, ob das Signal beim Training bereits bekannt war
        if (current_signal_short_name in self.signal_feature_discretized_rule):
            bins = self.signal_feature_discretized_rule[current_signal_short_name]
            feature_discretized = self.sort_value_to_bin(bins, current_interpreted_signal_double)
            df['feature_discretized'] = feature_discretized
            combined = self.signal_feature_discretization_name_order[
                    '{}{}'.format(feature_discretized, current_signal_short_name)]
            df['combined'] = combined
            #rows_with_feature_discretized = rows_with_feature_discretized.append(row)

            # self.plot_data_before_and_after_kbins_discretizer(data, rows_with_feature_discretized, signal)
            #data_feature_discretized = data_feature_discretized.append(rows_with_feature_discretized)
            #df['combined'] = data_feature_discretized['combined']

            self.evaluate_event_with_trained_model(self.last_combined_signal, self.last_delta_timestamp,
                                                   combined, current_normalized_delta_time, [2,2])
            self.last_combined_signal = combined

        # Für das nächste Event
        self.last_timestamp = current_timestamp
        #self.last_delta_timestamp = current_delta_timestamp
        self.last_delta_timestamp = current_normalized_delta_time
        self.last_signal_short_name = current_signal_short_name
        self.last_interpreted_signal_double = current_interpreted_signal_double


    def shutdown(self):
        self.sess.close()