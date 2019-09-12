from py4j.java_gateway import JavaGateway, CallbackServerParameters
import csv
import json
from os import path

from run_model import Model


class Reveiver():

    model = None
    lastTimeStamp = 0

    def __init__(self):
        print("Hello")
        self.model = Model()
        directory_path = 'data/model_infos/kbins_uniform_max20Bins_and_features_102/'
        self.model.load_pickle_data(directory_path  + 'dicit.pickle')
        # self.receive_data()
        # self.prepare_data()

    class Java:
        implements = ["com.bmw.mdr.anomaliedetecionbase.transmit.data.PythonReceiverInterface"]


    def receive_data(self, jsonData):
        #print("Hallo, Java ist da.")
        jsonDataList = json.loads(jsonData)
        for json_row in jsonDataList:
            json_row['delta_time_diff'] = json_row['timestamp'] - self.lastTimeStamp
            if (path.exists("outputfile.csv") == False):
                outputFile = open("outputfile.csv", 'w+', newline='')
                output = csv.writer(outputFile)
                output.writerow(json_row.keys())
            else:
                outputFile = open("outputfile.csv", 'a', newline='')
                output = csv.writer(outputFile)

            output.writerow(json_row.values())
            outputFile.close()

        self.lastTimeStamp = json_row['timestamp']
        # TODO: Die deltaTime muss noch normiert werden
        # self.model.prepare_data(jsonDataList)
        self.model.prepare_data(json_row)


if __name__ == "__main__":
    receiver = Reveiver();
    gateway = JavaGateway(
        python_server_entry_point=receiver,
        python_proxy_port=44445,
        callback_server_parameters=CallbackServerParameters(address='10.5.2.42', port=44445))

'''
    # Load old model functions
    def load_bins(self, signal_short_name):
        fileName = "{}_bins.out".format(signal_short_name)
        data = np.loadtxt(fileName)
        return data


    def load_pickle_data(self, path):
        fileObject = open(path, 'rb')
        return pickle.load(fileObject)


    def load1(self, df, directory_path):
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(directory_path + 'model.ckpt.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(directory_path))

            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            tx = graph.get_tensor_by_name("tx:0")
            y = graph.get_tensor_by_name("y:0")
            ty = graph.get_tensor_by_name("ty:0")
            s = graph.get_tensor_by_name("sequence-length:0")

            feed_dict = {x: [[5],[1]], tx: [[0.00050], [0.00050]], y: [[12],[1]], ty: [[0.200],[0.200]], s:[2, 2]}
            test = sess.run(['py:0'], feed_dict)
            sum = 0.0
            for i in range(0, len(test[0])):
                sum += test[0][i]
            print(sum)


    def load_old_model(self, df, directory_path):
        self.load1(df, directory_path)

        #imported_graph = tf.train.import_meta_graph(directory_path + 'model.ckpt.meta')
        #x = tf.placeholder(tf.int32, shape=[None, 1], name='x')
        #tx = tf.placeholder(tf.float32, shape=[None, 1], name='tx')
        #y = tf.placeholder(tf.int32, shape=[None, 1], name='y')
        #ty = tf.placeholder(tf.float32, shape=[None, 1], name='ty')
        #s= tf.placeholder(tf.float32, shape=[None, 1], name='s')

        #with tf.Session() as sess:
        #    imported_graph.restore(sess, directory_path + 'model.ckpt')

            #feed_dict = {x: [12], tx: [2.91563365e-0], y: [12], ty: [0.00250], s: 32}
            #sess.run(['py:0'], feed_dict=feed_dict)

            #imported_graph.restore(sess, directory_path + 'model.ckpt')

            #x = tf.placeholder(tf.int32, [], 'x')
            #tx = tf.placeholder(tf.float32, [], 'tx')
            #y = tf.placeholder(tf.int32, [], 'y')
            #ty = tf.placeholder(tf.float32, [], 'ty')
            #s = tf.placeholder(tf.int32, [], 'sequence-length')

            #x_ = 12, 12
            #tx_ = 2.91563365e-05,  4.81316315e-05
            #y_ = 12, 12
            #ty_ = 4.81316315e-05,  2.41582931e-04
            #s_ = 32
            #feed_dict = {x: x_, tx: tx_, y: y_, ty: ty_, s: s_}
            #print(sess.run([], feed_dict))



            ## import the graph from the file
            #imported_graph = tf.train.import_meta_graph(directory_path + 'model.ckpt.meta')
            ## list all the tensors in the graph
            #for tensor in tf.get_default_graph().get_operations():
            #    print(tensor.name)

            # First let's load meta graph and restore weights
            #saver = tf.train.import_meta_graph(directory_path + 'model.ckpt.meta')
            #saver.restore(sess, tf.train.latest_checkpoint(directory_path))
            #all_vars = tf.get_collection('vars')

            # print all tensors in checkpoint file
            # chkp.print_tensors_in_checkpoint_file(directory_path + 'model.ckpt', tensor_name='', all_tensors=True)

            #saver.restore(sess, directory_path + 'model')
            # x_out, xt_out, y_out, yt_out = sess.run([x, xt, y, yt])

            # x = 5
            # tx = 0.00110
            # y = 2
            # ty = 0.00250
            # s = 1
            # feed_dict = {x: x, tx: tx, y: y, ty: ty, s: s}
            # print(sess.run([], feed_dict))

            # x, tx, y, ty, s = d
            # feed = {model.x: x, model.tx: tx, model.y: y, model.ty: ty, model.s: s}
            # Test Model with current data
            # loss, accuracy = sess.run([model.loss, model.accuracy], feed)


    def sort_value_to_bin(self, bins, data):
        # Was passiert, sobald ein neuer Wert auftritt  -> Eigentlich gleich Aussreißer, da dieser Wert in den Trainingsdaten nicht bekannt war
        # - unendlich < x < bins[0]             -> neuer Wert:  -1
        # bins[len(bins) - 1] < x < unendlich   -> neuer Wert:  len(bins) + 1
        value = data['interpreted_signal_double']
        for i in range(0, len(bins)):
            if (bins[i] <= value) and (bins[i + 1] >= value):
                return float(i)


    def prepare_data(self):
        self.prepare_data(pd.read_csv("outputfile.csv"))


    def prepare_data(self, jsonData):
        # Pfad zum gespeicherten Modell, sowie den Bins
        directory_path = 'data/model_infos/kbins_uniform_max20Bins_and_features_102/'

        #df = pd.read_csv("outputfile.csv")
        current_timestamp = jsonData['timestamp']
        current_signal_short_name = jsonData['signalShortName']
        current_interpreted_signal_double = jsonData['interpretedSignalDouble']

        # Es muss so abgebildet werden, das man sich die vorherigen Signale merkt
        # und dann aus diesen Daten die unteschiedlichen Werte wie Zeitstempel und Signal heraus ableiten lassen

        last_timestamp = 0
        last_signal_short_name = 0
        last_interpreted_signal_double = 0

        if (last_timestamp == 0 and last_signal_short_name == 0 and last_interpreted_signal_double == 0):
            # Beim ersten Signal könnne wir keine Vorhersage machen, da wir das vorrangestellte Signal nicht kennen
            # aus diesem Grund wird der Vorgang abgebrochen
            last_timestamp = current_timestamp
            last_signal_short_name = current_signal_short_name
            last_interpreted_signal_double = current_interpreted_signal_double
            return

        df = pd.DataFrame()

        df['delta_time_diff'] = current_timestamp - last_timestamp

        feature_discretized_info = self.load_pickle_data(directory_path + 'dicit.pickle')
        signal_feature_discretized_rule = feature_discretized_info['feature_discretization_rule']
        signal_feature_discretization_name_order = feature_discretized_info['feature_discretization_name_order']

        data_feature_discretized = pd.DataFrame()

        rows_with_feature_discretized = pd.DataFrame()
        data = df[df.signal_short_name.eq(current_signal_short_name)]
        bins = signal_feature_discretized_rule[current_signal_short_name]
        for _, row in data.iterrows():
            feature_discretized = self.sort_value_to_bin(bins, row)
            row['feature_discretized'] = feature_discretized
            combined = signal_feature_discretization_name_order['{}{}'.format(feature_discretized, current_signal_short_name)]
            row['combined'] = combined
            rows_with_feature_discretized = rows_with_feature_discretized.append(row)

        # self.plot_data_before_and_after_kbins_discretizer(data, rows_with_feature_discretized, signal)
        data_feature_discretized = data_feature_discretized.append(rows_with_feature_discretized)
        df['combined'] = data_feature_discretized['combined']

        self.load_old_model(df, directory_path)

        last_timestamp = current_timestamp
        last_signal_short_name = current_signal_short_name
        last_interpreted_signal_double = current_interpreted_signal_double

        
        df = jsonData
        df['delta_time_diff'] = (df['timestamp'] - df['timestamp'].shift()).fillna(0)
        signal_short_name_list = df['signal_short_name'].drop_duplicates().values.tolist()

        feature_discretized_info = self.load_pickle_data(directory_path + 'dicit.pickle')
        signal_feature_discretized_rule = feature_discretized_info['feature_discretization_rule']
        signal_feature_discretization_name_order = feature_discretized_info['feature_discretization_name_order']

        data_feature_discretized = pd.DataFrame()

        for signal in signal_short_name_list:
            rows_with_feature_discretized = pd.DataFrame()
            data = df[df.signal_short_name.eq(signal)]
            if signal not in signal_feature_discretized_rule.keys():
                continue
            bins = signal_feature_discretized_rule[signal]
            for _, row in data.iterrows():
                feature_discretized = self.sort_value_to_bin(bins, row)
                row['feature_discretized'] = feature_discretized
                combined = signal_feature_discretization_name_order['{}{}'.format(feature_discretized, signal)]
                row['combined'] = combined
                rows_with_feature_discretized = rows_with_feature_discretized.append(row)

            #self.plot_data_before_and_after_kbins_discretizer(data, rows_with_feature_discretized, signal)
            data_feature_discretized = data_feature_discretized.append(rows_with_feature_discretized)

        #combined = (data_feature_discretized['feature_discretized'].map(str)
        #           + data_feature_discretized['signal_short_name'].map(str)).astype('category').cat.codes
        df['combined'] = data_feature_discretized['combined']

        self.load_old_model(directory_path)
        


        # Es muss bekannt sein:
        # welche Werte liefert ein Signal maximal
        # Einteilung in Bereiche zur FeatureDiskretisierung müssen bekannt sein -> Problem diese werden beim Trainieren des Netzes erzeugt müssen aber bei der Analyse auch bekannt sein
    '''