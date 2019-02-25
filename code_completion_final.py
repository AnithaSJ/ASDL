import tflearn
import numpy


class Code_Completion_Final:
    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}

    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector

    def padding_zero(self):
        vector = [0] * len(self.string_to_number)
        return vector

    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        # prepare x,y pairs
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                
                if idx == 0:
                    prefix_string_5 = self.padding_zero()
                    prefix_string_4 = self.padding_zero()
                    prefix_string_3 = self.padding_zero()
                    prefix_string_2 = self.padding_zero()
                    prefix_string_1 = self.padding_zero()

                
                if idx == 1:
                    prefix_string_5 = self.padding_zero()
                    prefix_string_4 = self.padding_zero()
                    prefix_string_3 = self.padding_zero()
                    prefix_string_2 = self.padding_zero()
                    prefix_string_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

               
                if idx == 2:
                    prefix_string_5 = self.padding_zero()
                    prefix_string_4 = self.padding_zero()
                    prefix_string_3 = self.padding_zero()
                    prefix_string_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_string_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

               
                if idx == 3:
                    prefix_string_5 = self.padding_zero()
                    prefix_string_4 = self.padding_zero()
                    prefix_string_3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    prefix_string_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_string_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

               
                if idx == 4:
                    prefix_string_5 = self.padding_zero()
                    prefix_string_4 = self.one_hot(self.token_to_string(token_list[idx - 4]))
                    prefix_string_3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    prefix_string_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_string_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

                
                if idx > 4:
                    prefix_string_5 = self.one_hot(self.token_to_string(token_list[idx - 5]))
                    prefix_string_4 = self.one_hot(self.token_to_string(token_list[idx - 4]))
                    prefix_string_3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    prefix_string_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_string_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

                xs.append([prefix_string_5, prefix_string_4, prefix_string_3, prefix_string_2,
                           prefix_string_1])

                token_string = self.token_to_string(token)
                ys.append(self.one_hot(token_string))

        print("x,y pairs: " + str(len(xs)))
        return (xs, ys)

    def create_network(self):
        self.net = tflearn.input_data(shape=[None, 5, len(self.string_to_number)])
        self.net = tflearn.lstm(self.net, 512, return_seq=True)
        self.net = tflearn.lstm(self.net, 256, return_seq=True)
        self.net = tflearn.lstm(self.net, 128)
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax', bias=True, trainable=True)
        self.net = tflearn.regression(self.net, optimizer='adam', loss='categorical_crossentropy', learning_rate= 1e-2)
        self.model = tflearn.DNN(self.net)

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)

    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        self.model.fit(xs, ys, n_epoch=10, batch_size=512, show_metric=True)
        self.model.save(model_file)

    def predict(self, x):
        y = self.model.predict([x])
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return best_token

    def query(self, prefix, suffix): 
        
        
        if len(prefix) == 0:
            prefix_string_5 = self.padding_zero()
            prefix_string_4 = self.padding_zero()
            prefix_string_3 = self.padding_zero()
            prefix_string_2 = self.padding_zero()
            prefix_string_1 = self.padding_zero()

       
        if len(prefix) == 1:
            prefix_string_5 = self.padding_zero()
            prefix_string_4 = self.padding_zero()
            prefix_string_3 = self.padding_zero()
            prefix_string_2 = self.padding_zero()
            prefix_string_1 = self.one_hot(self.token_to_string(prefix[-1]))

        
        if len(prefix) == 2:
            prefix_string_5 = self.padding_zero()
            prefix_string_4 = self.padding_zero()
            prefix_string_3 = self.padding_zero()
            prefix_string_2 = self.one_hot(self.token_to_string(prefix[-2]))
            prefix_string_1 = self.one_hot(self.token_to_string(prefix[-1]))

        
        if len(prefix) == 3:
            prefix_string_5 = self.padding_zero()
            prefix_string_4 = self.padding_zero()
            prefix_string_3 = self.one_hot(self.token_to_string(prefix[-3]))
            prefix_string_2 = self.one_hot(self.token_to_string(prefix[-2]))
            prefix_string_1 = self.one_hot(self.token_to_string(prefix[-1]))

       
        if len(prefix) == 4:
            prefix_string_5 = self.padding_zero()
            prefix_string_4 = self.one_hot(self.token_to_string(prefix[-4]))
            prefix_string_3 = self.one_hot(self.token_to_string(prefix[-3]))
            prefix_string_2 = self.one_hot(self.token_to_string(prefix[-2]))
            prefix_string_1 = self.one_hot(self.token_to_string(prefix[-1]))

        
        if len(prefix) > 4:
            prefix_string_5 = self.one_hot(self.token_to_string(prefix[-5]))
            prefix_string_4 = self.one_hot(self.token_to_string(prefix[-4]))
            prefix_string_3 = self.one_hot(self.token_to_string(prefix[-3]))
            prefix_string_2 = self.one_hot(self.token_to_string(prefix[-2]))
            prefix_string_1 = self.one_hot(self.token_to_string(prefix[-1]))

        # this holds the list of predicted tokens
        best_token = []
        
        x = [prefix_string_5, prefix_string_4, prefix_string_3, prefix_string_2, prefix_string_1]        
        predicted_seq_1 = self.predict(x)
        best_token.append(predicted_seq_1)
             
        return best_token
