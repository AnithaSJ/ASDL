import tflearn
import numpy

class Code_Completion_Baseline:

    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]
    
    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}
    
    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
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
        """
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                if idx > 0:
                    token_string = self.token_to_string(token)
                    previous_token_string = self.token_to_string(token_list[idx - 1])
                    xs.append(self.one_hot(previous_token_string))
                    ys.append(self.one_hot(token_string))
        """
        #Prepare data frame of 5 words
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                zero_vector = [0] * len(self.string_to_number)
                if idx == 0:
                   word1 = zero_vector
                   word2 = zero_vector
                   word3 = zero_vector
                   word4 = zero_vector
                   word5 = zero_vector
                   try:
                    word6 = self.one_hot(self.token_to_string(token_list[idx + 1]))
                   except IndexError:
                    word6 = zero_vector
                   try:
                    word7 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                   except IndexError:
                    word7 = zero_vector
                   try:
                    word8 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                   except IndexError:
                    word8 = zero_vector
                   try:
                    word9 = self.one_hot(self.token_to_string(token_list[idx + 4]))
                   except IndexError:
                    word9 = zero_vector
                   try:
                    word10 = self.one_hot(self.token_to_string(token_list[idx + 5]))
                   except IndexError:
                    word10 = zero_vector
                   
                if idx == 1:
                   word1 = zero_vector
                   word2 = zero_vector
                   word3 = zero_vector
                   word4 = zero_vector
                   word5 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                   try:
                    word6 = self.one_hot(self.token_to_string(token_list[idx + 1]))
                   except IndexError:
                    word6 = zero_vector
                   try:
                    word7 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                   except IndexError:
                    word7 = zero_vector
                   try:
                    word8 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                   except IndexError:
                    word8 = zero_vector
                   try:
                    word9 = self.one_hot(self.token_to_string(token_list[idx + 4]))
                   except IndexError:
                    word9 = zero_vector
                   try:
                    word10 = self.one_hot(self.token_to_string(token_list[idx + 5]))
                   except IndexError:
                    word10 = zero_vector
                   
                if idx == 2:
                   word1 = zero_vector
                   word2 = zero_vector
                   word3 = zero_vector
                   word4 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                   word5 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                   try:
                    word6 = self.one_hot(self.token_to_string(token_list[idx + 1]))
                   except IndexError:
                    word6 = zero_vector
                   try:
                    word7 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                   except IndexError:
                    word7 = zero_vector
                   try:
                    word8 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                   except IndexError:
                    word8 = zero_vector
                   try:
                    word9 = self.one_hot(self.token_to_string(token_list[idx + 4]))
                   except IndexError:
                    word9 = zero_vector
                   try:
                    word10 = self.one_hot(self.token_to_string(token_list[idx + 5]))
                   except IndexError:
                    word10 = zero_vector
                
                if idx == 3:
                   word1 = zero_vector
                   word2 = zero_vector
                   word3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                   word4 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                   word5 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                   try:
                    word6 = self.one_hot(self.token_to_string(token_list[idx + 1]))
                   except IndexError:
                    word6 = zero_vector
                   try:
                    word7 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                   except IndexError:
                    word7 = zero_vector
                   try:
                    word8 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                   except IndexError:
                    word8 = zero_vector
                   try:
                    word9 = self.one_hot(self.token_to_string(token_list[idx + 4]))
                   except IndexError:
                    word9 = zero_vector
                   try:
                    word10 = self.one_hot(self.token_to_string(token_list[idx + 5]))
                   except IndexError:
                    word10 = zero_vector
                
                if idx == 4:
                   word1 = zero_vector
                   word2 = self.one_hot(self.token_to_string(token_list[idx - 4]))
                   word3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                   word4 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                   word5 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                   try:
                    word6 = self.one_hot(self.token_to_string(token_list[idx + 1]))
                   except IndexError:
                    word6 = zero_vector
                   try:
                    word7 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                   except IndexError:
                    word7 = zero_vector
                   try:
                    word8 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                   except IndexError:
                    word8 = zero_vector
                   try:
                    word9 = self.one_hot(self.token_to_string(token_list[idx + 4]))
                   except IndexError:
                    word9 = zero_vector
                   try:
                    word10 = self.one_hot(self.token_to_string(token_list[idx + 5]))
                   except IndexError:
                    word10 = zero_vector
                    
                if idx > 4:
                   word1 = self.one_hot(self.token_to_string(token_list[idx - 5]))
                   word2 = self.one_hot(self.token_to_string(token_list[idx - 4]))
                   word3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                   word4 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                   word5 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                   try:
                    word6 = self.one_hot(self.token_to_string(token_list[idx + 1]))
                   except IndexError:
                    word6 = zero_vector
                   try:
                    word7 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                   except IndexError:
                    word7 = zero_vector
                   try:
                    word8 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                   except IndexError:
                    word8 = zero_vector
                   try:
                    word9 = self.one_hot(self.token_to_string(token_list[idx + 4]))
                   except IndexError:
                    word9 = zero_vector
                   try:
                    word10 = self.one_hot(self.token_to_string(token_list[idx + 5]))
                   except IndexError:
                    word10 = zero_vector
                   
                
                xs.append([word1,word2,word3,word4,word5,word6,word7,word8,word9,word10])
                #xs.append([word1,word2,word3,word4,word5])
                ys.append(self.one_hot(token_string))
                
        print("x,y pairs: " + str(len(xs)))        
        return (xs, ys)

    def create_network(self):
        #LSTM
        self.net = tflearn.input_data(shape=[None, 10, len(self.string_to_number)])
        self.net = tflearn.lstm(self.net, 512, return_seq=True)
        self.net = tflearn.lstm(self.net, 256, return_seq=True)
        self.net = tflearn.lstm(self.net, 128)
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax', bias=True, trainable=True)
        self.net = tflearn.regression(self.net, optimizer='adam', loss='categorical_crossentropy')
        self.model = tflearn.DNN(self.net)
        """
        #original
            self.net = tflearn.input_data(shape=[None, len(self.string_to_number)])
            self.net = tflearn.fully_connected(self.net, 32)
            self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax')
            self.net = tflearn.regression(self.net)
            self.model = tflearn.DNN(self.net)
        """
        
        
    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)
    
    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        self.model.fit(xs, ys, n_epoch=10, batch_size=512, show_metric=True)
        self.model.save(model_file)
        
    def query(self, prefix, suffix):
        """ 
        previous_token_string = self.token_to_string(prefix[-1])
        x = self.one_hot(previous_token_string)
        y = self.model.predict([x])
        predicted_seq = y[0]
        
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist() 
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        """
        
        x=[]
        zero_vector = [0] * len(self.string_to_number)
        idx= len(prefix)
        if idx == 0:
           word1 = zero_vector
           word2 = zero_vector
           word3 = zero_vector
           word4 = zero_vector
           word5 = zero_vector
           try:
            word6 = self.one_hot(self.token_to_string(suffix[0]))
           except IndexError:
            word6 = zero_vector
           try:
            word7 = self.one_hot(self.token_to_string(suffix[1]))
           except IndexError:
            word7 = zero_vector
           try:
            word8 = self.one_hot(self.token_to_string(suffix[2]))
           except IndexError:
            word8 = zero_vector
           try:
            word9 = self.one_hot(self.token_to_string(suffix[3]))
           except IndexError:
            word9 = zero_vector
           try:
            word10 = self.one_hot(self.token_to_string(suffix[4]))
           except IndexError:
            word10 = zero_vector
                   
        if idx == 1:
           word1 = zero_vector
           word2 = zero_vector
           word3 = zero_vector
           word4 = zero_vector
           word5 = self.one_hot(self.token_to_string(prefix[- 1]))
           try:
            word6 = self.one_hot(self.token_to_string(suffix[0]))
           except IndexError:
            word6 = zero_vector
           try:
            word7 = self.one_hot(self.token_to_string(suffix[1]))
           except IndexError:
            word7 = zero_vector
           try:
            word8 = self.one_hot(self.token_to_string(suffix[2]))
           except IndexError:
            word8 = zero_vector
           try:
            word9 = self.one_hot(self.token_to_string(suffix[3]))
           except IndexError:
            word9 = zero_vector
           try:
            word10 = self.one_hot(self.token_to_string(suffix[4]))
           except IndexError:
            word10 = zero_vector
           
        if idx == 2:
           word1 = zero_vector
           word2 = zero_vector
           word3 = zero_vector
           word4 = self.one_hot(self.token_to_string(prefix[-2]))
           word5 = self.one_hot(self.token_to_string(prefix[-1]))
           try:
            word6 = self.one_hot(self.token_to_string(suffix[0]))
           except IndexError:
            word6 = zero_vector
           try:
            word7 = self.one_hot(self.token_to_string(suffix[1]))
           except IndexError:
            word7 = zero_vector
           try:
            word8 = self.one_hot(self.token_to_string(suffix[2]))
           except IndexError:
            word8 = zero_vector
           try:
            word9 = self.one_hot(self.token_to_string(suffix[3]))
           except IndexError:
            word9 = zero_vector
           try:
            word10 = self.one_hot(self.token_to_string(suffix[4]))
           except IndexError:
            word10 = zero_vector
        
        if idx == 3:
           word1 = zero_vector
           word2 = zero_vector
           word3 = self.one_hot(self.token_to_string(prefix[-3]))
           word4 = self.one_hot(self.token_to_string(prefix[- 2]))
           word5 = self.one_hot(self.token_to_string(prefix[- 1]))
           try:
            word6 = self.one_hot(self.token_to_string(suffix[0]))
           except IndexError:
            word6 = zero_vector
           try:
            word7 = self.one_hot(self.token_to_string(suffix[1]))
           except IndexError:
            word7 = zero_vector
           try:
            word8 = self.one_hot(self.token_to_string(suffix[2]))
           except IndexError:
            word8 = zero_vector
           try:
            word9 = self.one_hot(self.token_to_string(suffix[3]))
           except IndexError:
            word9 = zero_vector
           try:
            word10 = self.one_hot(self.token_to_string(suffix[4]))
           except IndexError:
            word10 = zero_vector
        
        if idx == 4:
           word1 = zero_vector
           word2 = self.one_hot(self.token_to_string(prefix[- 4]))
           word3 = self.one_hot(self.token_to_string(prefix[- 3]))
           word4 = self.one_hot(self.token_to_string(prefix[- 2]))
           word5 = self.one_hot(self.token_to_string(prefix[- 1]))
           try:
            word6 = self.one_hot(self.token_to_string(suffix[0]))
           except IndexError:
            word6 = zero_vector
           try:
            word7 = self.one_hot(self.token_to_string(suffix[1]))
           except IndexError:
            word7 = zero_vector
           try:
            word8 = self.one_hot(self.token_to_string(suffix[2]))
           except IndexError:
            word8 = zero_vector
           try:
            word9 = self.one_hot(self.token_to_string(suffix[3]))
           except IndexError:
            word9 = zero_vector
           try:
            word10 = self.one_hot(self.token_to_string(suffix[4]))
           except IndexError:
            word10 = zero_vector
        
        if idx > 4:
           word1 = self.one_hot(self.token_to_string(prefix[- 5]))
           word2 = self.one_hot(self.token_to_string(prefix[- 4]))
           word3 = self.one_hot(self.token_to_string(prefix[- 3]))
           word4 = self.one_hot(self.token_to_string(prefix[- 2]))
           word5 = self.one_hot(self.token_to_string(prefix[- 1]))
           try:
            word6 = self.one_hot(self.token_to_string(suffix[0]))
           except IndexError:
            word6 = zero_vector
           try:
            word7 = self.one_hot(self.token_to_string(suffix[1]))
           except IndexError:
            word7 = zero_vector
           try:
            word8 = self.one_hot(self.token_to_string(suffix[2]))
           except IndexError:
            word8 = zero_vector
           try:
            word9 = self.one_hot(self.token_to_string(suffix[3]))
           except IndexError:
            word9 = zero_vector
           try:
            word10 = self.one_hot(self.token_to_string(suffix[4]))
           except IndexError:
            word10 = zero_vector
        
        x.append([word1,word2,word3,word4,word5,word6,word7,word8,word9,word10])
        #x.append([word1,word2,word3,word4,word5])
        y = self.model.predict(x)
        
        predicted_seq_1 = y[0]
        if type(predicted_seq_1) is numpy.ndarray:
            predicted_seq_1 = predicted_seq_1.tolist() 
        best_number = predicted_seq_1.index(max(predicted_seq_1))
        best_string = self.number_to_string[best_number]
        predicted_firsthole = self.one_hot(best_string)
        
        x.append([word2,word3,word4,word5,predicted_firsthole,word6,word7,word8,word9,word10])
        #x.append([word2,word3,word4,word5,predicted_firsthole])
        y = self.model.predict(x)
        predicted_seq_2 = y[0]
        
        x.append([word1,word2,word3,word4,word5,predicted_firsthole,word6,word7,word8,word9])
        y = self.model.predict(x)
        predicted_seq_3 = y[0]
               
        best_token = []
        
        if type(predicted_seq_1) is numpy.ndarray:
            predicted_seq_1 = predicted_seq_1.tolist() 
        best_number = predicted_seq_1.index(max(predicted_seq_1))
        best_string = self.number_to_string[best_number]
        best_token_1 = self.string_to_token(best_string)
        best_token.append(best_token_1)
        
        if type(predicted_seq_2) is numpy.ndarray:
            predicted_seq_2 = predicted_seq_2.tolist() 
        best_number = predicted_seq_2.index(max(predicted_seq_2))
        best_string = self.number_to_string[best_number]
        best_token_2 = self.string_to_token(best_string)
        best_token.append(best_token_2)
        
        if type(predicted_seq_3) is numpy.ndarray:
            predicted_seq_3 = predicted_seq_3.tolist() 
        best_number = predicted_seq_3.index(max(predicted_seq_3))
        best_string = self.number_to_string[best_number]
        best_token_3 = self.string_to_token(best_string)
        best_token.append(best_token_3)
        
        return best_token
    
