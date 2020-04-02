import numpy as np
import pandas as pd
from numpy.fft import fft
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Features:

    def __init__(self,df,time):
        self.df = df
        self.time = time
        self.feature_matrix = []
        self.decomposed_feature_matrix = []
    
    def features_extraction(self):
        

        for index,row in self.df.iterrows():

            ## glucose level
            feature_vector = []
            glucose_vector = []
            velocity_vector = []
            rms = []

            for index in range(len(row)):
                glucose_vector += [row[index]]    
            
            # print(glucose_vector)
            max_val = np.nanmax(glucose_vector)
            min_val = np.nanmin(glucose_vector)
            variance = np.nanvar(glucose_vector)
            # print(max_val,min_val,variance)
            feature_vector.append(max_val)
            feature_vector.append(min_val)
            feature_vector.append(variance)
            
            ## cgm velocity
            for index in range(len(row)-1):
                velocity_vector += [(row[index+1]-row[index])]    
            
            # print(velocity_vector)
            max_val = np.nanmax(velocity_vector)
            min_val = np.nanmin(velocity_vector)
            variance = np.nanvar(velocity_vector)
            feature_vector.append(max_val)
            feature_vector.append(min_val)
            feature_vector.append(variance)
            # print(feature_vector)


            ## rms
            for index in range(len(row)-5):
                rms_sum = sum([x*x for x in row[index:index+5]]) 
                rms_sum /= 5
                rms_sum = rms_sum ** 0.5
                rms.append(rms_sum)

            max_val = np.nanmax(rms)
            min_val = np.nanmin(rms)
            variance = np.nanvar(rms)
            feature_vector.append(max_val)
            feature_vector.append(min_val)
            feature_vector.append(variance)

            ## fft
            cgm_fft = np.abs(np.fft.fft(row))
            cgm_fft = cgm_fft.tolist()
            # print(cgm_fft)

            max_val = np.nanmax(cgm_fft)
            min_val = np.nanmin(cgm_fft)
            variance = np.nanvar(cgm_fft)
            feature_vector.append(max_val)
            feature_vector.append(min_val)
            feature_vector.append(variance)
            # print(feature_vector)
            if not np.isnan(feature_vector).any():
                self.feature_matrix.append(feature_vector)
        
        self.save_to_csv()
        # print(self.feature_matrix)
        self.normalized()

        # print(self.feature_matrix)
        # print(len(self.feature_matrix))
        # print(len(self.feature_matrix[0]))


    def normalized(self):
        self.feature_matrix = np.array(self.feature_matrix)
        self.feature_matrix /= self.feature_matrix.sum(axis=1)[:, np.newaxis]
        self.feature_matrix = self.feature_matrix.tolist()
    
    def pca_decomposition(self):
        pca = PCA(n_components=5, random_state=43)
        self.decomposed_feature_matrix = pca.fit_transform(self.feature_matrix)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.savefig('explained_variance.png')
        plt.show()
        return self.decomposed_feature_matrix

    def save_to_csv(self):
        file = pd.DataFrame(self.feature_matrix,columns=['Glucose_max','Glucose_min','Glucose_variance','velocity_max','velocity_min','velocity_variance','rms_max','rms_min','rms_variance','fft_max','fft_min','fft_variance'])
        # print(file)
        file.to_csv('feature_matrix.csv',index=False)

    def plot_time_series(self):
        new_list = list(map(list,zip(*self.decomposed_feature_matrix)))
        # print(self.decomposed_feature_matrix)
        for index in range(len(new_list)):
            plt.plot(range(len(new_list[0])), new_list[index])
            plt.savefig(fname='component%s.png' % str(index+1))
            plt.show()



        