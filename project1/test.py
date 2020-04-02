from preprocess import Preprocess
from features import Features
import pandas as pd

cgm_series_lunch = Preprocess('D:\Desktop\ASU\CSE572\DataFolder\CGMSeriesLunchPat1.csv')
time_series_lunch = Preprocess('D:\Desktop\ASU\CSE572\DataFolder\CGMDatenumLunchPat1.csv')
cgm_series_lunch1 = cgm_series_lunch.get_dataframe()
time_series_lunch1 = time_series_lunch.get_dataframe() 
# print(cgm_series_lunch1)
# print(time_series_lunch1)

feature = Features(cgm_series_lunch1, time_series_lunch1)
feature.features_extraction()
# feature.save_to_csv()
final_pca = pd.DataFrame(feature.pca_decomposition())
final_pca.to_csv('final_pca.csv',index=False)
feature.plot_time_series()