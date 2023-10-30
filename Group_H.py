# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:07:51 2023

@author: Patricio
"""
import pandas as pd
from sklearn.model_selection import train_test_split


df_train = pd.read_csv("ImdbRunAge0531_train.csv")
df_test =  pd.read_csv("ImdbRunAge0531_test.csv")

X_train = df_train.drop(['imdb_score', 'tmdb_score', 'title', 'description','Unnamed: 0.1','Unnamed: 0','imdb_level'], axis=1)
X_test = df_test.drop(['imdb_score', 'tmdb_score', 'title', 'description','Unnamed: 0.1','Unnamed: 0','imdb_level'], axis=1)
y_train = df_train['imdb_score']
y_test = df_test['imdb_score']


X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



"""用這個方式找到的最佳超參數也許不是Global Min但至少是個Local Min"""

######### 用找到的最佳超參數組合 訓練整個訓練集，並對測試集進行預測 #########


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 創建隨機森林回歸模型
rfFinal = RandomForestRegressor(n_estimators=120, max_features=12, min_samples_leaf=4, random_state=42)

# 使用訓練集進行訓練
rfFinal.fit(X_train, y_train)

# 預測測試集
y_pred = rfFinal.predict(X_test)

# 計算均方誤差 (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# 計算平均絕對誤差 (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

"""
換成XGBoost
"""


import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 創建空列表來儲存每次迭代的 MSE 和 MAE
mse_values = []
mae_values = []

# 設定 learning_rate 的範圍
learning_rate_range = np.arange(0.1, 2.0, 0.2)

# 進行迴圈
for learning_rate in learning_rate_range:
    # 創建 XGBoost 回歸模型
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=learning_rate, random_state=42)
    
    # 使用訓練集來訓練模型
    xgb_model.fit(X_train2, y_train2)
    
    # 預測驗證集的結果
    y_pred = xgb_model.predict(X_val)
    
    # 計算並儲存 MSE
    mse = mean_squared_error(y_val, y_pred)
    mse_values.append(mse)
    
    # 計算並儲存 MAE
    mae = mean_absolute_error(y_val, y_pred)
    mae_values.append(mae)

# # 使用 matplotlib 繪製 mse 和 mae
# plt.figure(figsize=(10, 6))
# plt.plot(learning_rate_range, mse_values, label='MSE')
# plt.plot(learning_rate_range, mae_values, label='MAE')
# plt.xlabel('Learning Rate')
# plt.legend()
# plt.show()

#####繼續調整learninng rate 在0.1到0.3之間#####

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 創建空列表來儲存每次迭代的 MSE 和 MAE
mse_values = []
mae_values = []

# 設定 learning_rate 的範圍
learning_rate_range = np.arange(0.1, 0.3, 0.02)

# 進行迴圈
for learning_rate in learning_rate_range:
    # 創建 XGBoost 回歸模型
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=learning_rate, random_state=42)
    
    # 使用訓練集來訓練模型
    xgb_model.fit(X_train2, y_train2)
    
    # 預測驗證集的結果
    y_pred = xgb_model.predict(X_val)
    
    # 計算並儲存 MSE
    mse = mean_squared_error(y_val, y_pred)
    mse_values.append(mse)
    
    # 計算並儲存 MAE
    mae = mean_absolute_error(y_val, y_pred)
    mae_values.append(mae)

# # 使用 matplotlib 繪製 mse 和 mae
# plt.figure(figsize=(10, 6))
# plt.plot(learning_rate_range, mse_values, label='MSE')
# plt.plot(learning_rate_range, mae_values, label='MAE')
# plt.xlabel('Learning Rate')
# plt.legend()
# plt.show()


# 所以設定 learning_rate為 0.14

######固定learning rate = 0.14，調整n_estimators ######
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

#創建空列表來儲存每次迭代的 MSE 和 MAE
mse_values = []
mae_values = []

#設定 n_estimators 的範圍
n_estimators_range = [50, 100, 150, 200, 250, 300]

#進行迴圈
for n_estimators in n_estimators_range:
  # 創建 XGBoost 回歸模型
  xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=0.14, random_state=42)

  # 使用訓練集來訓練模型
  xgb_model.fit(X_train2, y_train2)

  # 預測驗證集的結果
  y_pred = xgb_model.predict(X_val)

  # 計算並儲存 MSE
  mse = mean_squared_error(y_val, y_pred)
  mse_values.append(mse)

  # 計算並儲存 MAE
  mae = mean_absolute_error(y_val, y_pred)
  mae_values.append(mae)

# #使用 matplotlib 繪製 mse 和 mae
# plt.figure(figsize=(10, 6))
# plt.plot(n_estimators_range, mse_values, label='MSE')
# plt.plot(n_estimators_range, mae_values, label='MAE')
# plt.xlabel('n_estimators')
# plt.legend()
# plt.show()

##### 在75到125區間 更細的調整n_estimators #####

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

#創建空列表來儲存每次迭代的 MSE 和 MAE
mse_values = []
mae_values = []

#設定 n_estimators 的範圍
n_estimators_range = [75, 85, 95, 105, 115, 125]

#進行迴圈
for n_estimators in n_estimators_range:
  # 創建 XGBoost 回歸模型
  xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=0.14, random_state=42)

  # 使用訓練集來訓練模型
  xgb_model.fit(X_train2, y_train2)

  # 預測驗證集的結果
  y_pred = xgb_model.predict(X_val)

  # 計算並儲存 MSE
  mse = mean_squared_error(y_val, y_pred)
  mse_values.append(mse)

  # 計算並儲存 MAE
  mae = mean_absolute_error(y_val, y_pred)
  mae_values.append(mae)

# #使用 matplotlib 繪製 mse 和 mae
# plt.figure(figsize=(10, 6))
# plt.plot(n_estimators_range, mse_values, label='MSE')
# plt.plot(n_estimators_range, mae_values, label='MAE')
# plt.xlabel('n_estimators')
# plt.legend()
# plt.show()

'''
所以設定n_estimators為90
'''

######### 用找到的最佳超參數組合 訓練整個訓練集，並對測試集進行預測 #########


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 創建 XGBoost 回歸模型
xgb_model_Final = xgb.XGBRegressor(n_estimators= 90, learning_rate=0.14, random_state=42)

# 使用訓練集來訓練模型
xgb_model_Final.fit(X_train, y_train)
# 預測測試集
y_pred = xgb_model_Final.predict(X_test)

# 計算均方誤差 (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# 計算平均絕對誤差 (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)









############# 使用學習權重法組合 xgb_model_Final、random_forest_Final###############
from sklearn.linear_model import LinearRegression
import numpy as np

# 將兩個模型的預測結果作為特徵
X_ensemble = np.column_stack((rfFinal.predict(X_train), xgb_model_Final.predict(X_train)))

# 創建線性回歸模型
ensemble_model = LinearRegression()

# 使用訓練集來訓練ensemble模型
ensemble_model.fit(X_ensemble, y_train)

# 獲取模型的權重
weights = ensemble_model.coef_

# 輸出兩個模型的權重
print("Random Forest Weight:", weights[0])
print("XGBoost Weight:", weights[1])

from sklearn.metrics import mean_squared_error, mean_absolute_error

# 使用已訓練好的隨機森林模型和XGBoost模型對測試資料進行預測
rf_predictions = rfFinal.predict(X_test)
xgb_predictions = xgb_model_Final.predict(X_test)


# 使用學習得到的權重將兩個模型的預測結果結合起來
ensemble_predictions = (weights[0]/(weights[0]+weights[1])) * rf_predictions + (weights[1]/(weights[0]+weights[1])) * xgb_predictions

# 計算預測結果與實際目標值之間的MSE和MAE
# mse = mean_squared_error(y_test, ensemble_predictions)
# mae = mean_absolute_error(y_test, ensemble_predictions)

# print("MSE:", mse)
# print("MAE:", mae)



import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# ###資料處理
# file_train = r'ImdbRunAge0531_train.csv'
# file_test = r'ImdbRunAge0531_test.csv'
# 讀檔案，把不要用的欄位剔除
columns_to_ignore = ['Unnamed: 0', 'Unnamed: 0.1', 'title', 'description', 'imdb_score', 'tmdb_score']
# data = pd.read_csv(file_train)
X_trainCla = df_train.drop(columns=columns_to_ignore)

# ###機器學習
# 調整參數
randonforset_classifier = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=50, min_samples_leaf=1, max_depth=7)
X_trainCla = X_trainCla.drop(columns='imdb_level')
y_trainCla = df_train['imdb_level']
# 訓練
X_train, X_test, y_train, y_test = train_test_split(X_trainCla, y_trainCla, test_size=0.2, random_state=55)
randonforset_classifier.fit(X_train, y_train)

predict_X = randonforset_classifier.predict(X_train)






#顯示特徵用
df_show = pd.read_csv("ALL.csv")



import sys
from PyQt5.QtWidgets import QWidget,QApplication
from PyQt5 import uic,QtGui
import time

class AppDemo(QWidget):
    def __init__(self):
        # print('in_2')
        super().__init__()
        uic.loadUi('window.ui',self)
        #將分類的燈號歸零初始化
        self.Actual_CBar1.setValue(0)
        self.Actual_CBar2.setValue(0)
        self.Actual_CBar3.setValue(0)
        self.Actual_CBar4.setValue(0)
        
        self.Pre_CBar1.setValue(0)
        self.Pre_CBar2.setValue(0)
        self.Pre_CBar3.setValue(0)
        self.Pre_CBar4.setValue(0)
        
        
        self.testButton.clicked.connect(self.clicked)
    
    def clicked(self):
        self.Actual_CBar1.setValue(0)
        self.Actual_CBar2.setValue(0)
        self.Actual_CBar3.setValue(0)
        self.Actual_CBar4.setValue(0)
        
        self.Pre_CBar1.setValue(0)
        self.Pre_CBar2.setValue(0)
        self.Pre_CBar3.setValue(0)
        self.Pre_CBar4.setValue(0)
        # self.testLabel.setText("Bat Man")
             
        msg=self.lineEdit.text()
        # msg接收lineEdit
        # self.testLabel.setText(msg)
        
        file_name=msg+'.jpg'
        self.Photo.setPixmap(QtGui.QPixmap(file_name))
        self.Predict_Bar.setMaximum(100)
        self.Actual_Bar.setMaximum(100)
        
        
        
        
        ############################################顯示feature
        #找到那個row target_show存布林
        target_show=df_show['title'].str.find(msg)
        
        for i in range(0,len(target_show)):
            if target_show[i]!=-1:
                #把該列存成df
                df_show1=df_show.iloc[[i]]
                break
        
        #從df存出來是series
        show_year = df_show1['release_year']
        show_age = df_show1['age_certification']
        show_genres = df_show1['genres']
        show_country = df_show1['production_countries']
        
        #要把series print出來藥用iloc[第幾個]
        self.lineEdit_Year.setText(str(show_year.iloc[0]))
        self.lineEdit_Age.setText(str(show_age.iloc[0]))
        self.lineEdit_Genres.setText(str(show_genres.iloc[0]))
        self.lineEdit_Country.setText(str(show_country.iloc[0]))
        
        
        
        
        ############################################顯示回歸結果
        target=df_test['title'].str.find(msg)

        for i in range(0,len(target)):
            if target[i]!=-1:
                df_tar=df_test.iloc[[i]]
                break
        
        X_tar = df_tar.drop(['imdb_score', 'tmdb_score', 'title', 'description','Unnamed: 0.1','Unnamed: 0','imdb_level'], axis=1)
        y_tar = df_tar['imdb_score']
        
        #把feature丟入模型裡跑結果
        rf_tar = rfFinal.predict(X_tar)
        xgb_tar = xgb_model_Final.predict(X_tar)
        #集成學習
        ensemble_tar = (weights[0]/(weights[0]+weights[1])) * rf_tar + (weights[1]/(weights[0]+weights[1])) * xgb_tar
        
        
        #Bar只能呈現整數
        self.label_Ac.setText(str(round(y_tar.iloc[0],1)))
        self.Actual_Bar.setValue(round(y_tar.iloc[0]*10))
        #弄成動畫
        for i in range(0,round(ensemble_tar[0]*10)):
                time.sleep(0.1)
                self.Predict_Bar.setValue(i+1)
                self.label_Pre.setText(str((i+1)/10))
                
        
        ############################################顯示分類結果
        
        # msg="Jurassic World Dominion"
        target_RF=df_test['title'].str.find(msg)
        
        
        for i in range(0,len(target_RF)):
            if target_RF[i]!=-1:
                df_tarRF=df_test.iloc[[i]]
                break
            
        
        X_tarRF = df_tarRF.drop(columns_to_ignore, axis=1)
        X_tarRF=X_tarRF.drop(columns='imdb_level')
        y_tarRF = df_tarRF['imdb_level']
        
        tar_RF=randonforset_classifier.predict(X_tarRF)
        
        
        if y_tarRF.iloc[0]==3:
            cla=3
         
        if y_tarRF.iloc[0]==2:
            cla=2
        
        if y_tarRF.iloc[0]==1:
            cla=1
        
        if y_tarRF.iloc[0]==0:
            cla=0
        
        
        if tar_RF[0]==3:
            cla_pre=3
         
        if tar_RF[0]==2:
            cla_pre=2
        
        if tar_RF[0]==1:
            cla_pre=1
         
        if tar_RF[0]==0:
            cla_pre=0
            
        
        
        kcal=4-cla
        kkcla=4-cla_pre
        self.label_Ac_C.setText(str(kcal))
        self.label_Pre_C.setText(str(kkcla))
        
        if cla==0:
            self.Actual_CBar1.setValue(100)
            self.Actual_CBar2.setValue(100)
            self.Actual_CBar3.setValue(100)
            self.Actual_CBar4.setValue(100)
            
        if cla_pre==0:    
            self.Pre_CBar1.setValue(100)
            time.sleep(1)
            self.Pre_CBar2.setValue(100)
            time.sleep(1)
            self.Pre_CBar3.setValue(100)
            time.sleep(1)
            self.Pre_CBar4.setValue(100)
            
        
        if cla==1:
            self.Actual_CBar1.setValue(100)
            self.Actual_CBar2.setValue(100)
            self.Actual_CBar3.setValue(100)
         
            
        if cla_pre==1:     
            self.Pre_CBar1.setValue(100)
            time.sleep(1)
            self.Pre_CBar2.setValue(100)
            time.sleep(1)
            self.Pre_CBar3.setValue(100)
        
        if cla==2:
            self.Actual_CBar1.setValue(100)
            self.Actual_CBar2.setValue(100)
         
            
        if cla_pre==2:     
            self.Pre_CBar1.setValue(100)
            time.sleep(1)
            self.Pre_CBar2.setValue(100)
        
        
        if cla==3:
            self.Actual_CBar1.setValue(100)
         
            
        if cla_pre==3:     
            self.Pre_CBar1.setValue(100)

         

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = AppDemo()
    demo.show()
    # print('in')
    
    try :
        sys.exit(app.exec_())

    except:
        print('Closing Window')