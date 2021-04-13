import pandas as pd
from sklearn.utils import shuffle
#Author:ayu_DY

def data2csv():
    path_or_buf = 'E:/testshengduxuexi/moshi/letter/letter-recognition.data'
    df=pd.read_csv(path_or_buf, header=None,na_values='?')
    # print(df)
    df.to_csv("E:/testshengduxuexi/moshi/letter/2_data.csv",header=None)

def csv_random():
    data = pd.read_csv("E:/testshengduxuexi/moshi/letter/2_data.csv")
    data = shuffle(data)
    data.to_csv('E:/testshengduxuexi/moshi/letter/2_data2.csv',header=None)
    print(data)

def csv_div():
    data_tra = pd.read_csv('E:/testshengduxuexi/moshi/letter/2_data2.csv', nrows=15999)
    data_tra.to_csv('E:/testshengduxuexi/moshi/letter/data_tra.csv',index=None,header=None)
    data_tes = pd.read_csv('E:/testshengduxuexi/moshi/letter/2_data2.csv', skiprows=16000, nrows=19999)
    data_tes.to_csv('E:/testshengduxuexi/moshi/letter/data_tes.csv',index=None,header=None)

def letter_div():
    tra_path = 'E:/testshengduxuexi/moshi/letter/data_tra.csv'  # 读取数据
    df_all = pd.read_csv(tra_path, usecols=[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    frame_all = pd.DataFrame(df_all)
    for letter in range(65, 91):
        frame_2 = frame_all[(frame_all.values == chr(letter))]  # 查找标签
        frame_2.to_csv('E:/testshengduxuexi/moshi/letter/letter_new/data_' + chr(letter) + '.csv', index=None,header=None)
        # print(frame_2.shape)
        # print(frame_2)

def main():
    # data2csv()
    # csv_random()
    # csv_div()
    letter_div()

if __name__ == '__main__':
    main()

