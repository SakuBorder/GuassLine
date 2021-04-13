from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
import math

#选取数据集
print("0    letter\n1  Opt-digits\n2  Statlog-Satimage\n3    vowel")
data_n = int(input("请输入数据集序号: "))

#打开数据集
def open_data(cls,features,path):
    with open(path, 'r') as train_data:
        data_train = train_data.readlines()
    while data_n == 0:
        for line in data_train:
            # print(line)
            # print('----------------------------------------------------------------------------------')
            temp = line.split(',')  # 按列拆分
            temp[-1] = list(temp[-1])[0]
            cls.append(temp[0])  # 取类别
            features.append((temp[1::]))  # 取每一位独的特征值
        break

    while data_n == 1:
        '''Qdf有奇异值'''
        for line in data_train:
            # print(line)
            # print('----------------------------------------------------------------------------------')
            temp = line.split(',')  # 按列拆分
            temp[-1] = list(temp[-1])[0]
            cls.append(temp[-1])
            features.append((temp[0:len(temp) - 1:]))
        break
        # print('1')

    while data_n == 2:
        for line in data_train:
            # print(line)
            # print('----------------------------------------------------------------------------------')
            temp = line.split(' ')  # 按列拆分
            temp[-1] = list(temp[-1])[0]
            cls.append(temp[-1])
            features.append((temp[0:len(temp) - 1:]))
        break
        # print(cls,features)
    while data_n ==3:
        for line in data_train:
            # print(line)
            '''这个就很迷惑，准确率很低'''
            # print('----------------------------------------------------------------------------------')
            temp = line.split()
            temp[-1] = list(temp[-1])[0]
            cls.append(temp[-1])
            # print(cls)
            features.append((temp[3:len(temp) - 1:]))
            # print(features)
        break


#计算协方差矩阵
def matrix_cov(data_dict):
    for clas in data_dict:
        return np.cov(data_dict[clas])

# def calculate_covariance_matrix_I(features):
#     Y = features
#     data_cov = np.cov(Y, rowvar=False)
#     print(data_cov.shape)
#     try:
#         data_cov_I = np.linalg.inv(data_cov)
#     except:
#         data_cov_I = np.linalg.pinv(data_cov)
#
#     else:
#         return np.array(data_cov_I, dtype=float)

#判空转置
def matrix_T(features, Y=np.empty((0,0))):
    if not Y.any():
        Y = features
    data_tran = Y.T
    return np.array(data_tran, dtype=float)

#求期望
def data_Expection(features):
    # data_mean = np.mean(features)
    # print(data_mean.shape)
    expection = [0] * len(features)
    for row in range(0, len(features)):
        for col in range(0, len(features[0])):
            expection[row] += features[row][col]
    for i in range(0, len(expection)):
        expection[i] = expection[i] / len(features[0])
    return expection

#字符串转浮点型
def convert_float(str_list):
    for row in range(0, len(str_list)):
        for col in range(0, len(str_list[row])):
            str_list[row][col] = float(str_list[row][col])
    # return str_list

#字典操作，保存为以类别为键的字典，同时将特征值的列表作为value
def dict_le(feature,cls):
    data_dict = {}
    for i in range(len(cls)):
        if cls[i] not in data_dict:
            data_dict[cls[i]] = []
        data_dict[cls[i]].append(feature[i])
    for clas in data_dict:
        data_dict[clas] = np.array(data_dict[clas]).T
    return data_dict

#算先验概率wi，存字典
def pre(cls):
    pre = {}
    for chr in cls:
        if chr not in pre:
            pre[chr] = 0
        pre[chr] += 1
    for chr in pre:
        pre[chr] = pre[chr] / len(cls)
    return pre

#将参数加入列表，以方便读取
def data_param(data_dict, cls):
    data_param = [[], [], []]
    for clas in data_dict:
        data_param[0].append(clas)
        data_param[1].append(np.cov(data_dict[clas]))
        data_param[2].append(np.array(data_Expection(data_dict[clas])))
    data_param.append(pre(cls))
    return data_param

#ldf判别式，得出每个测试数据字母的最大可能性
def ldf(test_features, data_param):
    max=0
    max_cls = ''
    for i in range(0, len(data_param[0])):
        cls = data_param[0][i]
        cov = data_param[1][i]
        mean_le = data_param[2][i]
        # 判别是否可逆
        try:
            cov_I = np.linalg.inv(cov)
        except:
            cov_I = np.linalg.pinv(cov)
            # print("矩阵不可逆")
        cov_u=test_features-mean_le
        cov_T=cov_u.T
        prowi = np.log(data_param[3][cls]) * 2
        # print(prowi)
        equa1=np.matmul(cov_u,cov_I)
        equa2=np.matmul(equa1,cov_T)
        equa=-equa2+prowi
        if equa > max or max_cls == '':
            max = equa
            max_cls = cls
    return max_cls

#qdf判别式，得出每个测试数据字母的最大可能性
def qdf(test_features, data_param):
    max=0
    max_cls=''
    for i in range(0, len(data_param[0])):
        cls = data_param[0][i]
        cov = data_param[1][i]
        cov_det = np.linalg.det(cov)
        mean_le = data_param[2][i]
        #判别是否可逆
        try:
            cov_I = np.linalg.inv(cov)
        except:
            cov_I = np.linalg.pinv(cov)
            # print("矩阵可逆")
        cov_u=test_features-mean_le
        cov_T=cov_u.T
        if data_n==1:
            #数据集1有奇异值，加一个小数据
            prowi = - math.log(abs(cov_det)+10**(-6))
        else:
            prowi = - math.log(abs(cov_det))
        equa1=np.matmul(cov_u,cov_I)
        equa2=np.matmul(equa1,cov_T)
        equa=-equa2+prowi
        if equa > max or max_cls == '':# a Or b,a为真返回a,否则返回b
            max = equa
            max_cls = cls
    return max_cls

#使用ldf方法与对应训练集计算判别函数参数，并对测试集进行预测
def predict_ldf(features, cls, test_features):
    cls_pred = []
    cor_features = dict_le(features, cls)
    param = data_param(cor_features, cls)
    for i in range(0, len(test_features)):
        cls_pred.append(ldf(test_features[i], param))
    return cls_pred

#使用qdf方法与对应训练集计算判别函数参数，并对测试集进行预测
def predict_qdf(features, cls, test_features):
    cls_pred = []
    cor_features = dict_le(features, cls)
    param = data_param(cor_features, cls)
    for i in range(0, len(test_features)):
        cls_pred.append(qdf(test_features[i], param))
    return cls_pred

#读取测试集真实类别，与预测类别对比，计算准确率
def accuracy(cls_pred,cls):
    sum_r=0
    for i in range(len(cls)):
        if cls_pred[i]==cls[i]:
            sum_r +=1
    return float(sum_r)/len(cls)

#根据预测类别与真实类别绘制混淆矩阵
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#第一个数据集的混淆矩阵
def letter_cm(y_true,y_pred,method):
    cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    labels_name='A','B','C','D','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
    plot_confusion_matrix(cm, labels_name, "Confusion Matrix")
    plt.savefig('E:/testshengduxuexi/moshi/letter/letter_'+str(method)+'.png', format='png')
    # plt.show()

#第二个数据集的混淆矩阵
def opt_cm(y_true,y_pred,method):
    cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    labels_name='0','1','2','3','4','5','6','7','8','9'
    plot_confusion_matrix(cm, labels_name, "Confusion Matrix")
    for i in range(1, 3):
        plt.savefig('E:/testshengduxuexi/moshi/data/Opt-digits/opt_'+str(method)+'.png', format='png')
    # plt.show()

#第三个数据集的混淆矩阵
def sta_cm(y_true,y_pred,method):
    cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    labels_name='1','2','3','4','5','7'
    plot_confusion_matrix(cm, labels_name, "Confusion Matrix")
    for i in range(1, 3):
        plt.savefig('E:/testshengduxuexi/moshi/data/Statlog-Satimage/sta_'+str(method)+'.png', format='png')
    # plt.show()

#第四个数据集的混淆矩阵
def vow_cm(y_true,y_pred,method):
    cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    labels_name='0','1','2','3','4','5','6','7','8','9','10'
    plot_confusion_matrix(cm, labels_name, "Confusion Matrix")
    for i in range(1, 3):
        plt.savefig('E:/testshengduxuexi/moshi/data/vowel/vow_'+str(method)+'.png', format='png')
    # plt.show()

#选取数据集
def result(data_n):
    train_path0 = 'E:/testshengduxuexi/moshi/letter/data_tra.csv'
    train_path1 = 'E:/testshengduxuexi/moshi/data/Opt-digits/optdigits.tra'
    train_path2 = 'E:/testshengduxuexi/moshi/data/Statlog-Satimage/sat.trn'
    train_path3 = 'E:/testshengduxuexi/moshi/data/vowel/vowel.tra'
    test_path0 = 'E:/testshengduxuexi/moshi/letter/data_tes.csv'
    test_path1 = 'E:/testshengduxuexi/moshi/data/Opt-digits/optdigits.tes'
    test_path2 = 'E:/testshengduxuexi/moshi/data/Statlog-Satimage/sat.tst'
    test_path3 = 'E:/testshengduxuexi/moshi/data/vowel/vowel.tes'
    if data_n==0:
        train_path=train_path0
        test_path = test_path0
    elif data_n==1:
        train_path=train_path1
        test_path = test_path1
    elif data_n==2:
        train_path=train_path2
        test_path = test_path2
    elif data_n==3:
        train_path=train_path3
        test_path = test_path3
    else:
        print("数据集输入错误，重新输入")
    return train_path,test_path



def main():
    #初始化参数
    train_class = []
    train_features = []
    test_class = []
    test_features = []
    train_path,test_path=result(data_n)
    #分割训练集
    open_data(train_class, train_features, train_path)
    #字符串转浮点型
    convert_float(train_features)
    #分割测试集
    open_data(test_class, test_features, test_path)
    # 字符串转浮点型
    convert_float(test_features)
    # 数据集训练、测试、准确率
    LDF_predict = predict_ldf(train_features, train_class, test_features)
    LDF_accuracy = accuracy(LDF_predict, test_class)
    print('使用LDF测试的准确率为：', LDF_accuracy)
    QDF_predict = predict_qdf(train_features, train_class, test_features)
    QDF_accuracy = accuracy(QDF_predict, test_class)
    print('使用QDF测试的准确率为：', QDF_accuracy)
    #绘制混淆矩阵
    # letter_cm(test_class, LDF_predict,'LDF')
    # letter_cm(test_class, QDF_predict,'QDF')
    # opt_cm(test_class, LDF_predict,'LDF')
    # opt_cm(test_class, QDF_predict,'QDF')
    # sta_cm(test_class, LDF_predict,'LDF')
    # sta_cm(test_class, QDF_predict,'QDF')
    # vow_cm(test_class, LDF_predict,'LDF')
    # vow_cm(test_class, QDF_predict,'QDF')
if __name__ == "__main__":
    main()