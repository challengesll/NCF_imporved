

import pandas as pd
import numpy as np
import scipy.sparse as sp

#构建所有数据集
def build_data():

    filepath1 = "Data/ml-1m.train.rating"
    filepath2 = "Data/ml-1m.test.rating"
    train = pd.read_csv(filepath1,names=['user_id', 'item_id', 'rating', 'category'],sep = "\t",engine="python")

    test = pd.read_csv(filepath2,names=['user_id', 'item_id', 'rating', 'category'],sep = "\t",engine="python")
    train = pd.merge(train,test,on=['user_id', 'item_id', 'rating', 'category'],how="outer")
    #print(train1.user_id.shape[0])
    data = train.sort_values(by="category")
    #划分数据集，将每个用户的最后一个时刻的数据当成测试集
    #train_data = []
    test_data = []
    user = list(data.user_id.unique())

    for i in user:
        #get 某个用户的交互dataframe，得到时间最近的数据一条
        user_i = data[data.user_id.isin([i])].sort_values(by="category")[0:1]
        test_data.append(user_i.values.tolist()[0])
    test_dataframe = pd.DataFrame(test_data,columns=['user_id', 'item_id', 'rating', 'category'])
    #数据集和测试集的差集
    df1 = data.append(test_dataframe)
    df1 = df1.append(test_dataframe)
    train_dataframe = df1.drop_duplicates(subset=['user_id', 'item_id', 'rating', 'category'], keep=False)
    return train_dataframe, test_dataframe,train
def interac_list(train,id,_id,u_or_i,num):
    #从训练数据中得到某个用户的交互序列
    list = []
    #list.append(u_or_i)
    user_dataframe = train[train[id].isin([u_or_i])].sort_values(by="category")#.iloc[0:num]
    #print(user_dataframe)
    if user_dataframe[id].shape[0] >= num :
        intec_frame = user_dataframe.iloc[0:num]
        #print(intec_frame)
        intec_list = intec_frame[_id]
    else:
        #print(u_or_i,id)
        intec_list = []
        l = user_dataframe[id].shape[0]
        #print(l)
        intec_frame = user_dataframe.iloc[0:l]
        #print(intec_frame)
        intec = intec_frame[_id].tolist()
        add_list = np.full((num-l),u_or_i, dtype = "int32").tolist()
        intec_list.extend(add_list)
        intec_list.extend(intec)
        #print(intec_list)
    list.extend(intec_list)
    #print(list)
    return list

if __name__ == '__main__':
    #train_df, test_df, train = build_data()
    """#保存数据
    train_data = train_df.sort_values(by="category",ascending=False)
    test_data = test_df.sort_values(by="user_id")
    np.savetxt("Data/train_data.rating", train_data.values, fmt="%d", delimiter="\t")
    np.savetxt("Data/test_data.rating", test_data.values, fmt="%d", delimiter="\t")
    np.savetxt("Data/data.rating", train.sort_values(by="user_id").values, fmt="%d", delimiter="\t")

    """
    """#获取user和item的交互序列
    users_interact_list = []
    items_interact_list = []
    users = list(train_df.sort_values(by="user_id").user_id.unique())
    items = list(train_df.sort_values(by="item_id").item_id.unique())
    #print(items)
    for u in users:
        u_i_list = interac_list(train_df,"user_id", "item_id", u ,16)
        users_interact_list.append(u_i_list)
    #for j in items:
        #i_j_list = interac_list(train_df, "item_id", "user_id", j, 4)
        #print(j,i_j_list)
        #items_interact_list.append(i_j_list)
    users_interact_array = np.array(users_interact_list, dtype=int)
    #items_interact_array = np.array(items_interact_list, dtype=int)
    np.savetxt("Data/user_item_list_16.txt", users_interact_array, fmt="%d", delimiter=" ")
    #np.savetxt("Data/item_user_list.txt", items_interact_array, fmt="%d", delimiter=" ")
    """
    """
    #生成negative的代码
    #test_data = test_df.sort_values(by="user_id")
    #得到所有的评分数据矩阵，num_user,num_item
    num_users, num_items = 0,0
    with open("Data/data.rating", "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open("Data/data.rating", "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if (rating > 0):
                mat[user, item] = 1.0
            line = f.readline()
    #加载测试数据，其次将每个数组作为第一行，之后添加每个用户的negative
    test_list = []
    with open("Data/test_data.rating", "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            test_list.append([user, item])
            line = f.readline()
    neg_list = []
    #filepath = "Data/test_negative.txt"
    #file = open(filepath, "w")
    for (u,i) in test_list:
        u_list = []
        u_list.append((u,i))
        for j in range(99):
            j = np.random.randint(num_items)
            while (u,j) in mat.keys():
                j = np.random.randint(num_items)
            u_list.append(j)
        neg_list.append(u_list)
    list = np.array(neg_list)
    np.savetxt("Data/test_negative.txt", list, fmt="%s", delimiter="\t")
        #file.write(u_list)
    #file.close()
    """

    print("have finished")