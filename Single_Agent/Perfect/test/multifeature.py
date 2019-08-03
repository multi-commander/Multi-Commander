
##multi feature
##实例代码，分别转化的是均值，二阶矩，三阶矩，中位数
##现在原本的输入是29维数据，输出是7维数据
##一些变形的分析以及代码
##现在就是减小时间跨度实现小维度的抽样




##现在的state的信息
##lane_vehicle_count  每个车道的数量（56）
##start_lane_vehicle_count  开始车道上的数量（28）
##lane_waiting_vehicle_count  每条车道上等待车的数量（速度小于0.1m/s)
##lane_vehicles  车道上的车的id
##vehicle_speed  每个车的速度
##vehicle_distance  每个车辆已经行驶的距离
##current_time 现在的时间
##current_phase  现在的相位
##current_phase_time  现在的相位持续的时间


import numpy as np
from numpy import random
import cmath
import os



#1.这个代码的思路在于假设已经设定了一定时间的灯的颜色，然后假设ns,在n/2的时候进行数据采样分析，然后寻找变点的变化度量值
#如果超过，说明行车状态变化范围大
#如果低于，说明行车状态变化范围较小

def changepointrealm(q):
    '''
    input: 29*30 matrix
    return: (1,), distribution complexity
    '''
    n = len(q)
    p = len(q[0])
    XXX = [0]*n
    XX = []
    for i in range(n):
        D = np.ones((p,p))*0
        if i == 0:
            aa = q[0,:]
            bb = q[1:n,:]
        else:
            aa = q[0:i,:]
            bb = q[i+1:n,:]
        aa1 = np.cov(aa.T)
        bb1 = np.cov(bb.T)
        Sk = (aa1 * (i+1)+bb1*(n-i-1))/(n-2)
        summ = np.diag(Sk)
        for w in range(p):
            D[w,w]=summ[w]
        DD = np.linalg.inv(D)
        sum1= aa.sum(axis =0)
        sum2= q.sum(axis = 0)*(i+1)/n
        ss = sum1-sum2
        www= ss.reshape(ss.shape[0],1)
        www1=www.T
        DD1 =(www1/n/cmath.sqrt(p))@DD@www
        DD2 = i*(n-i)*cmath.sqrt(p)*(1+2/n)/pow(n,2)
        x = DD1-DD2
        XXX[i]=x[0]
    for h in range(3,n-3):
        XX.append((XXX[h]+XXX[h-1]+XXX[h+1])/3)
        XX = np.array(XX).reshape((-1))
    return max(XX)
#这里可以提取数据以后将各个向量输入之后做简单线性回归和feature筛选（AIC/BIC)
def complex(a,b,c):
     return 0.5*a+0.3*b+0.2*c 






#这个函数是用来创造相应phase的持续时间
# int a1 #节点1
# int a2 #节点2
def complextime(ss):
    currenttime = 0
    simulatetime = 0
    if ss <= a1:
        currenttime = 5
    elif ss > a1 and ss <= a2:
        currenttime = 10
    else:
        currenttime = 15

simulatetime = f(x)

return currenttime+simulatetime



phase.settime(complextime)


#q_num抽样间隔导致的抽样数量
#feature_num我要产生的新的feature的个数
#这是初始states
config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1 
config["state_size2"] = config["state_size"] + config["sizeplus"]
state.size = 29
state = env.get_state()
state = np.array(list(state['start_lane_vehicle_count'].values()) + [state['current_phase']] )
state = np.reshape(state, [1, state_size])
next_state = np.reshape(next_state, [1, state_size2])


feature_num = 4
config["sizeplus"] = feature_num

q = np.zeros([q_num,feature_num])
time = np.zeros(q_num)
for i in range(q_num):
    median_matrix = np.zeros(0)
    subsum1 = 0
    subsum2 = 0
    subsum3 = 0
    subsum4 = 0
    #here define how to get the feature
    for j in range(partition_num):
        subnumber = my_matrix[i*partition_num+j][1]
        subsum1 = subnumber + subsum1
        subsum2 = subnumber*subnumber + subsum2
        subsum3 = subnumber*subnumber*subnumber + subsum3
        median_matrix = np.append(median_matrix,subnumber)
        subsum4 = my_matrix[i*partition_num+j][0] + subsum4
    q[i][0] = subsum1/partition_num
    q[i][1] = subsum2/partition_num
    q[i][2] = subsum3/partition_num
    q[i][3] = np.median(median_matrix)
    if i ==0:
      time[i] = subsum4
    else:
      time[i] = subsum4+time[i-1]


##关于reward的一些讨论
#reward
def returnSum(myDict): 
    sum = 0
    for i in myDict: 
        sum = sum + myDict[i] 
    return sum


def get_reward(self):
    para1 = 0
    para2 = 0
    para3 = 0
    # a sample reward function which calculates the total of waiting vehicles
    lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
    lane_vehicle_speed = self.eng.get_vehicle_speed()
    reward1 = -1 * sum(list(lane_waiting_vehicle_count.values()))#100
    reward2 = -1* phasechangetimes#5
    reward3 = sum(list(lane_vehicle_speed.values()/returnSum(self.eng.get_lane_vehicle_count())))#10
    reward11 = reward1/(1+Math.exp(reward1))
    reward22 = reward2
    reward33 = 1/(1+Math.exp(reward3))


    return para1*reward11 + para2*reward22 + para3*reward33


