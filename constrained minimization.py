from scipy.optimize import minimize
import numpy as np
import pandas as pd

file = '/home/ronald/data/cripps20112014.csv'
data = pd.read_csv(file)

#data['MA'] = pd.rolling_mean(data['value'])

#print len(data)

data['lvW'] = np.nan
data['w2W'] = np.nan
data['y1W'] = np.nan
data['y2W'] = np.nan
data['mal'] = np.nan
data['maW'] = np.nan
data['PV'] = np.nan
data['err'] = np.nan

p_start, p_end = 105, 208

target = p_start

while target <= p_end:

    lv = data['value'][target-1] #previous value (1 week prior)
    w2 = data['value'][target-2] #value 2 weeks prior
    y1 = data['value'][target-52] #value 1 year prior
    y2 = data['value'][target-104] #value 2 years prior

    #lvw = 0.018002537
    #w2w = 0.318200251
    #y1w = 0.029406001
    #y2w = 0.223831008
    #maw = 0.408821425
    #mal = 52

    #MA = pd.rolling_mean(data['value'], mal)
    #print MA

    AV = data['value'][p_start-1:p_end-1].values

    def pred(x):
        rm = pd.rolling_mean(data['value'], 52)
        return abs((lv*x[0]+w2*x[1]+y1*x[2]+y2*x[3]+ rm[target] * x[4] - data['value'][target]))/data['value'][target]

    bnds = ((0,1), (0,1), (0,1), (0,1), (0,1))
    cf = [0.018002537, 0.318200251, 0.029406001, 0.223831008, 0.408821425]

    #Says one minus the sum of all variables must be zero
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})

    result = minimize(pred, cf, method ="SLSQP", bounds =bnds, constraints=cons)

    data['lvW'][target] = result.x[0]
    data['w2W'][target] = result.x[1]
    data['y1W'][target] = result.x[2]
    data['y2W'][target] = result.x[3]
    data['mal'][target] = 52
    data['maW'][target] = result.x[4]

    print 'SUM', result.x[0] + result.x[1] + result.x[2] + result.x[3] + result.x[4]
    print result.x[0], result.x[1], result.x[2], result.x[3], result.x[4]

    rmA = pd.rolling_mean(data['value'], data['mal'][target])

    data['PV'][target] = lv*data['lvW'][target] + w2*data['w2W'][target] + y1*data['y1W'][target] + y2*data['y2W'][target] + rmA[target]*data['maW'][target]

    data['err'][target] = abs(data['PV'][target] - data['value'][target]) / data['value'][target]

    target += 1

print data[['value', 'PV', 'err']][p_start:p_end]

#write pandas dataframe to csv file in project directory
#data.to_csv('cripps_fitted_prediction_values')

print 'lvW', np.mean(data['lvW'][p_start:p_end])
print 'w2W', np.mean(data['w2W'][p_start:p_end])
print 'y1W', np.mean(data['y1W'][p_start:p_end])
print 'y2W', np.mean(data['y2W'][p_start:p_end])
print 'mal', np.mean(data['mal'][p_start:p_end])
print 'maW', np.mean(data['maW'][p_start:p_end])

#res = minimize(calc, [-1.0,1.0], method='SLSQP', options={'disp': True})

#print(res.x)

