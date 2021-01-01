from functions import *
import time

data = loadmat('..//data//OD_3m.mat')
data = data['OD']
data = remove_weekends(data, start=5)

train_idx = start_end_idx('2017-07-03', '2017-07-28', weekend=False, night=False)
validate_idx = start_end_idx('2017-07-31', '2017-08-11', weekend=False, night=False)
test_idx = start_end_idx('2017-08-14', '2017-08-25', weekend=False, night=False)
train_data = data[:, train_idx]
test_data = data[:, test_idx]
validate_data = data[:, validate_idx]

flow0 = od2flow(data)
flow = np.zeros((flow0.shape[0]*2, flow0.shape[1]), dtype=flow0.dtype)
flow[0:flow0.shape[0], :] = flow0
flow[flow0.shape[0]:, 1:] = flow0[:, 0:-1]

m_train = train_idx.shape[0]
m_validate = validate_idx.shape[0]

h = [36, 33, 28, 19,8, 3, 30, 35, 4, 14]
bs = 36
h.sort()
h = np.array(h)

#%% Define variables for multi-step forecast
ns = 4  # The number of steps to forecast
# To store the forecast for each step
results = {i:np.zeros((data.shape[0], validate_idx.shape[0]+test_idx.shape[0])) for i in range(1, ns+1)}
buffer_OD = np.zeros((data.shape[0], max(2, ns-1)))  # To store previously/current forecast OD
buffer_flow = np.zeros((159, 2+ns-1))  # To store real/forecast flow
model1 = TimeVaryingDMD_standard_eX(h, 100, 50, 0.92, bs=bs)
model1.fit(train_data, flow[:, max(h) - 1:m_train - 1])

# Initialize buffer OD
now = validate_idx[0] - 2
for i in range(2):
    X = data[:, now-h].reshape((-1, 1), order='F')
    X = np.concatenate([X, flow[:, [now-1]]], axis=0)
    buffer_OD[:, [i]] = model1._forecast(X)
    now += 1

#%% Start multi-step forecast
now = validate_idx[0]

# Perform the forecast over the validation and the test set
for tt in range(validate_idx.shape[0]+test_idx.shape[0]):
    buffer_flow[:, 0:2] = flow0[:, now-2:now]
    small_data = data[:, now-max(h):now-2]
    # The `nn`-th step forecast at `tt`
    for nn in range(ns):
        new_small_data = np.concatenate((small_data[:, nn:], buffer_OD[:, 0:nn]), axis=1)
        X = new_small_data[:, -(h-2)].reshape((-1, 1), order='F')
        X = np.concatenate((X, buffer_flow[:, [nn+1, nn]].reshape((-1, 1), order='F')), axis=0)
        od = model1._forecast(X)
        results[nn+1][:, tt] = od.ravel()

        if (ns - 3) > nn:
            buffer_OD[:, nn+2] = od.ravel()
        if (ns - 1) > nn:
            buffer_flow[:, nn+2] = od2flow(od).ravel()

    buffer_OD[:, 0] = buffer_OD[:, 1]
    buffer_OD[:, 1] = results[1][:, tt]

    # If now is a multiple of one-day, perform online update
    if (now+1)%36 == 0:
        X, Y = stagger_data(data[:, now - 35 - max(h): now+1], h)
        X = np.concatenate([X, flow[:, now - 36:now]])
        model1.update_model(X, Y)
    now += 1

print(RMSE(data[:, test_idx], results[1][:, 360:720]))
print(RMSE(data[:, test_idx], results[2][:, 360-1:-1]))
print(RMSE(data[:, test_idx], results[3][:, 360-2:-2]))
print(RMSE(data[:, test_idx], results[4][:, 360-3:-3]))

for key, value in results.items():
    np.savez_compressed('..//data//OD_HWDMD_step{}.npz'.format(key), data=value[:, 360-(key-1):720-(key-1)])
