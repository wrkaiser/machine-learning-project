from  sklearn.ensemble  import  GradientBoostingRegressor
import  numpy as np

gbdt = GradientBoostingRegressor(
    loss='ls',
    learning_rate=0.1,
    n_estimators=100,
    subsample=1,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    init=None,
    random_state=None,
    max_features=None,
    alpha=0.9,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False
)
train_feat = np.genfromtxt("./data_set/GBDT/train_feat.txt", dtype=np.float32)
train_id = np.genfromtxt("./data_set/GBDT/train_id.txt", dtype=np.float32)
test_feat = np.genfromtxt("./data_set/GBDT/test_feat.txt", dtype=np.float32)
test_id = np.genfromtxt("./data_set/GBDT/test_id.txt", dtype=np.float32)
print(train_feat.shape,train_id.shape,test_feat.shape,test_id.shape)
gbdt.fit(train_feat,train_id)
pred = gbdt.predict(test_feat)
total_err=0
for i in range(pred.shape[0]):
    print(str(pred[i])+"------"+str(test_id[i]))
    err=(pred[i]-test_id[i])/test_id[i]
    total_err+=err*err
print(total_err/pred.shape[0])