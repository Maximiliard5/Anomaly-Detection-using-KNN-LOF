import sklearn
import pyod
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data, generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import make_blobs
from pyod.models.lof import LOF
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.utils.utility import standardizer
from pyod.models import combination

def leverage_scores(x,add_intercept=True):
    x=np.asarray(x).reshape(-1)
    if add_intercept:
        X=np.column_stack([np.ones_like(x),x])
    else:
        X=x[:,None]

    # Economy SVD: X = U Σ V^T
    U,S,Vt=np.linalg.svd(X,full_matrices=False)
    h=np.sum(U**2,axis=1)
    return h

def leverage_scores_X(X,add_intercept=True):
    X=np.asarray(X)
    if X.ndim==1: X=X.reshape(-1,1)
    if add_intercept: X=np.column_stack([np.ones(X.shape[0]),X])
    U,S,Vt=np.linalg.svd(X,full_matrices=False)
    return np.sum(U**2,axis=1)

#Ex1
print("Starting exercise 1\n")
#low noise
miu_x=0
s_x=0.1
miu_eps=0
s_eps=0.01
a=1
b=0

nr_samples=100
contamination=0.1
nr_outliners=int(contamination*nr_samples)

x=np.random.normal(miu_x,s_x,nr_samples)
eps=np.random.normal(miu_eps,s_eps,x.shape[0])

y=a*x+b+eps

idx_top=np.argsort(leverage_scores(x))[-nr_outliners:]

fig,axs=plt.subplots(2, 2, figsize=(12, 10))

ax=axs[0, 0]
ax.scatter(x,y,color='red',label='Noisy data')
ax.scatter(x[idx_top], y[idx_top], color='green',label=f'Outliers')

x_line=np.linspace(min(x),max(x), 100)
y_line=a*x_line+b
ax.plot(x_line,y_line,color='blue',label='True line')

ax.set_title("Ex1: Low noise")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()

#High var on x
miu_x=0
s_x=1
miu_eps=0
s_eps=0.1

x=np.random.normal(miu_x,s_x,nr_samples)
eps=np.random.normal(miu_eps,s_eps,x.shape[0])

y=a*x+b+eps

idx_top=np.argsort(leverage_scores(x))[-nr_outliners:]

ax=axs[0, 1]
ax.scatter(x,y,color='red',label='Noisy data')
ax.scatter(x[idx_top], y[idx_top], color='green',label=f'Outliers')

x_line=np.linspace(min(x),max(x), 100)
y_line=a*x_line+b
ax.plot(x_line,y_line,color='blue',label='True line')

ax.set_title("Ex1: High var on x")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()

#High var on y
miu_x=0
s_x=0.1
miu_eps=0
s_eps=1

x=np.random.normal(miu_x,s_x,nr_samples)
eps=np.random.normal(miu_eps,s_eps,x.shape[0])

y=a*x+b+eps

idx_top=np.argsort(leverage_scores(x))[-nr_outliners:]

ax=axs[1, 0]
ax.scatter(x,y,color='red',label='Noisy data')
ax.scatter(x[idx_top], y[idx_top], color='green',label=f'Outliers')

x_line=np.linspace(min(x),max(x), 100)
y_line=a*x_line+b
ax.plot(x_line,y_line,color='blue',label='True line')

ax.set_title("Ex1: High var on y")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()

#High var on x,y
miu_x=0
s_x=1
miu_eps=0
s_eps=1

x=np.random.normal(miu_x,s_x,nr_samples)
eps=np.random.normal(miu_eps,s_eps,x.shape[0])

y=a*x+b+eps

idx_top=np.argsort(leverage_scores(x))[-nr_outliners:]

ax=axs[1, 1]
ax.scatter(x,y,color='red',label='Noisy data')
ax.scatter(x[idx_top], y[idx_top], color='green',label=f'Outliers')

x_line=np.linspace(min(x),max(x), 100)
y_line=a*x_line+b
ax.plot(x_line,y_line,color='blue',label='True line')

ax.set_title("Ex1: High var on x,y")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

#Now for the 3D exercise
#Ex1
#low noise
miu_x1=0
s_x1=0.1
miu_x2=0
s_x2=0.1
miu_eps=0
s_eps=0.01
a=1
b=1
c=0

nr_samples=100
contamination=0.1
nr_outliners=int(contamination*nr_samples)

fig,axs=plt.subplots(2, 2, figsize=(12, 10),subplot_kw={'projection':'3d'})

# Example 1: Low noise
x1=np.random.normal(miu_x1,s_x1,nr_samples)
x2=np.random.normal(miu_x2,s_x2,nr_samples)
eps=np.random.normal(miu_eps,s_eps,x1.shape[0])
y=a*x1+b*x2+c+eps
h=leverage_scores_X(np.column_stack([x1,x2]))
idx_top=np.argsort(h)[-nr_outliners:]

ax=axs[0,0]
ax.scatter(x1,x2,y,color='red',label='Noisy data')
ax.scatter(x1[idx_top],x2[idx_top],y[idx_top],color='green',label='Outliers')

x1g=np.linspace(x1.min(),x1.max(),30)
x2g=np.linspace(x2.min(),x2.max(),30)
X1,X2=np.meshgrid(x1g,x2g)
Yp=a*X1+b*X2+c
ax.plot_surface(X1,X2,Yp,alpha=0.25,color='blue')

ax.set_title("Ex1: Low noise")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
ax.legend()

# Example 2: High var on x1,x2
miu_x=0
s_x=0.1
s_x1=1
s_x2=1
s_eps=0.1
x1=np.random.normal(miu_x1,s_x1,nr_samples)
x2=np.random.normal(miu_x2,s_x2,nr_samples)
eps=np.random.normal(miu_eps,s_eps,x1.shape[0])
y=a*x1+b*x2+c+eps
h=leverage_scores_X(np.column_stack([x1,x2]))
idx_top=np.argsort(h)[-nr_outliners:]

ax=axs[0,1]
ax.scatter(x1,x2,y,color='red',label='Noisy data')
ax.scatter(x1[idx_top],x2[idx_top],y[idx_top],color='green',label='Outliers')

x1g=np.linspace(x1.min(),x1.max(),30)
x2g=np.linspace(x2.min(),x2.max(),30)
X1,X2=np.meshgrid(x1g,x2g)
Yp=a*X1+b*X2+c
ax.plot_surface(X1,X2,Yp,alpha=0.25,color='blue')

ax.set_title("Ex1: High var on x1,x2")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
ax.legend()

# Example 3: High var on y (noise)
s_x1=0.1
s_x2=0.1
s_eps=1
x1=np.random.normal(miu_x1,s_x1,nr_samples)
x2=np.random.normal(miu_x2,s_x2,nr_samples)
eps=np.random.normal(miu_eps,s_eps,x1.shape[0])
y=a*x1+b*x2+c+eps
h=leverage_scores_X(np.column_stack([x1,x2]))
idx_top=np.argsort(h)[-nr_outliners:]

ax=axs[1,0]
ax.scatter(x1,x2,y,color='red',label='Noisy data')
ax.scatter(x1[idx_top],x2[idx_top],y[idx_top],color='green',label='Outliers')

x1g=np.linspace(x1.min(),x1.max(),30)
x2g=np.linspace(x2.min(),x2.max(),30)
X1,X2=np.meshgrid(x1g,x2g)
Yp=a*X1+b*X2+c
ax.plot_surface(X1,X2,Yp,alpha=0.25,color='blue')

ax.set_title("Ex1: High var on y")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
ax.legend()

# Example 4: High var on x1,x2 and y
s_x1=1
s_x2=1
s_eps=1
x1=np.random.normal(miu_x1,s_x1,nr_samples)
x2=np.random.normal(miu_x2,s_x2,nr_samples)
eps=np.random.normal(miu_eps,s_eps,x1.shape[0])
y=a*x1+b*x2+c+eps
h=leverage_scores_X(np.column_stack([x1,x2]))
idx_top=np.argsort(h)[-nr_outliners:]

ax=axs[1,1]
ax.scatter(x1,x2,y,color='red',label='Noisy data')
ax.scatter(x1[idx_top],x2[idx_top],y[idx_top],color='green',label='Outliers')

x1g=np.linspace(x1.min(),x1.max(),30)
x2g=np.linspace(x2.min(),x2.max(),30)
X1,X2=np.meshgrid(x1g,x2g)
Yp=a*X1+b*X2+c
ax.plot_surface(X1,X2,Yp,alpha=0.25,color='blue')

ax.set_title("Ex1: High var on x1,x2,y")
ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
ax.legend()

plt.tight_layout()
plt.show()

#Ex2
print('\n\nStarting exercise 2\n')
X_train, X_test, y_train, y_test = generate_data_clusters(n_train=400, n_test=200, n_features=2,contamination=0.1)

def plot_xy(ax,X,y,title):
    ax.scatter(X[y==0,0],X[y==0,1],c='blue',label='inliers',alpha=0.8,s=20)
    ax.scatter(X[y==1,0],X[y==1,1],c='red',label='outliers',alpha=0.8,s=20)
    ax.set_title(title); ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(True); ax.legend(loc='best')

ks=[3,5,10,20]
for k in ks:
    model= KNN(n_neighbors=k, contamination=contamination)
    model.fit(X_train)

    y_pred_train=model.labels_
    y_pred_test=model.predict(X_test)

    bal_tr=balanced_accuracy_score(y_train, y_pred_train)
    bal_te=balanced_accuracy_score(y_test, y_pred_test)
    print(f"k={k}: balanced_acc train={bal_tr:.3f}, test={bal_te:.3f}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plot_xy(axs[0, 0],X_train,y_train, "Train GT")
    plot_xy(axs[0, 1],X_train, y_pred_train, f"Train Pred (k={k})")
    plot_xy(axs[1, 0],X_test, y_test, "Test GT")
    plot_xy(axs[1, 1],X_test,y_pred_test, f"Test Pred (k={k})")
    plt.tight_layout()
    plt.show()

#Ex3
print('\n\nStarting exercise 3\n')
X, y_blobs = make_blobs(n_samples=[200, 100], centers=[(-10, -10), (10, 10)], cluster_std=[2, 6], n_features=2, shuffle=True)
contamination=0.07
ks=[3,5,10,20]
for k in ks:
    model=KNN(n_neighbors=k, contamination=contamination)
    model.fit(X)
    y_pred_knn=model.labels_

    #use lof
    lof=LOF(n_neighbors=k, contamination=contamination)
    lof.fit(X)
    y_pred_lof=lof.labels_

    #Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    #KNN plot
    axs[0].scatter(X[y_pred_knn == 0, 0], X[y_pred_knn == 0, 1], c='blue', label='inliers', s=20)
    axs[0].scatter(X[y_pred_knn == 1, 0], X[y_pred_knn == 1, 1], c='red', label='outliers', s=20)
    axs[0].set_title(f"KNN (k={k})")
    axs[0].set_xlabel("x1");
    axs[0].set_ylabel("x2")
    axs[0].grid(True);
    axs[0].legend(loc='best')

    #LOF plot
    axs[1].scatter(X[y_pred_lof == 0, 0], X[y_pred_lof == 0, 1], c='blue', label='inliers', s=20)
    axs[1].scatter(X[y_pred_lof == 1, 0], X[y_pred_lof == 1, 1], c='red', label='outliers', s=20)
    axs[1].set_title(f"LOF (k={k})")
    axs[1].set_xlabel("x1");
    axs[1].set_ylabel("x2")
    axs[1].grid(True);
    axs[1].legend(loc='best')

    plt.suptitle(f"Ex3: Comparison of KNN and LOF (n_neighbors={k})")
    plt.tight_layout()
    plt.show()

#Ex4
print('\n\nStarting exercise 4\n')
data=loadmat('cardio.mat')
X=data.get('X', data.get('x'))
y=data.get('y').astype(int).ravel()
print('X shape: ',X.shape)
print('y.shape',y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

contamination=y_train.mean()

ks=np.unique(np.linspace(30,120,10,dtype=int))
#I will make one function for both methods because I don't want to copy paste the code (the file is already big enough)
def run(model_name):
    print("\n"+"="*70)
    print(f"Ensemble of {model_name} models, k in {list(ks)}  |  contamination={contamination:.3f}")
    print("-"*70)

    tr_scores_all=[]
    te_scores_all=[]

    for k in ks:
        if model_name=='KNN':
            model=KNN(n_neighbors=k, contamination=contamination)
        else:
            model=LOF(n_neighbors=k, contamination=contamination)

        model.fit(X_train)

        #labels,decision scores for the training set
        y_pred_tr=model.labels_
        tr_raw=model.decision_scores_.reshape(-1,1)

        #tet predictions,raw decision scores
        y_pred_te=model.predict(X_test)
        te_raw=model.decision_function(X_test).reshape(-1,1)

        #balance accuracy
        bal_tr=balanced_accuracy_score(y_train,y_pred_tr)
        bal_te=balanced_accuracy_score(y_test,y_pred_te)
        print(f"{model_name}(k={k:>3}): BA train={bal_tr:.3f} | BA test={bal_te:.3f}")

        #standardize train,test scores
        tr_std,te_std=standardizer(tr_raw, te_raw)
        tr_scores_all.append(tr_std)
        te_scores_all.append(te_std)

    #stack to shape (n_samples,n_models)
    TR=np.hstack(tr_scores_all)
    TE=np.hstack(te_scores_all)

    for comb_name,comb_fn in [("average",combination.average),("maximization",combination.maximization)]:
        #combined scores
        tr_final=comb_fn(TR)
        te_final=comb_fn(TE)

        #thresholds from known contamination
        thr_tr=np.quantile(tr_final,1-contamination)
        thr_te=np.quantile(te_final,1-contamination)

        yhat_tr=(tr_final>thr_tr).astype(int)
        yhat_te=(te_final>thr_te).astype(int)

        ba_tr=balanced_accuracy_score(y_train,yhat_tr)
        ba_te=balanced_accuracy_score(y_test,yhat_te)
        print(f"[{model_name} | {comb_name:11s}]  BA train={ba_tr:.3f} | BA test={ba_te:.3f}")

run('KNN')
run('LOF')