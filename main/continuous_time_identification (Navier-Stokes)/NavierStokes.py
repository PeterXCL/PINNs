"""
Physics-Informed Neural Network for 2-D Navier–Stokes (Raissi et al.)
Updated for TensorFlow 2.20 (TF1-compatibility mode)
"""

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

# -------------------------------------------------------------------
# TensorFlow 2.x → TF1-compat mode
# -------------------------------------------------------------------
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.random.set_random_seed(1234)

# -------------------------------------------------------------------
# Add Utilities path
# -------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
utilities_path = os.path.join(project_root, "Utilities")
if utilities_path not in sys.path:
    sys.path.insert(0, utilities_path)

from plotting import newfig, savefig

# -------------------------------------------------------------------
# Physics-Informed Neural Network
# -------------------------------------------------------------------
class PhysicsInformedNN:
    def __init__(self, x, y, t, u, v, layers):
        X = np.concatenate([x, y, t], 1)
        self.lb, self.ub = X.min(0), X.max(0)
        self.x, self.y, self.t, self.u, self.v = X[:,0:1], X[:,1:2], X[:,2:3], u, v
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)

        # placeholders
        self.x_tf = tf.placeholder(tf.float32, [None, 1])
        self.y_tf = tf.placeholder(tf.float32, [None, 1])
        self.t_tf = tf.placeholder(tf.float32, [None, 1])
        self.u_tf = tf.placeholder(tf.float32, [None, 1])
        self.v_tf = tf.placeholder(tf.float32, [None, 1])

        # network + loss
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(
            self.x_tf, self.y_tf, self.t_tf)
        self.loss = (tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) +
                     tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) +
                     tf.reduce_mean(tf.square(self.f_u_pred)) +
                     tf.reduce_mean(tf.square(self.f_v_pred)))

        # optimizers
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        try:
            from tensorflow.contrib.opt import ScipyOptimizerInterface
            self.optimizer = ScipyOptimizerInterface(
                self.loss, method='L-BFGS-B',
                options={'maxiter':50000,'maxfun':50000,'maxcor':50,
                         'maxls':50,'ftol':np.finfo(float).eps})
        except Exception:
            self.optimizer = None

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def initialize_NN(self, layers):
        weights, biases = [], []
        for l in range(len(layers)-1):
            in_dim, out_dim = layers[l], layers[l+1]
            std = np.sqrt(2/(in_dim+out_dim))
            W = tf.Variable(tf.random.truncated_normal([in_dim,out_dim], stddev=std), dtype=tf.float32)
            b = tf.Variable(tf.zeros([1,out_dim], dtype=tf.float32), dtype=tf.float32)
            weights.append(W); biases.append(b)
        return weights, biases

    def neural_net(self, X, weights, biases):
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(len(weights)-1):
            H = tf.tanh(tf.add(tf.matmul(H, weights[l]), biases[l]))
        return tf.add(tf.matmul(H, weights[-1]), biases[-1])

    def net_NS(self, x, y, t):
        psi_and_p = self.neural_net(tf.concat([x,y,t],1), self.weights, self.biases)
        psi, p = psi_and_p[:,0:1], psi_and_p[:,1:2]
        u, v = tf.gradients(psi, y)[0], -tf.gradients(psi, x)[0]

        u_t,u_x,u_y = tf.gradients(u,t)[0],tf.gradients(u,x)[0],tf.gradients(u,y)[0]
        u_xx,u_yy = tf.gradients(u_x,x)[0],tf.gradients(u_y,y)[0]
        v_t,v_x,v_y = tf.gradients(v,t)[0],tf.gradients(v,x)[0],tf.gradients(v,y)[0]
        v_xx,v_yy = tf.gradients(v_x,x)[0],tf.gradients(v_y,y)[0]
        p_x,p_y = tf.gradients(p,x)[0],tf.gradients(p,y)[0]

        f_u = u_t + self.lambda_1*(u*u_x+v*u_y) + p_x - self.lambda_2*(u_xx+u_yy)
        f_v = v_t + self.lambda_1*(u*v_x+v*v_y) + p_y - self.lambda_2*(v_xx+v_yy)
        return u,v,p,f_u,f_v

    def callback(self, loss,l1,l2): 
        print('Loss: %.3e, l1: %.3f, l2: %.5f'%(loss,l1,l2))

    def train(self, nIter):
        tf_dict={self.x_tf:self.x,self.y_tf:self.y,self.t_tf:self.t,
                 self.u_tf:self.u,self.v_tf:self.v}
        start=time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam,tf_dict)
            if it%10==0:
                elapsed=time.time()-start
                loss_v,l1,l2=self.sess.run([self.loss,self.lambda_1,self.lambda_2],tf_dict)
                print(f"It:{it:6d}, Loss:{loss_v:.3e}, l1:{l1[0]:.3f}, l2:{l2[0]:.5f}, Time:{elapsed:.2f}")
                start=time.time()
        if self.optimizer is not None:
            self.optimizer.minimize(self.sess,feed_dict=tf_dict,
                                    fetches=[self.loss,self.lambda_1,self.lambda_2],
                                    loss_callback=self.callback)

    def predict(self,xs,ys,ts):
        tf_dict={self.x_tf:xs,self.y_tf:ys,self.t_tf:ts}
        return (self.sess.run(self.u_pred,tf_dict),
                self.sess.run(self.v_pred,tf_dict),
                self.sess.run(self.p_pred,tf_dict))

# -------------------------------------------------------------------
def plot_solution(X_star,u_star,index):
    lb,ub=X_star.min(0),X_star.max(0)
    x=np.linspace(lb[0],ub[0],200); y=np.linspace(lb[1],ub[1],200)
    X,Y=np.meshgrid(x,y)
    U_star=griddata(X_star,u_star.flatten(),(X,Y),method='cubic')
    plt.figure(index); plt.pcolor(X,Y,U_star,cmap='jet'); plt.colorbar()

def axisEqual3D(ax):
    ext=np.array([getattr(ax,f'get_{d}lim')() for d in 'xyz'])
    sz=ext[:,1]-ext[:,0]; ctr=np.mean(ext,1); m=max(abs(sz)); r=m/4
    for c,d in zip(ctr,'xyz'): getattr(ax,f'set_{d}lim')(c-r,c+r)

# -------------------------------------------------------------------
if __name__=="__main__":
    N_train=5000
    layers=[3,20,20,20,20,20,20,20,20,2]

    print("Loading data ...")
    data = scipy.io.loadmat(r'D:\1A_Coding\1A_Python\Git\PINN\PINNs\main\Data\cylinder_nektar_wake.mat')
    U_star,P_star,t_star,X_star=data['U_star'],data['p_star'],data['t'],data['X_star']
    N,T=X_star.shape[0],t_star.shape[0]

    XX,YY=np.tile(X_star[:,0:1],(1,T)),np.tile(X_star[:,1:2],(1,T))
    TT=np.tile(t_star,(1,N)).T
    UU,VV,PP=U_star[:,0,:],U_star[:,1,:],P_star
    x,y,t=XX.flatten()[:,None],YY.flatten()[:,None],TT.flatten()[:,None]
    u,v,p=UU.flatten()[:,None],VV.flatten()[:,None],PP.flatten()[:,None]

    idx=np.random.choice(N*T,N_train,replace=False)
    x_t,y_t,t_t,u_t,v_t=x[idx],y[idx],t[idx],u[idx],v[idx]

    print("Training (clean data)...")
    model=PhysicsInformedNN(x_t,y_t,t_t,u_t,v_t,layers)
    model.train(200000)

    snap=np.array([100])
    x_star,y_star,t_star_plot=X_star[:,0:1],X_star[:,1:2],TT[:,snap]
    u_star,v_star,p_star=U_star[:,0,snap],U_star[:,1,snap],P_star[:,snap]
    u_pred,v_pred,p_pred=model.predict(x_star,y_star,t_star_plot)
    l1,l2=model.sess.run([model.lambda_1,model.lambda_2])

    erru=np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    errv=np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    errp=np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)
    print(f"Errors: u={erru:.3e}, v={errv:.3e}, p={errp:.3e}")
    print(f"λ1={l1[0]:.4f}, λ2={l2[0]:.5f}")

    # add 1% noise
    print("Training (1% noisy data)...")
    noise=0.01
    u_tn=u_t+noise*np.std(u_t)*np.random.randn(*u_t.shape)
    v_tn=v_t+noise*np.std(v_t)*np.random.randn(*v_t.shape)
    modelN=PhysicsInformedNN(x_t,y_t,t_t,u_tn,v_tn,layers)
    modelN.train(200000)
    l1n,l2n=modelN.sess.run([modelN.lambda_1,modelN.lambda_2])
    print(f"Noisy λ1={l1n[0]:.4f}, λ2={l2n[0]:.5f}")

    # plotting
    lb,ub=X_star.min(0),X_star.max(0)
    x=np.linspace(lb[0],ub[0],200); y=np.linspace(lb[1],ub[1],200)
    X,Y=np.meshgrid(x,y)
    UU_star=griddata(X_star,u_pred.flatten(),(X,Y),method='cubic')
    VV_star=griddata(X_star,v_pred.flatten(),(X,Y),method='cubic')
    PP_star=griddata(X_star,p_pred.flatten(),(X,Y),method='cubic')
    plt.figure(); plt.imshow(PP_star,cmap='jet'); plt.colorbar(); plt.title("Predicted Pressure")
    plt.show()
