from model import Magnetometer, Gyroscope
import logging 
import argparse
import numpy as np 
from scipy.optimize import minimize
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser() 
parser.add_argument('--B', type=float, default=0.1, help='magnetic field strength')
parser.add_argument('--gamma', type=float, default=0.05, help='dephasing strength')
parser.add_argument('--t', type=float, default=5, help='total sensing time')
parser.add_argument('--control-type', type=str, default='QEC', help='a str used to judge kind control, can be one of [QEC, Unitary]')
parser.add_argument('--is-analytical', type=bool, default=False, help='a tag juding whether porbability distribution can be obtained analytically')
parser.add_argument('--Nsample', type=int, default=1000, help='number of adaptive process repeated')

def update(model, prev_data, params, params_est, m):
    """
    Parameters:
    ---------------------------------------------------------------
    model: an instance of dynamics model
    prev_data: list, consists of turple (params_est, sample_p)
    """
    p = model.probs(params, params_est)
    if len(p) == 2:
        num0 = np.sum(np.random.uniform(size=m)<p[0])
        q = np.array([num0/m, 1-num0/m])
    elif len(p) == 3:
        sample = np.random.uniform(size=m)
        num0 = np.sum(sample<p[0])
        num2 = np.sum(sample>p[0]+p[1])
        q = np.array([num0, m-num0-num2, num2]) / m
        assert np.abs(np.sum(q)-1) < 1e-5

    def cross_entropy(x):
        logpx = model.logprobs(x, params_est)
        loss = -np.sum(q*logpx)
        if prev_data != []:
            for pdata in prev_data:
                logpx = model.logprobs(x, pdata[0])
                loss -= np.sum(pdata[1]*logpx)
        return np.array(loss / (len(prev_data) + 1))

    if model.args.is_analytical:
        def grad_cross_entropy(x):
            dlogpx = model.grad_logprobs(x, params_est)
            grad = -np.sum(q*dlogpx)
            if prev_data != []:
                for pdata in prev_data:
                    dlogpx = model.grad_logprobs(x, pdata[0])
                    grad -= np.sum(pdata[1]*dlogpx)
            return grad / (1 + len(prev_data))
        alpha = 1e-4
        x_best = np.random.normal(params, 0.5*np.abs(params_est-params))
        grad = grad_cross_entropy(x_best)
        while np.abs(grad) > 1e-3:
            x_best -= alpha * grad
            grad = grad_cross_entropy(x_best)
        return x_best, (params_est, q)

    else:
        loss = np.inf 
        dis = np.abs(params - params_est)
        x_var = np.linspace(params - dis, params + dis, 100)
        x_best = params_est
        for x in x_var:
            loss_c = cross_entropy(x)
            if loss_c < loss:
                loss = loss_c
                x_best = x 
        return x_best, (params_est, q)


def adaptive(model, params, params_est, m, Nstep):
    result = [params_est]
    params_hat = params_est
    prev_data = []
    for i in range(Nstep):
        params_hat, data = update(model, prev_data, params, params_hat, m)
        prev_data.append(data) 
        result.append(params_hat)
    return result


if __name__ == '__main__':
    args = parser.parse_args()
    # model = Magnetometer(args)
    model = Gyroscope(args)
    logging.basicConfig(filename=model.model_name+'.log', filemode='w', level=logging.DEBUG, format='%(asctime)s | %(levelname)s: %(message)s')
    theta = 0.3
    # theta_hat = 0.4
    m = 10
    Nstep = 10
    estimators = []
    logging.info('The following log is recording data collecting process of '+ model.model_name + '!')
    logging.info('=' * 100)
    theta_hat = 0.2
    for i in range(args.Nsample):
        res = adaptive(model, theta, theta_hat, m, Nstep)
        estimators.append(res)
        # print(res)
        if i % 10 == 1:
            np.save('./data/'+model.model_name+'.npy', np.array(estimators))
            logging.info('Epoch {} | SAVED!'.format(i))
            print('Epoch {} | SAVED!'.format(i))
    np.save('./data/'+model.model_name+'.npy', np.array(estimators))
    logging.info('#MISSION ACCOMPLISH#')







