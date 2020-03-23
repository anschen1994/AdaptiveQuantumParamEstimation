import numpy as np 
import logging 
from qutip import *


class DynaicsModel():
    def __init__(self, args, allowed_control):
        self.args = args
        # self.model_name = args.model_name
        self.allowed_control = allowed_control
        self._check()

    def probs(self, params, params_est):
        """
        probability func of args
        """
        pass 

    def logprobs(self, params, params_est):
        return np.log(self.probs(params, params_est))

    def grad_logprobs(self, params, params_est):
        """
        gradient of log likelihood
        """
        pass 
    
    def _getname(self):
        print(self.name)

    def _getmodel_info(self):
        print('This general model for people to do quantum parameter estimation adaptively. \n People can construct an instance of this class to study their research')

    def _check(self):
        if self.args.control_type not in self.allowed_control:
            logging.error('User input control_type {} ,which is not allowable, should be one of [{}, {}]'.format(self.args.control_type, self.allowed_control[0], self.allowed_control[1]))
            raise NameError('This kind of control has not been included in our study')


class Magnetometer(DynaicsModel):
    def __init__(self, args, allowed_control=['QEC', 'Unitary']):
        super(Magnetometer, self).__init__(args, allowed_control)
        self.control_type = args.control_type
        self.model_name = 'Magnetometer' + args.control_type

    def probs(self, params, params_est):
        if self.control_type == 'QEC':
            delta = params - params_est
            dephasing = np.exp(-2*self.args.gamma*self.args.t*np.sin(delta)**2)
            phase = np.cos(2*self.args.B*self.args.t*np.sin(delta))
            p0 = 0.5 * (1 + dephasing * phase)
            p1 = 0.5 * (1 - dephasing * phase)
            return np.array([p0, p1])

        elif self.control_type == 'Unitary':
            delta = params - params_est
            p_avg = 0.5 * (params + params_est)
            phase = self.args.B * self.args.t * np.sqrt(2-2*np.cos(delta))
            p0 = np.cos(phase)**2
            p1 = np.cos(p_avg)**2 * np.sin(phase)**2
            p2 = np.sin(p_avg)**2 * np.sin(phase)**2
            return np.array([p0, p1, p2])

    def grad_logprobs(self, params, params_est):
        if self.control_type == 'QEC':
            delta = params - params_est
            f1 = np.cos(2*self.args.B*self.args.t*np.sin(delta))
            f2 = np.sin(2*self.args.B*self.args.t*np.sin(delta))
            f3 = np.exp(2*self.args.gamma*self.args.t*np.sin(delta)**2)
            numerator = 2*self.args.t*np.cos(delta)*(2*self.args.gamma*np.sin(delta)*f1+self.args.B*f2)
            dlogp0 = - numerator / (f1 + f3)
            dlogp1 = numerator / (f3 - f1)
            return np.array([dlogp0, dlogp1])

        elif self.control_type == 'Unitary':
            pass 


class Gyroscope(DynaicsModel):
    def __init__(self, args, allowed_control=['QEC', 'Unitary']):
        super(Gyroscope, self).__init__(args, allowed_control)
        self.control_type = args.control_type
        self.model_name = 'Gyroscope' + args.control_type

    def probs(self, params, params_est):
        if self.control_type == 'QEC':
            delta = params - params_est
            dephasing = np.exp(-self.args.gamma*self.args.t + 0.5 * self.args.gamma / delta * np.sin(2*self.args.t*delta))
            phase = np.cos(2*self.args.B*(1-np.cos(delta*self.args.t))/delta)
            p0 = 0.5 * (1 + dephasing * phase)
            p1 = 0.5 * (1 - dephasing * phase)
            return np.array([p0, p1])
            
        elif self.control_type == 'Unitary':
            H0 = -0.5 * params_est * sigmaz()
            H1 = self.args.B * sigmax()
            H2 = self.args.B * sigmay()
            Args = {'Omega': params, 'Omegaest': params_est}
            def H1_coef(t, args):
                return np.cos(args['Omega']*t) - np.cos(args['Omegaest']*t)
            def H2_coef(t, args):
                return np.sin(args['Omegaest']*t) - np.sin(args['Omega']*t)
            H = [H0, [H1, H1_coef], [H2, H2_coef]]
            psi0 = ket('1')
            t_var = np.linspace(0,self.args.t,10)
            proj1 = 0.5 * (sigmaz() + identity(2))
            proj0 = 0.5 * (identity(2) - sigmaz())
            res = sesolve(H, psi0, t_var, e_ops=[proj0, proj1], args=Args)
            [p0, p1] = res.expect
            return np.array([p0[-1], p1[-1]])
    
# import argparse
# p = argparse.ArgumentParser()
# p.add_argument('--model-name', type=str, default='magnet')
# p.add_argument('--control-type', type=str, default='QE')
# args = p.parse_args()
# model = Magnetometer(args)



