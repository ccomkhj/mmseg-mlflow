from tools.train_arg_in import main as train_mmseg
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import numpy as np

def objective(params):
    global epoch_hyperopt
    lr = params.get('lr')
    momentum = params.get('momentum')
    weight_decay = params.get('weight_decay')
    config = 'configs/hexa/GU_new_50000_96_Cross.py'
    loss = train_mmseg(config, lr, momentum, weight_decay, epoch=epoch_hyperopt)
    epoch_hyperopt += 1
    return {'loss': loss, 'status': STATUS_OK}

def main():

    global epoch_hyperopt
    epoch_hyperopt =  0

    search_space = {'lr':hp.loguniform('lr',np.log(0.001), np.log(0.05)),
                    'momentum':hp.uniform('momentum',0.3,0.9),
                    'weight_decay':hp.loguniform('weight_decay',np.log(0.00001),np.log(0.005))}

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=Trials()
    )

    print(best_result)
    return best_result

if __name__ == "__main__":
    main()