sst_params = {
    'pct':
    {
        'synonym':
        {
            0.1: 0.5,
            0.2: 0.4,
            0.3: 0.6,
            0.4: 0.6,
            0.5: 0.5,
            0.6: 0.5,
            0.7: 0.5,
            0.8: 0.9,
            0.9: 0.7,
            1.0: 0.8
        },
        'trans':
        {
            0.1: 0.5,
            0.2: 0.3,
            0.3: 0.3,
            0.4: 0.1,
            0.5: 0.2,
            0.6: 0.1,
            0.7: 0.2,
            0.8: 0.1,
            0.9: 0.2,
            1.0: 0.3
        },
        'context':
        {
            0.1: 0.6,
            0.2: 0.1,
            0.3: 0.2,
            0.4: 0.7,
            0.5: 0.1,
            0.6: 0.1,
            0.7: 0.1,
            0.8: 0.1,
            0.9: 0.1,
            1.0: 0.1
        }
    },
    'bal':
    {
        'synonym':
        {
            0.1: 0.6,
            0.2: 0.6,
            0.3: 0.7,
            0.4: 0.7,
            0.5: 0.7,
            0.6: 0.7,
            0.7: 0.8,
            0.8: 0.7,
            0.9: 0.8
        },
        'trans':
        {
            0.1: 0.6,
            0.2: 0.6,
            0.3: 0.6,
            0.4: 0.6,
            0.5: 0.9,
            0.6: 0.9,
            0.7: 0.9,
            0.8: 0.9,
            0.9: 0.9
        },
        'context':
        {
            0.1: 0.3,
            0.2: 0.3,
            0.3: 0.4,
            0.4: 0.4,
            0.5: 0.4,
            0.6: 0.4, 
            0.7: 0.4, 
            0.8: 0.4, 
            0.9: 0.9
        }
    }
}

subj_params = {
    'pct':
    {
        'synonym':
        {
            0.1: 0.4,
            0.2: 0.5,
            0.3: 0.3,
            0.4: 0.4,
            0.5: 0.6,
            0.6: 0.4,
            0.7: 0.6,
            0.8: 0.4,
            0.9: 0.3,
            1.0: 0.4
        },
        'trans':
        {
            0.1: 0.3,
            0.2: 0.2,
            0.3: 0.3,
            0.4: 0.3,
            0.5: 0.3,
            0.6: 0.3,
            0.7: 0.5,
            0.8: 0.5,
            0.9: 0.5,
            1.0: 0.5
        },
        'context':
        {
            0.1: 0.2,
            0.2: 0.1,
            0.3: 0.1,
            0.4: 0.1,
            0.5: 0.2,
            0.6: 0.2,
            0.7: 0.2,
            0.8: 0.1,
            0.9: 0.2,
            1.0: 0.3
        }
    },
    'bal':
    {
        'synonym':
        {
            0.1: 0.4,
            0.2: 0.4,
            0.3: 0.4,
            0.4: 0.5,
            0.5: 0.5,
            0.6: 0.6,
            0.7: 0.7,
            0.8: 0.7,
            0.9: 0.7
        },
        'trans':
        {
            0.1: 0.3,
            0.2: 0.4,
            0.3: 0.6,
            0.4: 0.7,
            0.5: 0.8,
            0.6: 0.9,
            0.7: 0.9,
            0.8: 0.9,
            0.9: 0.9
        },
        'context':
        {
            0.1: 0.2,
            0.2: 0.2,
            0.3: 0.3,
            0.4: 0.3,
            0.5: 0.3,
            0.6: 0.4,
            0.7: 0.7,
            0.8: 0.7,
            0.9: 0.7
        }
    }
}

sfu_params = {
    'pct':
    {
        'synonym':
        {
            0.2: 0.6,
            0.4: 0.2,
            0.6: 0.3,
            0.8: 0.6,
            1.0: 0.1
        },
        'trans':
        {
            0.2: 0.3,
            0.4: 0.5,
            0.6: 0.5,
            0.8: 0.5,
            1.0: 0.8
        },
        'context':
        {
            0.2: 0.6,
            0.4: 0.5,
            0.6: 0.3,
            0.8: 0.3, 
            1.0: 0.1
        }
    },
    'bal':
    {
        'synonym':
        {
            0.2: 0.1,
            0.4: 0.3,
            0.6: 0.3,
            0.8: 0.3
        },
        'trans':
        {
            0.2: 0.1,
            0.4: 0.4,
            0.6: 0.9,
            0.8: 0.8
        },
        'context':
        {
            0.2: 0.1,
            0.4: 0.5,
            0.6: 0.4,
            0.8: 0.8
        }
    }
}

if __name__=="__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import matplotlib.style as style
    # style.use('fivethirtyeight')
    from mpl_toolkits.mplot3d import Axes3D

    param_list = [sst_params, subj_params, sfu_params]
    data_name_list = ['sst', 'subj', 'sfu']
    for params, data_name in zip(param_list, data_name_list):
        for setting in ['bal', 'pct']:
            colors = ['tab:blue', 'tab:orange', 'tab:purple']
            methods = ['synonym', 'trans', 'context']
            method_names = ['Synonym Replacement', 'Backtranslation', 'BERT Augmentation']

            for method, method_name, color in zip(methods, method_names, colors):
                sns.lineplot(
                    list(params[setting][method].keys()), 
                    list(params[setting][method].values()),
                    label=method_name,
                    color=color
                )

            plt.yticks(np.arange(0.0, 1.1, 0.1))
            if setting == 'bal':
                plt.xlabel('Percentage of Minority Label Examples Left In Training Set')
            else:
                plt.xlabel('Percentage of Training Set')
            plt.ylabel('Augmentation Parameter')
            plt.legend(loc='upper left')
            filename = 'figures/parameter-{}-{}.png'.format(data_name, setting)
            plt.savefig(filename, dpi=200)
            plt.show()