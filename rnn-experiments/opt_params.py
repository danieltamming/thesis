sst_params = {
    'pct':
    {
        'synonym':
        {
            0.1: {'aug': (0.5, 28), 'no': 26},
            0.2: {'aug': (0.4, 54), 'no': 53}, 
            0.3: {'aug': (0.6, 28), 'no': 47}, 
            0.4: {'aug': (0.6, 72), 'no': 57}, 
            0.5: {'aug': (0.5, 39), 'no': 63}, 
            0.6: {'aug': (0.5, 26), 'no': 36}, 
            0.7: {'aug': (0.5, 79), 'no': 98}, 
            0.8: {'aug': (0.9, 94), 'no': 36}, 
            0.9: {'aug': (0.7, 77), 'no': 86},
            1.0: {'aug': (0.8, 68), 'no': 88} 
        },
        'trans':
        {
            0.1: {'aug': (0.5, 23), 'no': 18},
            0.2: {'aug': (0.3, 32), 'no': 27}, 
            0.3: {'aug': (0.3, 77), 'no': 33}, 
            0.4: {'aug': (0.1, 50), 'no': 59}, 
            0.5: {'aug': (0.2, 50), 'no': 62}, 
            0.6: {'aug': (0.1, 68), 'no': 62}, 
            0.7: {'aug': (0.2, 64), 'no': 35}, 
            0.8: {'aug': (0.1, 41), 'no': 34}, 
            0.9: {'aug': (0.2, 65), 'no': 91},
            1.0: {'aug': (0.3, 73), 'no': 53} 
        },
        'context':
        {
            0.1: {'aug': (0.6, 19), 'no': 15},
            0.2: {'aug': (0.1, 22), 'no': 98}, 
            0.3: {'aug': (0.2, 37), 'no': 59}, 
            0.4: {'aug': (0.7, 94), 'no': 87}, 
            0.5: {'aug': (0.1, 39), 'no': 74}, 
            0.6: {'aug': (0.1, 39), 'no': 91}, 
            0.7: {'aug': (0.1, 20), 'no': 21}, 
            0.8: {'aug': (0.1, 32), 'no': 98}, 
            0.9: {'aug': (0.1, 21), 'no': 52},
            1.0: {'aug': (0.1, 33), 'no': 50} 
        }
    },
    'bal':
    {
        'synonym':
        {
            0.1: {'aug': (0.6, 1), 'under': 20, 'over': 1},
            0.2: {'aug': (0.6, 10), 'under': 40, 'over': 7}, 
            0.3: {'aug': (0.7, 7), 'under': 23, 'over': 14}, 
            0.4: {'aug': (0.7, 14), 'under': 63, 'over': 22}, 
            0.5: {'aug': (0.7, 18), 'under': 52, 'over': 40}, 
            0.6: {'aug': (0.7, 44), 'under': 69, 'over': 49}, 
            0.7: {'aug': (0.8, 79), 'under': 92, 'over': 61}, 
            0.8: {'aug': (0.7, 99), 'under': 40, 'over': 99}, 
            0.9: {'aug': (0.8, 91), 'under': 83, 'over': 81} 
        },
        'trans':
        {
            0.1: {'aug': (0.6, 5), 'under': 86, 'over': 3},
            0.2: {'aug': (0.6, 93), 'under': 35, 'over': 2}, 
            0.3: {'aug': (0.6, 39), 'under': 25, 'over': 5}, 
            0.4: {'aug': (0.6, 35), 'under': 90, 'over': 16}, 
            0.5: {'aug': (0.9, 81), 'under': 98, 'over': 12}, 
            0.6: {'aug': (0.9, 78), 'under': 61, 'over': 15}, 
            0.7: {'aug': (0.9, 64), 'under': 96, 'over': 44}, 
            0.8: {'aug': (0.9, 93), 'under': 91, 'over': 92}, 
            0.9: {'aug': (0.9, 79), 'under': 37, 'over': 74} 
        },
        'context':
        {
            0.1: {'aug': (0.3, 1), 'under': 31, 'over': 1},
            0.2: {'aug': (0.3, 3), 'under': 41, 'over': 9}, 
            0.3: {'aug': (0.4, 22), 'under': 30, 'over': 17}, 
            0.4: {'aug': (0.4, 74), 'under': 86, 'over': 10}, 
            0.5: {'aug': (0.4, 58), 'under': 33, 'over': 74}, 
            0.6: {'aug': (0.4, 33), 'under': 27, 'over': 12}, 
            0.7: {'aug': (0.4, 36), 'under': 59, 'over': 88}, 
            0.8: {'aug': (0.4, 33), 'under': 98, 'over': 56}, 
            0.9: {'aug': (0.9, 78), 'under': 35, 'over': 75} 
        }
    }
}

subj_params = {
    'pct':
    {
        'synonym':
        {
            0.1: {'aug': (0.4, 95), 'no': 73},
            0.2: {'aug': (0.5, 92), 'no': 90}, 
            0.3: {'aug': (0.3, 94), 'no': 99}, 
            0.4: {'aug': (0.4, 86), 'no': 62}, 
            0.5: {'aug': (0.6, 83), 'no': 76}, 
            0.6: {'aug': (0.4, 91), 'no': 89}, 
            0.7: {'aug': (0.6, 69), 'no': 85}, 
            0.8: {'aug': (0.4, 89), 'no': 89}, 
            0.9: {'aug': (0.3, 95), 'no': 74},
            1.0: {'aug': (0.4, 63), 'no': 65} 
        },
        'trans':
        {
            0.1: {'aug': (0.3, 66), 'no': 75},
            0.2: {'aug': (0.2, 69), 'no': 68}, 
            0.3: {'aug': (0.3, 61), 'no': 60}, 
            0.4: {'aug': (0.3, 87), 'no': 94}, 
            0.5: {'aug': (0.3, 77), 'no': 95}, 
            0.6: {'aug': (0.3, 85), 'no': 98}, 
            0.7: {'aug': (0.5, 98), 'no': 87}, 
            0.8: {'aug': (0.5, 68), 'no': 55}, 
            0.9: {'aug': (0.5, 81), 'no': 63},
            1.0: {'aug': (0.5, 78), 'no': 79} 
        },
        'context':
        {
            0.1: {'aug': (0.2, 43), 'no': 98},
            0.2: {'aug': (0.1, 96), 'no': 53}, 
            0.3: {'aug': (0.1, 71), 'no': 68}, 
            0.4: {'aug': (0.1, 76), 'no': 91}, 
            0.5: {'aug': (0.2, 86), 'no': 54}, 
            0.6: {'aug': (0.2, 57), 'no': 77}, 
            0.7: {'aug': (0.2, 84), 'no': 82}, 
            0.8: {'aug': (0.1, 87), 'no': 84}, 
            0.9: {'aug': (0.2, 65), 'no': 93},
            1.0: {'aug': (0.3, 96), 'no': 64} 
        }
    },
    'bal':
    {
        'synonym':
        {
            0.1: {'aug': (0.4, 2), 'under': 53, 'over': 3},
            0.2: {'aug': (0.4, 31), 'under': 74, 'over': 5}, 
            0.3: {'aug': (0.4, 96), 'under': 31, 'over': 40}, 
            0.4: {'aug': (0.5, 95), 'under': 90, 'over': 57}, 
            0.5: {'aug': (0.5, 93), 'under': 99, 'over': 38}, 
            0.6: {'aug': (0.6, 85), 'under': 95, 'over': 78}, 
            0.7: {'aug': (0.7, 97), 'under': 99, 'over': 92}, 
            0.8: {'aug': (0.7, 89), 'under': 81, 'over': 77}, 
            0.9: {'aug': (0.7, 98), 'under': 82, 'over': 94} 
        },
        'trans':
        {
            0.1: {'aug': (0.3, 68), 'under': 45, 'over': 4},
            0.2: {'aug': (0.4, 51), 'under': 23, 'over': 6}, 
            0.3: {'aug': (0.6, 55), 'under': 87, 'over': 74}, 
            0.4: {'aug': (0.7, 90), 'under': 91, 'over': 95}, 
            0.5: {'aug': (0.8, 89), 'under': 98, 'over': 99}, 
            0.6: {'aug': (0.9, 62), 'under': 83, 'over': 75}, 
            0.7: {'aug': (0.9, 90), 'under': 99, 'over': 99}, 
            0.8: {'aug': (0.9, 80), 'under': 39, 'over': 90}, 
            0.9: {'aug': (0.9, 95), 'under': 98, 'over': 87} 
        },
        'context':
        {
            0.1: {'aug': (0.2, 58), 'under': 51, 'over': 1},
            0.2: {'aug': (0.2, 71), 'under': 37, 'over': 7}, 
            0.3: {'aug': (0.3, 89), 'under': 39, 'over': 88}, 
            0.4: {'aug': (0.3, 92), 'under': 90, 'over': 47}, 
            0.5: {'aug': (0.3, 79), 'under': 77, 'over': 85}, 
            0.6: {'aug': (0.4, 83), 'under': 88, 'over': 67}, 
            0.7: {'aug': (0.7, 93), 'under': 81, 'over': 80}, 
            0.8: {'aug': (0.7, 70), 'under': 97, 'over': 60}, 
            0.9: {'aug': (0.7, 53), 'under': 62, 'over': 92} 
        }
    }
}

sfu_params = {
    'pct':
    {
        'synonym':
        {
            0.2: {'aug': (0.6, 45), 'no': 91}, 
            0.4: {'aug': (0.2, 43), 'no': 49}, 
            0.6: {'aug': (0.3, 61), 'no': 80}, 
            0.8: {'aug': (0.6, 61), 'no': 29}, 
            1.0: {'aug': (0.1, 90), 'no': 79} 
        },
        'trans':
        {
            0.2: {'aug': (0.3, 81), 'no': 58}, 
            0.4: {'aug': (0.5, 73), 'no': 76}, 
            0.6: {'aug': (0.5, 47), 'no': 84}, 
            0.8: {'aug': (0.5, 58), 'no': 79}, 
            1.0: {'aug': (0.8, 84), 'no': 54} 
        },
        'context':
        {
            0.2: {'aug': (0.6, 57), 'no': 34}, 
            0.4: {'aug': (0.5, 69), 'no': 87}, 
            0.6: {'aug': (0.3, 48), 'no': 27}, 
            0.8: {'aug': (0.3, 99), 'no': 73},
            1.0: {'aug': (0.1, 95), 'no': 51} 
        }
    },
    'bal':
    {
        'synonym':
        {
            0.2: {'aug': (0.1, 45), 'under': 99, 'over': 81}, 
            0.4: {'aug': (0.3, 47), 'under': 63, 'over': 99}, 
            0.6: {'aug': (0.3, 65), 'under': 52, 'over': 61}, 
            0.8: {'aug': (0.3, 56), 'under': 76, 'over': 54}, 
        },
        'trans':
        {
            0.2: {'aug': (0.1, 26), 'under': 90, 'over': 45}, 
            0.4: {'aug': (0.4, 97), 'under': 85, 'over': 54}, 
            0.6: {'aug': (0.9, 72), 'under': 77, 'over': 51}, 
            0.8: {'aug': (0.8, 79), 'under': 95, 'over': 44}, 
        },
        'context':
        {
            0.2: {'aug': (0.1, 85), 'under': 10, 'over': 37}, 
            0.4: {'aug': (0.5, 49), 'under': 54, 'over': 55}, 
            0.6: {'aug': (0.4, 88), 'under': 68, 'over': 15}, 
            0.8: {'aug': (0.8, 69), 'under': 73, 'over': 84}, 
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

    plt.rc('font', size=45)

    param_list = [sst_params, subj_params, sfu_params]
    data_name_list = ['sst', 'subj', 'sfu']
    for params, data_name in zip(param_list, data_name_list):
        for setting in ['bal', 'pct']:
            colors = ['tab:blue', 'tab:orange', 'tab:purple']
            methods = ['synonym', 'trans', 'context']
            method_names = ['Synonym Replacement', 'Backtranslation', 'Contextual Augmentation']
            plt.figure(figsize=(21.6, 16.2))

            for method, method_name, color in zip(methods, method_names, colors):
                sns.lineplot(
                    list(params[setting][method].keys()), 
                    [d['aug'][0] for d in params[setting][method].values()],
                    label=method_name,
                    color=color,
                    marker='o', 
                    markerfacecolor='none', 
                    markeredgecolor=color, 
                    markeredgewidth=1.5
                )

            plt.yticks(np.arange(0.0, 1.1, 0.1))
            if setting == 'bal':
                plt.xlabel('Percentage of Minority Label Examples Left In Training Set')
            else:
                plt.xlabel('Percentage of Training Set')
            plt.ylabel('Augmentation Parameter')
            if setting == 'bal':
                plt.legend(loc='lower right')
            else:
                plt.legend(loc='upper left')
            filename = 'figures/parameter-{}-{}.png'.format(data_name, setting)
            # plt.savefig(filename, dpi=200)
            plt.tight_layout()
            # plt.savefig(filename)
            plt.show()