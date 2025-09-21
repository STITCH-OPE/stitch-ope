import numpy as np
import matplotlib.pyplot as plt

def plot_diffusion_comparison(arr_guided, arr_unguided, target_monte_carlo, behavior_monte_carlo):
    guided_diffusion = np.mean(arr_guided)
    unguided_diffusion = np.mean(arr_unguided)
    guided_std = np.std(arr_guided)
    unguided_std = np.std(arr_unguided)
    
    methods = ['Guided Diffusion', 'Unguided Diffusion']
    means = [guided_diffusion, unguided_diffusion]
    stds = [guided_std, unguided_std]
    
    fig, ax = plt.subplots()
    ax.bar(methods, means, yerr=stds, capsize=10, color=['blue', 'orange'])
    
    ax.axhline(y=target_monte_carlo, color='r', linestyle='--', label='Target Monte Carlo')
    ax.axhline(y=behavior_monte_carlo, color='g', linestyle='--', label='Behavior Monte Carlo')
    
    ax.set_ylabel('Mean Value')
    ax.set_title('Comparison of Guided and Unguided Diffusion')
    ax.legend()

    y_min = min(min(arr_guided), min(arr_unguided), target_monte_carlo, behavior_monte_carlo) - 1
    y_max = max(max(arr_guided), max(arr_unguided), target_monte_carlo, behavior_monte_carlo) + 1
    ax.set_ylim(y_min, y_max)
    
    plt.savefig('diffusion_cmp.pdf')

arr_guided = [50.05767044236528, 51.35555253907515, 52.0432337022697, 52.03796691764933]
arr_unguided = [53.30890602703753, 53.84446736865064, 54.17339721672438, 52.72909077558124]
target_monte_carlo = 50
behavior_monte_carlo = 53
#should probably load from the json you mentioned

plot_diffusion_comparison(arr_guided=arr_guided, arr_unguided=arr_unguided, target_monte_carlo=target_monte_carlo, behavior_monte_carlo=behavior_monte_carlo)
