import math


def linear_explore(init, final, num_steps, current_steps):
    single_step = (init-final)/float(num_steps)
    if current_steps < num_steps:
        return init - (single_step*current_steps)
    else:
        return final


def update_epsilon(current_steps):
    eps_final = 0.01
    eps_start = 1.0
    decay = 200000
    epsilon = eps_final + (eps_start - eps_final) * math.exp(-1 * ((current_steps + 1) / decay))
    return epsilon