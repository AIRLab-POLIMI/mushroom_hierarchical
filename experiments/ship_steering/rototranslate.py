import numpy as np
from mushroom.utils.angles import normalize_angle


def rototranslate(inputs):
        new_states = np.zeros(4)
        active_direction = inputs[0]
        x = inputs[1][0]
        y = inputs[1][1]
        theta = inputs[1][2]
        theta_dot = inputs[1][3]
        x0 = inputs[2][0]
        y0 = inputs[2][1]

        if active_direction < 4:
            small_offset = 40
            large_offset = 75
        else:
            small_offset = 40
            large_offset = 40

        if active_direction == 0:   #R
            new_states[0] = x-x0+small_offset
            new_states[1] = y-y0+large_offset
            new_states[2] = normalize_angle(theta)
        elif active_direction == 1: #D
            new_states[0] = y0-y+small_offset
            new_states[1] = x-x0+large_offset
            new_states[2] = normalize_angle(theta + np.pi/2)
        elif active_direction == 2: #L
            new_states[0] = x0-x+small_offset
            new_states[1] = y0-y+large_offset
            new_states[2] = normalize_angle(theta +np.pi)
        elif active_direction == 3: #U
            new_states[0] = y-y0+small_offset
            new_states[1] = x0-x+large_offset
            new_states[2] = normalize_angle(theta+1.5*np.pi)
        elif active_direction == 4: #UR
            new_states[0] = x-x0+small_offset
            new_states[1] = y-y0+small_offset
            new_states[2] = normalize_angle(theta)
        elif active_direction == 5: #DR
            new_states[0] = y0-y+small_offset
            new_states[1] = x-x0+small_offset
            new_states[2] = normalize_angle(theta + np.pi/2)
        elif active_direction == 6: #DL
            new_states[0] = x0-x+small_offset
            new_states[1] = y0-y+small_offset
            new_states[2] = normalize_angle(theta + np.pi)
        else:                       #UL
            new_states[0] = y-y0+small_offset
            new_states[1] = x0-x+small_offset
            new_states[2] = normalize_angle(theta + np.pi*1.5)

        new_states[3] = theta_dot

        return new_states

