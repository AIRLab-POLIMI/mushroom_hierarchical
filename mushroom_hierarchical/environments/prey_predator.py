import numpy as np

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces
from mushroom.utils.angles import *
from mushroom.utils.viewer import Viewer


class PreyPredator(Environment):
    """
    A prey-predator environment environment. A Predator must catch a faster prey
    in an environment with obstacles.
    """
    def __init__(self):
        self._rotation_radius = 0.6
        self._catch_radius = 0.4

        self._v_prey = 0.13
        self._v_predator = 0.1
        self._dt = 0.1

        self._omega_prey = self._v_prey / self._rotation_radius
        self._omega_predator = self._v_predator / self._rotation_radius

        self._max_x = 5.0
        self._max_y = 5.0

        self._obstacles = [
            (np.array([self._max_x/5,
                       self._max_y - 3.8*self._catch_radius]),
             np.array([self._max_x,
                       self._max_y - 3.8*self._catch_radius])),

            (np.array([-3/5*self._max_x,
                       self._max_y/4]),
             np.array([-3/5*self._max_x,
                       -3/10*self._max_y])),

            (np.array([-3/5*self._max_x + 3.8*self._catch_radius,
                       self._max_y / 4]),
             np.array([-3/5*self._max_x + 3.8*self._catch_radius,
                       -3/10*self._max_y])),

            (np.array([-3/5*self._max_x,
                       self._max_y/4]),
             np.array([-3/5*self._max_x + 3.8*self._catch_radius,
                       self._max_y/4]))
            ]

        # Add bounds of the map
        self._obstacles += [(np.array([-self._max_x, -self._max_y]),
                             np.array([-self._max_x, self._max_y])),

                            (np.array([-self._max_x, -self._max_y]),
                            np.array([self._max_x, -self._max_y])),

                            (np.array([self._max_x, self._max_y]),
                             np.array([-self._max_x, self._max_y])),

                            (np.array([self._max_x, self._max_y]),
                             np.array([self._max_x, -self._max_y]))
                            ]

        high = np.array([self._max_x, self._max_y, np.pi,
                         self._max_x, self._max_y, np.pi])


        # MDP properties
        horizon = 500
        gamma = 0.99

        observation_space = spaces.Box(low=-high, high=high)
        action_space = spaces.Box(low=np.array([0,
                                                -self._omega_predator]),
                                  high=np.array([self._v_predator,
                                                 self._omega_predator]))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # Visualization
        width = 500
        height = int(width * self._max_y / self._max_x)
        self._viewer = Viewer(2*self._max_x, 2*self._max_y, width, height)

        super(PreyPredator, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = np.array([0., 0., 0.,
                                    self._max_x/2, self._max_y/2, np.pi/2])
            self._state = np.array([3., 1., np.pi/2,
                                    3., 2., np.pi/2])

            ok = False

            while not ok:
                self._state = np.random.uniform(
                    low=self.info.observation_space.low,
                    high=self.info.observation_space.high)

                delta_norm = np.linalg.norm(self._state[:2] - self._state[3:5])
                ok = delta_norm > self._catch_radius

        else:
            self._state = state
            self._state[2] = normalize_angle(self._state[2])
            self._state[5] = normalize_angle(self._state[5])

        return self._state

    def step(self, action):
        # compute new predator state
        u = self._bound(action,
                        self.info.action_space.low,
                        self.info.action_space.high)

        state_predator = self._state[:3]
        state_predator = self._differential_drive_dynamics(state_predator, u)

        # Compute new prey state
        u_prey = self._prey_controller(self._state)
        state_prey = self._state[3:]
        state_prey = self._differential_drive_dynamics(state_prey, u_prey)

        # Update state
        self._state = np.concatenate([state_predator, state_prey], 0)

        delta_norm_new = np.linalg.norm(self._state[:2]-self._state[3:5])

        if delta_norm_new < self._catch_radius:
            collision, _ = self._check_collision(self._state[:2],
                                                 self._state[3:5])
            if collision is None:
                absorbing = True
            else:
                absorbing = False
        else:
            absorbing = False
        reward = -delta_norm_new

        return self._state, reward, absorbing, {}

    def _prey_controller(self, state):
        delta_norm = np.linalg.norm(state[:2] - state[3:5])

        if delta_norm > 3.0:
            velocity_prey = 0
        elif delta_norm > 1.5:
            velocity_prey = self._v_prey / 2
        else:
            velocity_prey = self._v_prey

        attack_angle = normalize_angle(np.arctan2(state[4] - state[1],
                                                  state[3] - state[0]))

        angle_current = shortest_angular_distance(state[5], attack_angle)

        if velocity_prey > 0:
            # check attack angle collision
            cos_theta = np.cos(attack_angle)
            sin_theta = np.sin(attack_angle)
            increment = 2.5*self._rotation_radius*np.array([cos_theta,
                                                            sin_theta])

            collision, i = self._check_collision(state[3:5],
                                                 state[3:5]+increment)

            if collision is not None:
                obstacle = self._obstacles[i]

                v_obst = self._segment_to_vector(*obstacle)
                v_attack = state[3:5] - state[0:2]

                angle = self._vector_angle(v_obst, v_attack)

                if 0 <= angle <= np.pi/2 or angle <= -np.pi/2:
                    rotation_sign = +1
                else:
                    rotation_sign = -1

                evasion_angle = attack_angle + rotation_sign * np.pi/2
                angle_current = shortest_angular_distance(state[5],
                                                          evasion_angle)

                alpha = normalize_angle(state[5] + rotation_sign*np.pi/2)

                cos_alpha = np.cos(alpha)
                sin_alpha = np.sin(alpha)
                increment = 1.5*self._rotation_radius * np.array(
                    [cos_alpha, sin_alpha])

                lateral_collision, _ = self._check_collision(state[3:5],
                                                             state[3:5] +
                                                             increment)

                if lateral_collision is not None:

                    rotation_sign *= -1
                    angle_current = rotation_sign * np.pi

        omega_prey = angle_current / np.pi

        u_prey = np.empty(2)
        u_prey[0] = self._bound(velocity_prey, 0, self._v_prey)
        u_prey[1] = self._bound(omega_prey, -self._omega_prey, self._omega_prey)

        return u_prey

    @staticmethod
    def _cross_2d(vecr, vecs):
        return vecr[0] * vecs[1] - vecr[1] * vecs[0]

    @staticmethod
    def _segment_to_vector(*segment):
        a = segment[0]
        b = segment[1]

        if (b[0] > a[0]) or (b[0] == a[0] and b[1] > a[1]):
            return b - a
        else:
            return a - b

    @staticmethod
    def _vector_angle(x, y):
        angle = np.arctan2(x[1], x[0]) - np.arctan2(y[1], y[0])

        return normalize_angle(angle)

    def _check_collision(self, start, end):
        collision = None
        index = None

        min_u = np.inf

        for i, obstacle in enumerate(self._obstacles):
            r = obstacle[1] - obstacle[0]
            s = end - start
            den = self._cross_2d(vecr=r, vecs=s)

            if den != 0:
                t = self._cross_2d((start - obstacle[0]), s) / den
                u = self._cross_2d((start - obstacle[0]), r) / den

                if 1 >= u >= 0 and 1 >= t >= 0:

                    if u < min_u:
                        collision = start + (u-1e-2)*s
                        min_u = u
                        index = i

        return collision, index

    def _differential_drive_dynamics(self, state, u):
        delta = np.empty(3)

        delta[0] = np.cos(state[2]) * u[0]
        delta[1] = np.sin(state[2]) * u[0]
        delta[2] = u[1]

        new_state = state + delta

        collision, _ = self._check_collision(state[:2], new_state[:2])

        if collision is not None:
            new_state[:2] = collision

        new_state[0] = self._bound(new_state[0], -self._max_x, self._max_x)
        new_state[1] = self._bound(new_state[1], -self._max_y, self._max_y)
        new_state[2] = normalize_angle(new_state[2])

        return new_state

    def render(self, mode='human'):
        center = np.array([self._max_x, self._max_y])

        predator_pos = self._state[:2]
        predator_theta = self._state[2]

        prey_pos = self._state[3:5]
        prey_theta = self._state[5]


        # Predator
        self._viewer.circle(center + predator_pos, self._catch_radius,
                            (255, 255, 255))
        self._viewer.arrow_head(center + predator_pos, self._catch_radius,
                                predator_theta, (255, 0, 0))

        # Prey
        self._viewer.arrow_head(center + prey_pos, self._catch_radius,
                                prey_theta, (0, 0, 255))

        # Obstacles
        for obstacle in self._obstacles:
            start = obstacle[0]
            end = obstacle[1]
            self._viewer.line(center + start, center + end)

        self._viewer.display(self._dt)



