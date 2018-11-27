import numpy as np
from potential_point import PotentialPoint


class HindsightPolicy:
    def __init__(self, config, replay_buffer, predict_reward_and_status_func):
        self.config = config
        self.replay_buffer = replay_buffer
        self.target_potential_point = PotentialPoint.from_config(config)[-1]
        self.predict_reward_and_status_func = predict_reward_and_status_func
        # the following buffer saves the transition we are about to add
        self.augmented_buffer = []

    def append_to_replay_buffer(self, status, states, actions, rewards, goal_pose, goal_joints, workspace_image):
        self.augmented_buffer = []
        for i in range(len(actions)):
            current_state = states[i]
            next_state = states[i+1]
            action_used = actions[i]
            current_reward = rewards[i]
            # only the last state is a terminal state
            is_terminal = i == len(actions) - 1 and status != 1
            self.replay_buffer.add(
                goal_pose, goal_joints, workspace_image, current_state, action_used, current_reward, is_terminal,
                next_state
            )
            self._add_extra_data(i, status, states, actions, rewards, workspace_image)
        self._score_extra_data_and_add_to_buffer()

    def _score_extra_data_and_add_to_buffer(self):
        if len(self.augmented_buffer) == 0:
            return
        if self.config['hindsight']['score_with_reward_model']:
            rewards = self.predict_reward_and_status_func(self.augmented_buffer)
            for i, transition in enumerate(self.augmented_buffer):
                goal_pose, goal_joints, workspace_image, current_state, action_used, current_reward, is_terminal, \
                next_state = transition
                self.replay_buffer.add(
                    goal_pose, goal_joints, workspace_image, current_state, action_used, rewards[i], is_terminal,
                    next_state
                )
        else:
            for transition in self.augmented_buffer:
                self.replay_buffer.add(*zip(transition))

    def _add_extra_data(self, current_state_index, status, states, actions, rewards, workspace_image):
        if not self.config['hindsight']['enable']:
            return
        if self.config['hindsight']['type'] == 'goal':
            self._execute_goal_policy(current_state_index, status, states, actions, rewards, workspace_image)
        elif self.config['hindsight']['type'] == 'future':
            self._execute_future_policy(current_state_index, status, states, actions, rewards, workspace_image)
        else:
            assert False

    def _execute_goal_policy(self, current_state_index, status, states, actions, rewards, workspace_image):
        # if the last state is already close to the goal, don't need to include a similar state
        if status == 3:
            return
        # if the trajectory ended free, the goal is the last state
        if status == 1:
            if len(states) > 1:
                self._add_goal_at_index(current_state_index, len(states)-1, states, actions, rewards, workspace_image)
        # if the trajectory ended in collision, the goal is the before last state
        elif status == 2:
            if len(states) > 2:
                self._add_goal_at_index(current_state_index, len(states) - 2, states, actions, rewards, workspace_image)

    def _execute_future_policy(self, current_state_index, status, states, actions, rewards, workspace_image):
        # the last possible index depends if the trajectory ended in collision
        last_index = len(states) if status != 2 else len(states)-1
        times = self.config['hindsight']['k']
        candidates = list(range(current_state_index+1, last_index))
        if len(candidates) > times:
            goal_indices = np.random.choice(candidates, times, replace=False)
        else:
            goal_indices = candidates
        for goal_state_index in goal_indices:
            self._add_goal_at_index(current_state_index, goal_state_index, states, actions, rewards, workspace_image)

    def _add_goal_at_index(self, current_state_index, goal_state_index, states, actions, rewards, workspace_image):
        if current_state_index >= goal_state_index:
            return
        goal_state = states[goal_state_index]
        goal_joints = goal_state[0]
        goal_pose = goal_state[1][self.target_potential_point.tuple]
        current_state = states[current_state_index]
        action_used = actions[current_state_index]
        next_state = states[current_state_index + 1]
        current_reward = 1.0 if current_state_index + 1 == goal_state_index else rewards[current_state_index]
        is_terminal = True if current_state_index + 1 == goal_state_index else False
        transition = goal_pose, goal_joints, workspace_image, current_state, action_used, current_reward, is_terminal,\
                     next_state
        self.augmented_buffer.append(transition)
