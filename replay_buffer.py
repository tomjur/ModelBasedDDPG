from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, config):
        self.buffer_size = config['model']['buffer_size']
        self.count = 0
        self.buffer = deque()

    def add(self, goal_pose, goal_joints, workspace_id, current_state, action, reward, terminated, next_state):
        experience = (goal_pose, goal_joints, workspace_id, current_state, action, reward, terminated, next_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        count = min([batch_size, self.count])
        batch = random.sample(self.buffer, count)
        return zip(*batch)

    # def clear(self):
    #     self.buffer.clear()
    #     self.count = 0


