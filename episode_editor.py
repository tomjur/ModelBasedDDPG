import numpy as np


class EpisodeEditor:
    def __init__(self, alter_episode_mode, pre_trained_reward, image_cache, joints_dimension=4, pose_dimension=2,
                 status_dimension=3, image_dimension=(55, 111)):
        self.alter_episode_mode = alter_episode_mode
        self.pre_trained_reward = pre_trained_reward
        self.image_cache = image_cache
        self.joints_dimension = joints_dimension
        self.pose_dimension = pose_dimension
        self.status_dimension = status_dimension
        self.image_dimension = image_dimension

        self.current_joints_buffer = None
        self.goal_joints_buffer = None
        self.actions_buffer = None
        self.goal_poses_buffer = None
        self.status_buffer = None
        self.images_buffer = None

    def _clear_buffers(self):
        self.current_joints_buffer = np.zeros((0, self.joints_dimension), dtype=np.float32)
        self.goal_joints_buffer = np.zeros((0, self.joints_dimension), dtype=np.float32)
        self.actions_buffer = np.zeros((0, self.joints_dimension), dtype=np.float32)
        self.goal_poses_buffer = np.zeros((0, self.pose_dimension), dtype=np.float32)
        if self.alter_episode_mode == 2:
            self.status_buffer = np.zeros((0, self.status_dimension), dtype=np.float32)
        if self.image_cache is not None:
            self.images_buffer = np.zeros((0, self.image_dimension[0], self.image_dimension[1]), dtype=np.int32)

    def _append_to_buffers(self, current_joints, goal_joints, actions, goal_poses, status, images):
        self.current_joints_buffer = np.append(self.current_joints_buffer, current_joints, axis=0)
        self.goal_joints_buffer = np.append(self.goal_joints_buffer, goal_joints, axis=0)
        self.actions_buffer = np.append(self.actions_buffer, actions, axis=0)
        self.goal_poses_buffer = np.append(self.goal_poses_buffer, goal_poses, axis=0)
        if self.status_buffer is not None:
            self.status_buffer = np.append(self.status_buffer, status, axis=0)
        if self.images_buffer is not None:
            self.images_buffer = np.append(self.images_buffer, images, axis=0)

    def process_episodes(self, episodes, sess):
        # no alteration
        if self.alter_episode_mode == 0:
            return episodes
        assert self.pre_trained_reward is not None
        # clear reward network input buffers
        self._clear_buffers()
        episode_start_indices = []
        for episode_agent_trajectory in episodes:
            # save the start index for every episode
            episode_start_indices.append(len(self.current_joints_buffer))
            # add data to buffers
            status, states, actions, rewards, goal_pose, goal_joints, workspace_id = episode_agent_trajectory
            current_joints = [state[0] for state in states[:-1]]
            one_hot_status = None
            if self.alter_episode_mode == 2:
                one_hot_status = np.zeros((len(rewards), 3), dtype=np.float32)
                one_hot_status[:-1, 0] = 1.0
                one_hot_status[-1, 2] = 1.0
            images = None
            if self.images_buffer is not None:
                image = self.image_cache.get_image(workspace_id)
                images = [image] * len(actions)
            self._append_to_buffers(current_joints, [goal_joints] * len(actions), actions, [goal_pose] * len(actions),
                                    one_hot_status, images)
        # predict for all the episodes in the same time
        fake_rewards, fake_status_prob = self.pre_trained_reward.make_prediction(
            sess, self.current_joints_buffer, self.goal_joints_buffer, self.actions_buffer, self.goal_poses_buffer,
            self.status_buffer
        )
        # partition the results by episode
        resulting_episodes = []
        for episode_start_index, episode_agent_trajectory in zip(episode_start_indices, episodes):
            status, states, actions, rewards, goal_pose, goal_joints, workspace_id = episode_agent_trajectory
            # get the relevant rewards
            relevant_rewards = fake_rewards[episode_start_index: episode_start_index+len(rewards)]
            if self.alter_episode_mode == 2:
                altered_result = status, states, actions, relevant_rewards, goal_pose, goal_joints, workspace_id
            elif self.alter_episode_mode == 1:
                relevant_fake_status = fake_status_prob[episode_start_index: episode_start_index+len(rewards)]
                fake_status = np.argmax(np.array(relevant_fake_status), axis=1)
                fake_status += 1
                # iterate over approximated episode and see if truncation is needed
                truncation_index = 0
                for truncation_index in range(len(fake_status)):
                    if fake_status[truncation_index] != 1:
                        break
                # return the status of the last transition, truncated list of states and actions, the fake rewards (also
                # truncated) and the goal parameters as-is.
                altered_status = fake_status[truncation_index]
                altered_states = states[:truncation_index + 2]
                altered_actions = actions[:truncation_index + 1]
                altered_rewards = [r[0] for r in relevant_rewards[:truncation_index + 1]]
                altered_result = altered_status, altered_states, altered_actions, altered_rewards, goal_pose, \
                                 goal_joints, workspace_id
            else:
                assert False
            resulting_episodes.append(altered_result)
        return resulting_episodes
