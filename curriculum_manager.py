class CurriculumManager:

    def __init__(self, config, print_messages):
        self.config = config
        self.print_messages = print_messages
        self.enable = config['curriculum']['enable']
        if not self.enable:
            if self.print_messages:
                print 'curriculum disabled'
            return
        self.current_length = config['curriculum']['initial_length']
        self.length_increments = config['curriculum']['length_increments']
        self.success_rate_increase = config['curriculum']['success_rate_increase']
        self.minimal_episodes = config['curriculum']['minimal_episodes']
        self._print_message(0.0, self.current_length, 0.0)

    def get_next_parameters(self, episodes, successful_episodes):
        if not self.enable:
            # curriculum disabled
            return None, False
        changed = False
        if episodes >= self.minimal_episodes:
            success_rate = float(successful_episodes) / episodes
            if success_rate > self.success_rate_increase:
                new_length = self.current_length + self.length_increments
                self._print_message(self.current_length, new_length, success_rate)
                self.current_length = new_length
                changed = True
        return self.current_length, changed

    def _print_message(self, old_length, new_length, success_rate):
        if not self.print_messages:
            return
        print 'curriculum: success rate was {} greater than {} increasing length'.format(
            success_rate, self.success_rate_increase)
        print 'curriculum: old length was {} new length is {}'.format(old_length, new_length)