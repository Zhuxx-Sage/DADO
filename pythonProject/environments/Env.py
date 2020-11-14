class Env(object):
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    action_space = None
    observation_space = None

    def step(self, acion):
        '''
        :param acion: excute the action
        :return: next_state，reward, done, info
        '''
        raise NotImplementedError

    def reset(self):
        '''
        :return: reset the environment
        '''
        raise NotImplementedError
