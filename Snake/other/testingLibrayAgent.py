import gym
from gym.envs.registration import register
from testing.gymSnakeGameEnv import SnakeGameEnv
from snakeAI import build_model, build_agent


import tensorflow as tf

from keras import __version__
tf.keras.__version__ = __version__

from tensorflow.keras.optimizers.legacy import Adam
Adam._name = 'hey'

def main():
    register(
        id='SnakeGame-v0',
        entry_point='snakeGameEnv:SnakeGameEnv',  # Replace 'your_module' and 'SnakeGameEnv' with your module and class names
        max_episode_steps=100,  # Adjust max episode steps if needed
    )

    env = gym.make('SnakeGame-v0', render_mode = "human")

    print(env.observation_space.shape)
    input_shape = (1,) + env.observation_space.shape
    model = build_model(input_shape, 4)
    dqn = build_agent(model, 4)
    dqn.compile(Adam(lr = 1e-4, clipnorm = 1.0))
    dqn.fit(env, nb_steps=2, visualize=False, verbose=1)

    print("fitting Done!")
    _ = dqn.test(env, nb_episodes=5, visualize=True)



    env.close()



if __name__ == "__main__":
    main()