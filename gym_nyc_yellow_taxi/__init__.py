from gym.envs.registration import register

register(
    id='nyc-yellow-taxi-v0',
    entry_point='gym_nyc_yellow_taxi.envs:BasicEnv',
)

