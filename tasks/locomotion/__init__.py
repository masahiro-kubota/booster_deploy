from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .locomotion import (
    K1WalkControllerCfg,
    T1WalkControllerCfg
)

# Register locomotion tasks


@configclass
class T1WalkControllerCfg1(T1WalkControllerCfg):
    '''Human-like walk for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/t1_walk.pt"


@configclass
class K1WalkControllerCfg1(K1WalkControllerCfg):
    """Velocity-commanded walk for K1 using the current training actuator profile."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/k1_locomotion.pt"
        self.robot.joint_stiffness = [
            3.9478417602100686, 3.9478417602100686,
            3.9478417602100686, 3.9478417602100686, 3.9478417602100686, 3.9478417602100686,
            3.9478417602100686, 3.9478417602100686, 3.9478417602100686, 3.9478417602100686,
            30.200989465607023, 21.447961045805584, 17.846013389258083,
            60.401978931214046, 35.692026778516166, 35.692026778516166,
            30.200989465607023, 21.447961045805584, 17.846013389258083,
            60.401978931214046, 35.692026778516166, 35.692026778516166,
        ]
        self.robot.joint_damping = [
            0.25132741228, 0.25132741228,
            0.25132741228, 0.25132741228, 0.25132741228, 0.25132741228,
            0.25132741228, 0.25132741228, 0.25132741228, 0.25132741228,
            3.60497756989125, 2.560161764834957, 2.1302109340993156,
            4.806636759855, 4.260421868198631, 4.260421868198631,
            3.60497756989125, 2.560161764834957, 2.1302109340993156,
            4.806636759855, 4.260421868198631, 4.260421868198631,
        ]
        self.robot.effort_limit = [
            6.0, 6.0,
            14.0, 14.0, 14.0, 14.0,
            14.0, 14.0, 14.0, 14.0,
            68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
            68.0, 76.0, 38.3, 112.0, 38.3, 38.3,
        ]


register_task("k1_walk", K1WalkControllerCfg1())
register_task(
    "t1_walk", T1WalkControllerCfg1())
