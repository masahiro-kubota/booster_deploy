from booster_deploy.utils.registry import register_task
from booster_deploy.utils.isaaclab.configclass import configclass
from .beyond_mimic import K1BeyondMimicControllerCfg
from .diffusion_joystick import K1BeyondMimicDiffusionJoystickControllerCfg


@configclass
class K1MJ2ControllerCfg(K1BeyondMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "motions/k1_mj2_seg1.npz"
        self.policy.checkpoint_path = "models/k1_mj_dance_002_2025-12-03_00-10-28.pt"
        self.robot.joint_stiffness = [
            10.0, 10.0,
            4., 4., 4., 4.,
            4., 4., 4., 4.,
            80., 80., 80., 80., 30., 30.,
            80., 80., 80., 80., 30., 30.,
        ]
        self.robot.joint_damping = [
            2., 2.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.
        ]
        self.robot.effort_limit = [
            6, 6,
            14, 14, 14, 14,
            14, 14, 14, 14,
            30, 35, 20, 40, 20, 20,
            30, 35, 20, 40, 20, 20,
        ]


@configclass
class K1FightControllerCfg(K1BeyondMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "motions/k1_fight_final_deploy.npz"
        self.policy.checkpoint_path = "models/k1_fight_001_model_11000.pt"
        self.robot.joint_stiffness = [
            10.0, 10.0,
            3.95, 3.95, 3.95, 3.95,
            3.95, 3.95, 3.95, 3.95,
            80., 80., 80., 80., 30., 30.,
            80., 80., 80., 80., 30., 30.,
        ]
        self.robot.joint_damping = [
            2., 2.,
            0.3, 0.3, 0.3, 0.3,
            0.3, 0.3, 0.3, 0.3,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.
        ]
        self.robot.effort_limit = [
            4, 4,
            12, 12, 12, 12,
            12, 12, 12, 12,
            30, 35, 20, 40, 20, 20,
            30, 35, 20, 40, 20, 20,
        ]


@configclass
class K1FightOfficialControllerCfg(K1FightControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/k1_fight_001.pt"


@configclass
class K1FightCurrentTrainControllerCfg(K1BeyondMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "motions/k1_fight_final_deploy.npz"
        self.policy.checkpoint_path = "models/k1_fight_001_model_11000.pt"
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


@configclass
class K1KimodoOutputControllerCfg(K1FightCurrentTrainControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "motions/k1_kimodo_output.npz"
        self.policy.checkpoint_path = "models/k1_kimodo_output_model_22000.pt"
        self.mujoco.init_joint_pos = [
            0.06901567, 0.20200270,
            -0.03479115, -1.29708230, 0.76652944, -0.41176012,
            0.10545869, 1.25796783, 0.51180345, 0.48174441,
            0.00769035, 0.06449573, 0.13956706, 0.03814247, 0.20349492, -0.14522916,
            -0.00105525, -0.03136613, 0.13933265, 0.02891732, 0.14312220, 0.20138903,
        ]
        self.mujoco.init_pos = [0.0, 0.0, 0.54955659]
        self.mujoco.init_quat = [0.998560, 0.000096, -0.053641, 0.000005]


@configclass
class K1PikachuIdleControllerCfg(K1FightCurrentTrainControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "motions/k1_pikachu_idle_pingpong_tile3_blender50_grounded_hold0p5.npz"
        self.policy.checkpoint_path = "models/k1_pikachu_idle_model_6000.pt"
        self.policy.loop_motion = True
        self.mujoco.init_joint_pos = [
            -0.22235082, -0.33899999,
            -1.10408545, -1.74000001, -2.26999998, -0.96159136,
            -0.56962961, 1.13585114, -2.24332809, 0.46619520,
            -0.25673947, 0.22823051, 0.27210751, 0.06455009,
            0.11622748, -0.15281184,
            -0.04732056, -0.46903962, -0.49038759, 0.18649599,
            0.27503318, 0.34299999,
        ]
        self.mujoco.init_pos = [0.0, 0.0, 0.56711070]
        self.mujoco.init_quat = [0.99647439, -0.00771415, 0.08354028, 0.00064671]


register_task("k1_mj2", K1MJ2ControllerCfg())
register_task("k1_fight", K1FightControllerCfg())
register_task("k1_fight_official", K1FightOfficialControllerCfg())
register_task("k1_fight_current_train", K1FightCurrentTrainControllerCfg())
register_task("k1_kimodo_output", K1KimodoOutputControllerCfg())
register_task("k1_pikachu_idle", K1PikachuIdleControllerCfg())
register_task("k1_beyondmimic_joystick", K1BeyondMimicDiffusionJoystickControllerCfg())
