import numpy as np
import gymnasium
import json
import struct
import os
import hashlib
import pufferlib
from pufferlib.ocean.drive import binding
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

_POLICY_TYPE_PADDED = 0
_EMPTY_PARTNER_EPS = 1e-8
_EGO_TRAILER_STATE_FEATURES = 4
_DYNAMICS_MODEL_IDS = {"classic": 0, "jerk": 1, "articulated": 2}
_NON_KINEMATIC_PARAM_ORDER = [
    "tractor_length",
    "trailer_length",
    "width",
    "trailer_width",
    "vehicle_height",
    "trailer_height",
    "tractor2hitch",
    "trailer2hitch",
    "tractor_d_rear_axle2rear_bumper",
    "tractor_d_rear_axle2front_axle",
    "tractor_d_front_axle2front_bumper",
    "trailer_d_rear_axel2_rear_bumper",
    "trailer_d_real_axel2_front_bumper",
]
_NON_KINEMATIC_PARAM_CACHE = {}
DEFAULT_SDC_RUNTIME_TRUCK_REF_BIN = (
    "tests/artifacts/drive/traversing_traffic_light_intersection__97be27351e915863__97be27351e915863.bin"
)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _load_non_kinematic_vehicle_params_from_bin(binary_path):
    cached = _NON_KINEMATIC_PARAM_CACHE.get(binary_path)
    if cached is not None:
        return cached

    with open(binary_path, "rb") as f:
        _ = struct.unpack("<i", f.read(4))[0]
        num_tracks_to_predict = struct.unpack("<i", f.read(4))[0]
        f.seek(4 * num_tracks_to_predict, os.SEEK_CUR)
        num_objects = struct.unpack("<i", f.read(4))[0]
        num_roads = struct.unpack("<i", f.read(4))[0]

        for _ in range(num_objects):
            _ = struct.unpack("<i", f.read(4))[0]
            _ = struct.unpack("<i", f.read(4))[0]
            _ = struct.unpack("<i", f.read(4))[0]
            trajectory_length = struct.unpack("<i", f.read(4))[0]
            f.seek(4 * trajectory_length * 3, os.SEEK_CUR)
            f.seek((4 * trajectory_length * 4) + (4 * trajectory_length), os.SEEK_CUR)
            f.seek((6 * 4) + 4, os.SEEK_CUR)

        for _ in range(num_roads):
            _ = struct.unpack("<i", f.read(4))[0]
            _ = struct.unpack("<i", f.read(4))[0]
            _ = struct.unpack("<i", f.read(4))[0]
            array_size = struct.unpack("<i", f.read(4))[0]
            f.seek(4 * array_size * 3, os.SEEK_CUR)
            f.seek((6 * 4) + 4, os.SEEK_CUR)

        extension_magic = struct.unpack("<i", f.read(4))[0]
        extension_version = struct.unpack("<i", f.read(4))[0]
        if extension_magic != 0x54524C52:
            raise ValueError(f"Extension magic mismatch in {binary_path}")
        if extension_version != 2:
            raise ValueError(f"Unsupported extension version {extension_version} in {binary_path}")

        f.seek(4 * 2, os.SEEK_CUR)
        object_meta_count = struct.unpack("<i", f.read(4))[0]
        f.seek(object_meta_count * (8 + 4 + 4), os.SEEK_CUR)

        vehicle_param_count = struct.unpack("<i", f.read(4))[0]
        if vehicle_param_count != len(_NON_KINEMATIC_PARAM_ORDER):
            raise ValueError(
                f"Expected {len(_NON_KINEMATIC_PARAM_ORDER)} non-kinematic params in {binary_path}, got {vehicle_param_count}"
            )
        values = struct.unpack(f"<{vehicle_param_count}f", f.read(4 * vehicle_param_count))

    params = tuple(float(v) for v in values)
    _NON_KINEMATIC_PARAM_CACHE[binary_path] = params
    return params


class Drive(pufferlib.PufferEnv):
    def __init__(
        self,
        render_mode=None,
        report_interval=1,
        width=1280,
        height=1024,
        human_agent_idx=0,
        reward_vehicle_collision=-0.1,
        reward_offroad_collision=-0.1,
        reward_goal=1.0,
        reward_goal_post_respawn=0.5,
        goal_behavior=0,
        goal_target_distance=10.0,
        goal_radius=2.0,
        goal_speed=20.0,
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        episode_length=None,
        termination_mode=None,
        resample_frequency=91,
        num_maps=100,
        num_agents=512,
        action_type="discrete",
        dynamics_model="classic",
        max_controlled_agents=-1,
        buf=None,
        seed=1,
        init_steps=0,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        observation_mode="default",
        map_dir="resources/drive/binaries/training",
        sequential_map_sampling=False,
        sdc_runtime_truck_override=False,
        sdc_runtime_truck_ref_bin=None,
        force_truck_params_from_ref_bin=None,
        force_zero_trailer_articulation_at_init=False,
    ):
        # env
        self.dt = dt
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.goal_radius = goal_radius
        self.goal_speed = goal_speed
        self.goal_behavior = goal_behavior
        self.goal_target_distance = goal_target_distance
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.human_agent_idx = human_agent_idx
        self.episode_length = episode_length
        self.termination_mode = termination_mode
        self.resample_frequency = resample_frequency
        self.dynamics_model = dynamics_model
        self.type_classes = binding.POLICY_TYPE_CLASS_COUNT

        # Observation space calculation
        self._base_ego_features = {"classic": binding.EGO_FEATURES_CLASSIC, "articulated": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}.get(
            dynamics_model
        )

        # Extract observation shapes from constants
        # These need to be defined in C, since they determine the shape of the arrays
        self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        self.max_partner_objects = binding.MAX_AGENTS - 1
        self._base_partner_features = binding.PARTNER_FEATURES
        self.road_features = binding.ROAD_FEATURES

        self._sim_num_obs = (
            self._base_ego_features
            + self.max_partner_objects * self._base_partner_features
            + self.max_road_objects * self.road_features
        )

        self.init_steps = init_steps
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode
        self.observation_mode_str = observation_mode
        self.map_dir = map_dir
        self.force_zero_trailer_articulation_at_init = _as_bool(force_zero_trailer_articulation_at_init)
        self.non_kinematic_vehicle_params_override = None
        if isinstance(sdc_runtime_truck_ref_bin, str):
            stripped_ref = sdc_runtime_truck_ref_bin.strip()
            if stripped_ref == "" or stripped_ref.lower() == "none":
                sdc_runtime_truck_ref_bin = None
        if _as_bool(sdc_runtime_truck_override):
            self.force_zero_trailer_articulation_at_init = True
            if force_truck_params_from_ref_bin is None:
                force_truck_params_from_ref_bin = (
                    sdc_runtime_truck_ref_bin
                    if sdc_runtime_truck_ref_bin is not None
                    else DEFAULT_SDC_RUNTIME_TRUCK_REF_BIN
                )
        if force_truck_params_from_ref_bin is not None:
            reference_bin = os.path.abspath(force_truck_params_from_ref_bin)
            if not os.path.exists(reference_bin):
                raise FileNotFoundError(f"Truck reference artifact not found: {reference_bin}")
            self.non_kinematic_vehicle_params_override = _load_non_kinematic_vehicle_params_from_bin(reference_bin)

        if self.control_mode_str == "control_vehicles":
            self.control_mode = 0
        elif self.control_mode_str == "control_agents":
            self.control_mode = 1
        elif self.control_mode_str == "control_wosac":
            self.control_mode = 2
        elif self.control_mode_str == "control_sdc_only":
            self.control_mode = 3
        else:
            raise ValueError(
                f"control_mode must be one of 'control_vehicles', 'control_wosac', or 'control_agents'. Got: {self.control_mode_str}"
            )
        if self.observation_mode_str == "default":
            self.observation_mode = 0
        elif self.observation_mode_str == "sdc_only_with_trailer":
            self.observation_mode = 1
        else:
            raise ValueError(
                "observation_mode must be one of 'default' or 'sdc_only_with_trailer'. "
                f"Got: {self.observation_mode_str}"
            )
        if self.observation_mode == 0:
            self.ego_features = self._base_ego_features
            self.partner_features = self._base_partner_features
        else:
            self.ego_features = self._base_ego_features + 1 + _EGO_TRAILER_STATE_FEATURES
            self.partner_features = self._base_partner_features + 1
        self.num_obs = (
            self.ego_features
            + self.max_partner_objects * self.partner_features
            + self.max_road_objects * self.road_features
        )
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)
        if self.init_mode_str == "create_all_valid":
            self.init_mode = 0
        elif self.init_mode_str == "create_only_controlled":
            self.init_mode = 1
        else:
            raise ValueError(
                f"init_mode must be one of 'create_all_valid' or 'create_only_controlled'. Got: {self.init_mode_str}"
            )

        if action_type == "discrete":
            if dynamics_model in ("classic", "articulated"):
                # Joint action space (assume dependence)
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
                # Multi discrete (assume independence)
                # self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
            elif dynamics_model == "jerk":
                # Joint action space (assume dependence) - 4 longitudinal × 3 lateral = 12
                self.single_action_space = gymnasium.spaces.MultiDiscrete([4 * 3])
            else:
                raise ValueError(f"dynamics_model must be 'classic', 'articulated' or 'jerk'. Got: {dynamics_model}")
        elif action_type == "continuous":
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"action_space must be 'discrete' or 'continuous'. Got: {action_type}")

        self._action_type_flag = 0 if action_type == "discrete" else 1

        # Check if resources directory exists
        binary_path = f"{map_dir}/map_000.bin"
        if not os.path.exists(binary_path):
            raise FileNotFoundError(
                f"Required directory {binary_path} not found. Please ensure the Drive maps are downloaded and installed correctly per docs."
            )

        # Check maps availability
        available_maps = len([name for name in os.listdir(map_dir) if name.endswith(".bin")])
        if num_maps > available_maps:
            raise ValueError(
                f"num_maps ({num_maps}) exceeds available maps in directory ({available_maps}). Please reduce num_maps or add more maps to resources/drive/binaries."
            )
        self.max_controlled_agents = int(max_controlled_agents)

        # Iterate through all maps to count total agents that can be initialized for each map
        agent_offsets, map_ids, num_envs = binding.shared(
            map_dir=map_dir,
            num_agents=num_agents,
            num_maps=num_maps,
            dynamics_model=_DYNAMICS_MODEL_IDS[dynamics_model],
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            goal_target_distance=self.goal_target_distance,
            sequential_map_sampling=sequential_map_sampling,
            non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
            force_zero_trailer_articulation_at_init=int(self.force_zero_trailer_articulation_at_init),
        )

        # agent_offsets[-1] works in both cases, just making it explicit that num_agents is ignored if sequential_map_sampling is True
        self.num_agents = num_agents if not sequential_map_sampling else agent_offsets[-1]
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        super().__init__(buf=buf)
        self._sim_observations = self.observations
        if self.observation_mode == 1:
            self._sim_observations = np.zeros((self.num_agents, self._sim_num_obs), dtype=np.float32)
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            env_id = binding.env_init(
                self._sim_observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=human_agent_idx,
                reward_vehicle_collision=reward_vehicle_collision,
                reward_offroad_collision=reward_offroad_collision,
                reward_goal=reward_goal,
                reward_goal_post_respawn=reward_goal_post_respawn,
                goal_radius=goal_radius,
                goal_speed=goal_speed,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=dt,
                dynamics_model=_DYNAMICS_MODEL_IDS[dynamics_model],
                episode_length=(int(episode_length) if episode_length is not None else None),
                termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
                max_controlled_agents=self.max_controlled_agents,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file="pufferlib/config/ocean/drive.ini",
                init_steps=init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=map_dir,
                non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
                force_zero_trailer_articulation_at_init=int(self.force_zero_trailer_articulation_at_init),
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

    def _resample_vector_envs(self, seed):
        binding.vec_close(self.c_envs)
        agent_offsets, map_ids, num_envs = binding.shared(
            num_agents=self.num_agents,
            num_maps=self.num_maps,
            dynamics_model=_DYNAMICS_MODEL_IDS[self.dynamics_model],
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            goal_target_distance=self.goal_target_distance,
            goal_speed=self.goal_speed,
            map_dir=self.map_dir,
            sequential_map_sampling=False,
            non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
            force_zero_trailer_articulation_at_init=int(self.force_zero_trailer_articulation_at_init),
        )
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            env_id = binding.env_init(
                self._sim_observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=self.human_agent_idx,
                reward_vehicle_collision=self.reward_vehicle_collision,
                reward_offroad_collision=self.reward_offroad_collision,
                reward_goal=self.reward_goal,
                reward_goal_post_respawn=self.reward_goal_post_respawn,
                goal_radius=self.goal_radius,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                goal_speed=self.goal_speed,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=self.dt,
                dynamics_model=_DYNAMICS_MODEL_IDS[self.dynamics_model],
                episode_length=(int(self.episode_length) if self.episode_length is not None else None),
                max_controlled_agents=self.max_controlled_agents,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file="pufferlib/config/ocean/drive.ini",
                init_steps=self.init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=self.map_dir,
                non_kinematic_vehicle_params_override=self.non_kinematic_vehicle_params_override,
                force_zero_trailer_articulation_at_init=int(self.force_zero_trailer_articulation_at_init),
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)
        binding.vec_reset(self.c_envs, seed)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        max_resample_attempts = 8
        attempts = 0
        while binding.vec_has_invalid_initial_trailer_state(self.c_envs):
            attempts += 1
            if attempts > max_resample_attempts:
                raise RuntimeError(
                    f"Exceeded {max_resample_attempts} resample attempts while rejecting invalid initial trailer states."
                )
            self._resample_vector_envs(np.random.randint(0, 2**32 - 1))
        self.tick = 0
        self._postprocess_observations()
        return self.observations, []

    def step(self, actions):
        self.terminals[:] = 0
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, self.num_agents)
            if log:
                info.append(log)
                # print(log)
        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.tick = 0
            seed = np.random.randint(0, 2**32 - 1)
            self._resample_vector_envs(seed)
            self.terminals[:] = 1
        self._postprocess_observations()
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def get_global_agent_state(self, include_sdc_trailer=False, include_types=False):
        """Get current global state of all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'id', 'length', 'width' containing numpy arrays
            of shape (num_active_agents,)
        """
        num_agents = self.num_agents

        states = {
            "x": np.zeros(num_agents, dtype=np.float32),
            "y": np.zeros(num_agents, dtype=np.float32),
            "z": np.zeros(num_agents, dtype=np.float32),
            "heading": np.zeros(num_agents, dtype=np.float32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "length": np.zeros(num_agents, dtype=np.float32),
            "width": np.zeros(num_agents, dtype=np.float32),
        }

        binding.vec_get_global_agent_state(
            self.c_envs,
            states["x"],
            states["y"],
            states["z"],
            states["heading"],
            states["id"],
            states["length"],
            states["width"],
        )

        if include_sdc_trailer:
            states["sdc_trailer"] = self.get_sdc_trailer_state()
        if include_types:
            states["type"] = self.get_global_agent_types()

        return states

    def get_global_agent_types(self):
        """Get current type id for all active agents."""
        types = np.zeros(self.num_agents, dtype=np.int32)
        binding.vec_get_global_agent_types(self.c_envs, types)
        return types

    def get_partner_types(self):
        """Get partner type ids in the same ordering as partner observations."""
        types = np.zeros((self.num_agents, self.max_partner_objects), dtype=np.int32)
        binding.vec_get_partner_types(self.c_envs, types)
        return types

    def get_sdc_trailer_state(self):
        trailer = {
            "has_trailer": np.zeros(self.num_envs, dtype=np.int32),
            "x": np.zeros(self.num_envs, dtype=np.float32),
            "y": np.zeros(self.num_envs, dtype=np.float32),
            "z": np.zeros(self.num_envs, dtype=np.float32),
            "heading": np.zeros(self.num_envs, dtype=np.float32),
            "id": np.zeros(self.num_envs, dtype=np.int32),
            "length": np.zeros(self.num_envs, dtype=np.float32),
            "width": np.zeros(self.num_envs, dtype=np.float32),
        }

        binding.vec_get_sdc_trailer_state(
            self.c_envs,
            trailer["has_trailer"],
            trailer["x"],
            trailer["y"],
            trailer["z"],
            trailer["heading"],
            trailer["id"],
            trailer["length"],
            trailer["width"],
        )
        return trailer

    def get_ego_trailer_obs_features(self):
        trailer_features = {
            "rel_x": np.zeros(self.num_agents, dtype=np.float32),
            "rel_y": np.zeros(self.num_agents, dtype=np.float32),
            "rel_heading_x": np.zeros(self.num_agents, dtype=np.float32),
            "rel_heading_y": np.zeros(self.num_agents, dtype=np.float32),
        }

        binding.vec_get_ego_trailer_obs_features(
            self.c_envs,
            trailer_features["rel_x"],
            trailer_features["rel_y"],
            trailer_features["rel_heading_x"],
            trailer_features["rel_heading_y"],
        )
        return trailer_features

    def _postprocess_observations(self):
        if getattr(self, "observation_mode", 0) != 1:
            return

        base_ego = self._base_ego_features
        base_partner = self._base_partner_features
        partner_count = self.max_partner_objects
        base_partner_dim = partner_count * base_partner
        base_road_start = base_ego + base_partner_dim
        road_dim = self.max_road_objects * self.road_features

        aug_ego = self.ego_features
        aug_partner = self.partner_features
        aug_partner_dim = partner_count * aug_partner
        aug_road_start = aug_ego + aug_partner_dim

        self.observations[:] = 0.0
        self.observations[:, :base_ego] = self._sim_observations[:, :base_ego]

        sim_partner = self._sim_observations[:, base_ego:base_road_start].reshape(
            self.num_agents, partner_count, base_partner
        )
        aug_partner_view = self.observations[:, aug_ego:aug_road_start].reshape(self.num_agents, partner_count, aug_partner)
        aug_partner_view[:, :, :base_partner] = sim_partner
        self.observations[:, aug_road_start : aug_road_start + road_dim] = self._sim_observations[
            :, base_road_start : base_road_start + road_dim
        ]

        policy_type_max = self.type_classes - 1
        ego_types = np.clip(self.get_global_agent_types(), _POLICY_TYPE_PADDED, policy_type_max).astype(np.float32)
        partner_types = np.clip(self.get_partner_types(), _POLICY_TYPE_PADDED, policy_type_max).astype(np.float32)
        ego_trailer_features = self.get_ego_trailer_obs_features()

        ego_type_idx = base_ego
        self.observations[:, ego_type_idx] = ego_types
        trailer_feature_start = ego_type_idx + 1
        self.observations[:, trailer_feature_start] = ego_trailer_features["rel_x"]
        self.observations[:, trailer_feature_start + 1] = ego_trailer_features["rel_y"]
        self.observations[:, trailer_feature_start + 2] = ego_trailer_features["rel_heading_x"]
        self.observations[:, trailer_feature_start + 3] = ego_trailer_features["rel_heading_y"]

        # Populate partner type channel only for occupied partner slots.
        occupied_partner_slots = np.any(np.abs(sim_partner) > _EMPTY_PARTNER_EPS, axis=2)
        aug_partner_view[:, :, base_partner] = np.where(
            occupied_partner_slots, partner_types, _POLICY_TYPE_PADDED
        ).astype(np.float32)

    def get_ground_truth_trajectories(self):
        """Get ground truth trajectories for all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'valid', 'id', 'scenario_id' containing numpy arrays.
        """
        num_agents = self.num_agents

        trajectories = {
            "x": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "valid": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.int32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "is_vehicle": np.zeros(num_agents, dtype=np.int32),
            "scenario_id": np.zeros(num_agents, dtype=np.int32),
        }

        binding.vec_get_global_ground_truth_trajectories(
            self.c_envs,
            trajectories["x"],
            trajectories["y"],
            trajectories["z"],
            trajectories["heading"],
            trajectories["valid"],
            trajectories["id"],
            trajectories["is_vehicle"],
            trajectories["scenario_id"],
        )

        for key in trajectories:
            trajectories[key] = trajectories[key][:, None]

        return trajectories

    def get_road_edge_polylines(self):
        """Get road edge polylines for all scenarios.

        Returns:
            dict with keys 'x', 'y', 'lengths', 'scenario_id' containing numpy arrays.
            x, y are flattened point coordinates; lengths indicates points per polyline.
        """
        num_polylines, total_points = binding.vec_get_road_edge_counts(self.c_envs)

        polylines = {
            "x": np.zeros(total_points, dtype=np.float32),
            "y": np.zeros(total_points, dtype=np.float32),
            "lengths": np.zeros(num_polylines, dtype=np.int32),
            "scenario_id": np.zeros(num_polylines, dtype=np.int32),
        }

        binding.vec_get_road_edge_polylines(
            self.c_envs,
            polylines["x"],
            polylines["y"],
            polylines["lengths"],
            polylines["scenario_id"],
        )

        return polylines

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy


def simplify_polyline(geometry, polyline_reduction_threshold, max_segment_length):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt, optimized for Python."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry  # Not enough points to simplify

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)
            if area < polyline_reduction_threshold and dist(point1, point3) <= max_segment_length:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def save_map_binary(map_data, output_file, unique_map_id):
    trajectory_length = 91
    """Saves map data in a binary format readable by C"""
    with open(output_file, "wb") as f:
        extension_magic = 0x54524C52
        extension_version = 2

        def stable_track_hash(value):
            if value is None:
                return 0
            digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
            return struct.unpack("<Q", digest)[0]

        def normalize_int32_id(value):
            int32_min = -(2**31)
            int32_max = 2**31 - 1

            if isinstance(value, bool):
                value = int(value)

            try:
                intval = int(value)
                if int32_min <= intval <= int32_max:
                    return intval
                return ((intval + 2**31) % 2**32) - 2**31
            except (TypeError, ValueError):
                digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest()
                uint32 = struct.unpack("<I", digest)[0]
                return uint32 - 2**32 if uint32 >= 2**31 else uint32

        # Get metadata
        metadata = map_data.get("metadata", {})
        sdc_track_index = metadata.get("sdc_track_index", -1)  # -1 as default if not found
        tracks_to_predict = metadata.get("tracks_to_predict", [])
        has_ego_trailer = int(bool(metadata.get("has_ego_trailer", False)))
        ego_trailer_track_index = int(metadata.get("ego_trailer_track_index", -1))
        non_kinematic_vehicle_params = metadata.get("non_kinematic_vehicle_params", {})
        if not isinstance(non_kinematic_vehicle_params, dict):
            non_kinematic_vehicle_params = {}
        non_kinematic_aliases = {
            "trailer_d_rear_axel2_rear_bumper": [
                "trailer_d_rear_axel2_rear_bumper",
                "trailer_d_rear_axle2_rear_bumper",
            ],
            "trailer_d_real_axel2_front_bumper": [
                "trailer_d_real_axel2_front_bumper",
                "trailer_d_rear_axel2_front_bumper",
                "trailer_d_rear_axle2_front_bumper",
            ],
        }

        # Write sdc_track_index
        f.write(struct.pack("i", sdc_track_index))

        # Write tracks_to_predict info (indices only)
        f.write(struct.pack("i", len(tracks_to_predict)))
        for track in tracks_to_predict:
            if isinstance(track, dict):
                track_index = track.get("track_index", -1)
            else:
                track_index = int(track)
            f.write(struct.pack("i", track_index))

        # Count total entities
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        # num_entities = num_objects + num_roads
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))
        # f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get("objects", []):
            # Write unique map id
            f.write(struct.pack("i", unique_map_id))

            # Write base entity data
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", obj_type))  # type
            f.write(struct.pack("i", normalize_int32_id(obj.get("id", 0))))  # id
            f.write(struct.pack("i", trajectory_length))  # array_size
            # Write position arrays
            positions = obj.get("position", [])
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("x", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("y", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("z", 0.0))))

            # Write velocity arrays
            velocities = obj.get("velocity", [])
            for arr, key in [(velocities, "x"), (velocities, "y"), (velocities, "z")]:
                for i in range(trajectory_length):
                    vel = arr[i] if i < len(arr) else {"x": 0.0, "y": 0.0, "z": 0.0}
                    f.write(struct.pack("f", float(vel.get(key, 0.0))))

            # Write heading and valid arrays
            headings = obj.get("heading", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}f",
                    *[float(headings[i]) if i < len(headings) else 0.0 for i in range(trajectory_length)],
                )
            )

            valids = obj.get("valid", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}i",
                    *[int(valids[i]) if i < len(valids) else 0 for i in range(trajectory_length)],
                )
            )

            # Write scalar fields
            f.write(struct.pack("f", float(obj.get("width", 0.0))))
            f.write(struct.pack("f", float(obj.get("length", 0.0))))
            f.write(struct.pack("f", float(obj.get("height", 0.0))))
            goal_pos = obj.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", obj.get("mark_as_expert", 0)))

        # Write roads
        for idx, road in enumerate(map_data.get("roads", [])):
            f.write(struct.pack("i", unique_map_id))

            geometry = road.get("geometry", [])
            road_type = road.get("map_element_id", 0)
            road_type_word = road.get("type", 0)
            if road_type_word == "lane":
                road_type = 2
            elif road_type_word == "road_edge":
                road_type = 15
            # breakpoint()
            if len(geometry) > 10 and road_type <= 16:
                geometry = simplify_polyline(geometry, 0.1, 250)
            size = len(geometry)
            # breakpoint()
            if road_type >= 0 and road_type <= 3:
                road_type = 4
            elif road_type >= 5 and road_type <= 13:
                road_type = 5
            elif road_type >= 14 and road_type <= 16:
                road_type = 6
            elif road_type == 17:
                road_type = 7
            elif road_type == 18:
                road_type = 8
            elif road_type == 19:
                road_type = 9
            elif road_type == 20:
                road_type = 10
            # Write base entity data
            f.write(struct.pack("i", road_type))  # type
            f.write(struct.pack("i", normalize_int32_id(road.get("id", 0))))  # id
            f.write(struct.pack("i", size))  # array_size

            # Write position arrays
            for coord in ["x", "y", "z"]:
                for point in geometry:
                    f.write(struct.pack("f", float(point.get(coord, 0.0))))

            # Write scalar fields
            f.write(struct.pack("f", float(road.get("width", 0.0))))
            f.write(struct.pack("f", float(road.get("length", 0.0))))
            f.write(struct.pack("f", float(road.get("height", 0.0))))
            goal_pos = road.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", road.get("mark_as_expert", 0)))

        objects = map_data.get("objects", [])
        f.write(struct.pack("i", extension_magic))
        f.write(struct.pack("i", extension_version))
        f.write(struct.pack("i", has_ego_trailer))
        f.write(struct.pack("i", ego_trailer_track_index))
        f.write(struct.pack("i", len(objects)))
        for obj_idx, obj in enumerate(objects):
            source_track_id_hash = stable_track_hash(obj.get("source_track_id"))
            is_trailer = int(has_ego_trailer and obj_idx == ego_trailer_track_index)
            parent_track_index = int(sdc_track_index if is_trailer else -1)
            f.write(struct.pack("Q", source_track_id_hash))
            f.write(struct.pack("i", is_trailer))
            f.write(struct.pack("i", parent_track_index))

        f.write(struct.pack("i", len(_NON_KINEMATIC_PARAM_ORDER)))
        for param_key in _NON_KINEMATIC_PARAM_ORDER:
            value = None
            for alias in non_kinematic_aliases.get(param_key, [param_key]):
                if alias in non_kinematic_vehicle_params:
                    value = non_kinematic_vehicle_params.get(alias)
                    break
            f.write(struct.pack("f", float(value if value is not None else 0.0)))


def load_map(map_name, unique_map_id, binary_output=None):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, "r") as f:
        map_data = json.load(f)

    if binary_output:
        save_map_binary(map_data, binary_output, unique_map_id)


def _process_single_map(args):
    """Worker function to process a single map file"""
    i, map_path, binary_path = args
    try:
        load_map(str(map_path), i, str(binary_path))
        return (i, map_path.name, True, None)
    except Exception as e:
        return (i, map_path.name, False, str(e))


def process_all_maps(
    data_folder="data/processed/training",
    output_folder=None,
    max_maps=50_000,
    num_workers=None,
):
    """Process all maps and save them as binaries using multiprocessing

    Args:
        data_folder: Path to the folder containing JSON map files
        output_folder: Path to save binary files (defaults to resources/drive/binaries/{dataset_name})
        max_maps: Maximum number of maps to process
        num_workers: Number of parallel workers (defaults to cpu_count())
    """
    from pathlib import Path

    if num_workers is None:
        num_workers = cpu_count()

    # Path to the training data
    data_dir = Path(data_folder)
    dataset_name = data_dir.name

    # Create the binaries directory if it doesn't exist
    if output_folder is None:
        binary_dir = Path(f"resources/drive/binaries/{dataset_name}")
    else:
        binary_dir = Path(output_folder)
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files in the training directory
    json_files = sorted(data_dir.glob("*.json"))

    # Prepare arguments for parallel processing
    tasks = []
    for i, map_path in enumerate(json_files[:max_maps]):
        binary_file = f"map_{i:03d}.bin"
        binary_path = binary_dir / binary_file
        tasks.append((i, map_path, binary_path))

    # Process maps in parallel with progress bar
    with Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(_process_single_map, tasks), total=len(tasks), desc="Processing maps", unit="map")
        )

    # Collect statistics
    successful = sum(1 for _, _, success, _ in results if success)
    failed = sum(1 for _, _, success, _ in results if not success)

    if failed > 0:
        print(f"\nFailed {failed}/{len(results)} files:")
        for i, name, success, error in results:
            if not success:
                print(f"  {name}: {error}")


def test_performance(timeout=10, atn_cache=1024, num_agents=1024):
    import time

    env = Drive(
        num_agents=num_agents,
        num_maps=1,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        init_steps=0,
        episode_length=91,
    )

    env.reset()

    tick = 0
    actions = np.stack(
        [np.random.randint(0, space.n + 1, (atn_cache, num_agents)) for space in env.single_action_space], axis=-1
    )

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {num_agents * tick / (time.time() - start)}")

    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert JSON map files to binary format")
    parser.add_argument("--input-folder", type=str, default="data/processed/training",
                        help="Path to folder containing JSON map files")
    parser.add_argument("--output-folder", type=str, default=None,
                        help="Path to save binary files (default: resources/drive/binaries/{dataset_name})")
    parser.add_argument("--max-maps", type=int, default=50_000,
                        help="Maximum number of maps to process")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")
    
    args = parser.parse_args()
    
    process_all_maps(
        data_folder=args.input_folder,
        output_folder=args.output_folder,
        max_maps=args.max_maps,
        num_workers=args.num_workers
    )
