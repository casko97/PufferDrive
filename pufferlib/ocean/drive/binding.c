#include <Python.h>
#include <time.h>

static int startup_timing_enabled = 0;
static double startup_timing_load_map_binary = 0.0;
static double startup_timing_set_means = 0.0;
static double startup_timing_init_grid_map = 0.0;
static double startup_timing_init_neighbor_offsets = 0.0;
static double startup_timing_cache_neighbor_offsets = 0.0;
static double startup_timing_set_active_agents = 0.0;
static double startup_timing_remove_bad_trajectories = 0.0;
static double startup_timing_set_start_position = 0.0;
static double startup_timing_invalid_initial_trailer_state = 0.0;
static double startup_timing_init_goal_positions = 0.0;
static double startup_timing_alloc_logs = 0.0;
static long startup_timing_init_calls = 0;

static double elapsed_seconds(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0;
}

#include "drive.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
static PyObject *vec_has_invalid_initial_trailer_state(PyObject *self, PyObject *args);
static PyObject *inspect_map(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *startup_timing_enable(PyObject *self, PyObject *args);
static PyObject *startup_timing_reset(PyObject *self, PyObject *args);
static PyObject *startup_timing_get(PyObject *self, PyObject *args);
static PyObject *env_get_active_agent_count(PyObject *self, PyObject *args);
static PyObject *env_get_active_agent_info(PyObject *self, PyObject *args);
static PyObject *env_get_partner_types(PyObject *self, PyObject *args);
static PyObject *env_get_ego_trailer_obs_features(PyObject *self, PyObject *args);
static PyObject *env_get_config(PyObject *self, PyObject *args);
static PyObject *env_set_logged_timestep(PyObject *self, PyObject *args);
static PyObject *env_copy_observations(PyObject *self, PyObject *args);
static PyObject *env_fit_discrete_action_sequence(PyObject *self, PyObject *args);
#define MY_METHODS                                                                                                     \
    {"vec_has_invalid_initial_trailer_state", vec_has_invalid_initial_trailer_state, METH_VARARGS,                   \
     "Return True if any sub-environment has invalid initial trailer state"},                                         \
    {"inspect_map", (PyCFunction)inspect_map, METH_VARARGS | METH_KEYWORDS,                                          \
     "Inspect a single map path and return initialization metadata"},                                                 \
    {"startup_timing_enable", startup_timing_enable, METH_VARARGS, "Enable or disable startup timing hooks"},        \
    {"startup_timing_reset", startup_timing_reset, METH_NOARGS, "Reset startup timing accumulators"},                \
    {"startup_timing_get", startup_timing_get, METH_NOARGS, "Get startup timing accumulators"},                      \
    {"env_get_active_agent_count", env_get_active_agent_count, METH_VARARGS, "Get active agent count"},              \
    {"env_get_active_agent_info", env_get_active_agent_info, METH_VARARGS, "Get active agent scenario/id info"},     \
    {"env_get_partner_types", env_get_partner_types, METH_VARARGS, "Get partner type ids"},                          \
    {"env_get_ego_trailer_obs_features", env_get_ego_trailer_obs_features, METH_VARARGS,                             \
     "Get ego trailer-relative observation features"},                                                                 \
    {"env_get_config", env_get_config, METH_VARARGS, "Get resolved environment config values"},                      \
    {"env_set_logged_timestep", env_set_logged_timestep, METH_VARARGS, "Set env state to a logged timestep"},        \
    {"env_copy_observations", env_copy_observations, METH_VARARGS, "Copy current observation buffer"},               \
    {"env_fit_discrete_action_sequence", env_fit_discrete_action_sequence, METH_VARARGS,                              \
     "Fit a discrete action sequence against GT"}
#include "../env_binding.h"

static int unpack_non_kinematic_override(PyObject *kwargs, float *dst, int *enabled) {
    *enabled = 0;
    PyObject *obj = kwargs ? PyDict_GetItemString(kwargs, "non_kinematic_vehicle_params_override") : NULL;
    if (obj == NULL || obj == Py_None) {
        return 0;
    }
    if (!PySequence_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "non_kinematic_vehicle_params_override must be a sequence of 13 floats");
        return -1;
    }

    Py_ssize_t n = PySequence_Size(obj);
    if (n != 13) {
        PyErr_SetString(PyExc_ValueError, "non_kinematic_vehicle_params_override must have length 13");
        return -1;
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (!item) {
            return -1;
        }
        double v = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "non_kinematic_vehicle_params_override contains non-float value");
            return -1;
        }
        dst[i] = (float)v;
    }

    *enabled = 1;
    return 0;
}

static int has_kwarg(PyObject *kwargs, const char *key) {
    return kwargs && PyDict_GetItemString(kwargs, key) != NULL;
}

static int override_int(PyObject *kwargs, const char *key, int *dst) {
    if (!has_kwarg(kwargs, key)) {
        return 0;
    }
    *dst = (int)unpack(kwargs, key);
    return PyErr_Occurred() ? -1 : 0;
}

static int override_float(PyObject *kwargs, const char *key, float *dst) {
    if (!has_kwarg(kwargs, key)) {
        return 0;
    }
    *dst = (float)unpack(kwargs, key);
    return PyErr_Occurred() ? -1 : 0;
}

static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    PyObject *obs = PyDict_GetItemString(kwargs, "observations");
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject *observations = (PyArrayObject *)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return 1;
    }
    env->observations = PyArray_DATA(observations);

    PyObject *act = PyDict_GetItemString(kwargs, "actions");
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return 1;
    }
    PyArrayObject *actions = (PyArrayObject *)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return 1;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return 1;
    }

    PyObject *rew = PyDict_GetItemString(kwargs, "rewards");
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return 1;
    }
    PyArrayObject *rewards = (PyArrayObject *)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return 1;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject *term = PyDict_GetItemString(kwargs, "terminals");
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return 1;
    }
    PyArrayObject *terminals = (PyArrayObject *)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return 1;
    }
    env->terminals = PyArray_DATA(terminals);
    return 0;
}

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs) {
    // Legacy helper retained for compatibility. The primary Drive runtime no longer
    // depends on this dataset-wide prescan path for startup, resampling, or eval.
    char *map_dir = unpack_str(kwargs, "map_dir");
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    int dynamics_model = unpack(kwargs, "dynamics_model");
    int init_mode = unpack(kwargs, "init_mode");
    int control_mode = unpack(kwargs, "control_mode");
    int init_steps = unpack(kwargs, "init_steps");
    int max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    int goal_behavior = unpack(kwargs, "goal_behavior");
    float goal_target_distance = unpack(kwargs, "goal_target_distance");
    int sequential_map_sampling = unpack(kwargs, "sequential_map_sampling");
    float non_kinematic_override[13] = {0};
    int override_non_kinematic = 0;
    if (unpack_non_kinematic_override(kwargs, non_kinematic_override, &override_non_kinematic) != 0) {
        return NULL;
    }
    int force_zero_trailer_articulation_at_init = unpack(kwargs, "force_zero_trailer_articulation_at_init");
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec);
    int total_agent_count = 0;
    int env_count = 0;
    int max_envs = sequential_map_sampling ? num_maps : num_agents;
    int map_idx = 0;
    int maps_checked = 0;
    PyObject *agent_offsets = PyList_New(max_envs + 1);
    PyObject *map_ids = PyList_New(max_envs);
    // getting env count
    while (sequential_map_sampling ? map_idx < max_envs : total_agent_count < num_agents && env_count < max_envs) {
        char map_file[512];
        // Take the next map in sequence or a random map
        int map_id = sequential_map_sampling ? map_idx++ : rand() % num_maps;
        Drive *env = calloc(1, sizeof(Drive));
        env->init_mode = init_mode;
        env->control_mode = control_mode;
        env->init_steps = init_steps;
        env->dynamics_model = dynamics_model;
        env->max_controlled_agents = max_controlled_agents;
        env->goal_behavior = goal_behavior;
        env->goal_target_distance = goal_target_distance;
        env->override_non_kinematic_vehicle_params = override_non_kinematic;
        env->force_zero_trailer_articulation_at_init = force_zero_trailer_articulation_at_init;
        if (override_non_kinematic) {
            for (int j = 0; j < 13; j++) {
                env->non_kinematic_vehicle_params_override[j] = non_kinematic_override[j];
            }
        }
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
        env->entities = load_map_binary(map_file, env);
        set_active_agents(env);
        set_start_position(env);
        int invalid_initial_trailer_state = has_invalid_initial_sdc_trailer_collision(env);

        // Skip map if it doesn't contain any controllable agents
        if (env->active_agent_count == 0 || invalid_initial_trailer_state) {
            if (!sequential_map_sampling) {
                maps_checked++;

                // Safeguard: if we've checked all available maps and found no active agents, raise an error
                if (maps_checked >= num_maps) {
                    for (int j = 0; j < env->num_entities; j++) {
                        free_entity(&env->entities[j]);
                    }
                    free(env->entities);
                    free(env->active_agent_indices);
                    free(env->static_agent_indices);
                    free(env->expert_static_agent_indices);
                    free(env->tracks_to_predict_indices);
                    free(env);
                    Py_DECREF(agent_offsets);
                    Py_DECREF(map_ids);
                    char error_msg[256];
                    if (invalid_initial_trailer_state) {
                        sprintf(error_msg,
                                "No valid maps left: all %d candidates had invalid initial SDC trailer state",
                                num_maps);
                    } else {
                        sprintf(error_msg, "No controllable agents found in any of the %d available maps", num_maps);
                    }
                    PyErr_SetString(PyExc_ValueError, error_msg);
                    return NULL;
                }
            }

            for (int j = 0; j < env->num_entities; j++) {
                free_entity(&env->entities[j]);
            }
            free(env->entities);
            free(env->active_agent_indices);
            free(env->static_agent_indices);
            free(env->expert_static_agent_indices);
            free(env->tracks_to_predict_indices);
            free(env);
            continue;
        }

        // Store map_id
        PyObject *map_id_obj = PyLong_FromLong(map_id);
        PyList_SetItem(map_ids, env_count, map_id_obj);
        // Store agent offset
        PyObject *offset = PyLong_FromLong(total_agent_count);
        PyList_SetItem(agent_offsets, env_count, offset);
        total_agent_count += env->active_agent_count;
        env_count++;
        for (int j = 0; j < env->num_entities; j++) {
            free_entity(&env->entities[j]);
        }
        free(env->entities);
        free(env->active_agent_indices);
        free(env->static_agent_indices);
        free(env->expert_static_agent_indices);
        free(env->tracks_to_predict_indices);
        free(env);
    }
    // printf("Generated %d environments to cover %d agents (requested %d agents)\n", env_count, total_agent_count,
    // num_agents);
    if (!sequential_map_sampling && total_agent_count >= num_agents) {
        total_agent_count = num_agents;
    }
    PyObject *final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, env_count, final_total_agent_count);
    PyObject *final_env_count = PyLong_FromLong(env_count);
    // resize lists
    PyObject *resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject *resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    PyObject *tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    return tuple;
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    if (ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "episode_length")) {
        conf.episode_length = (int)unpack(kwargs, "episode_length");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_vehicle_collision")) {
        conf.reward_vehicle_collision = (float)unpack(kwargs, "reward_vehicle_collision");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_offroad_collision")) {
        conf.reward_offroad_collision = (float)unpack(kwargs, "reward_offroad_collision");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_goal")) {
        conf.reward_goal = (float)unpack(kwargs, "reward_goal");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "reward_goal_post_respawn")) {
        conf.reward_goal_post_respawn = (float)unpack(kwargs, "reward_goal_post_respawn");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "goal_radius")) {
        conf.goal_radius = (float)unpack(kwargs, "goal_radius");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "goal_speed")) {
        conf.goal_speed = (float)unpack(kwargs, "goal_speed");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "goal_behavior")) {
        conf.goal_behavior = (int)unpack(kwargs, "goal_behavior");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "goal_target_distance")) {
        conf.goal_target_distance = (float)unpack(kwargs, "goal_target_distance");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "collision_behavior")) {
        conf.collision_behavior = (int)unpack(kwargs, "collision_behavior");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "offroad_behavior")) {
        conf.offroad_behavior = (int)unpack(kwargs, "offroad_behavior");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "termination_mode")) {
        conf.termination_mode = (int)unpack(kwargs, "termination_mode");
    }
    if (kwargs && PyDict_GetItemString(kwargs, "dt")) {
        conf.dt = (float)unpack(kwargs, "dt");
    }
    if (conf.episode_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "episode_length must be > 0 (set in INI or kwargs)");
        return -1;
    }
    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    env->observation_mode = conf.observation_mode;
    env->extend_classic_action_space = conf.extend_classic_action_space;
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->episode_length = conf.episode_length;
    env->termination_mode = conf.termination_mode;
    env->collision_behavior = conf.collision_behavior;
    env->offroad_behavior = conf.offroad_behavior;
    env->max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    env->dt = conf.dt;
    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    env->goal_behavior = (int)unpack(kwargs, "goal_behavior");
    env->goal_target_distance = (float)unpack(kwargs, "goal_target_distance");
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    env->goal_speed = (float)unpack(kwargs, "goal_speed");
    if (override_int(kwargs, "action_type", &env->action_type) != 0 ||
        override_int(kwargs, "dynamics_model", &env->dynamics_model) != 0 ||
        override_int(kwargs, "observation_mode", &env->observation_mode) != 0 ||
        override_int(kwargs, "extend_classic_action_space", &env->extend_classic_action_space) != 0 ||
        override_float(kwargs, "reward_vehicle_collision", &env->reward_vehicle_collision) != 0 ||
        override_float(kwargs, "reward_offroad_collision", &env->reward_offroad_collision) != 0 ||
        override_float(kwargs, "reward_goal", &env->reward_goal) != 0 ||
        override_float(kwargs, "reward_goal_post_respawn", &env->reward_goal_post_respawn) != 0 ||
        override_int(kwargs, "episode_length", &env->episode_length) != 0 ||
        override_int(kwargs, "termination_mode", &env->termination_mode) != 0 ||
        override_int(kwargs, "collision_behavior", &env->collision_behavior) != 0 ||
        override_int(kwargs, "offroad_behavior", &env->offroad_behavior) != 0 ||
        override_float(kwargs, "dt", &env->dt) != 0 ||
        override_int(kwargs, "init_mode", &env->init_mode) != 0 ||
        override_int(kwargs, "control_mode", &env->control_mode) != 0 ||
        override_int(kwargs, "goal_behavior", &env->goal_behavior) != 0 ||
        override_float(kwargs, "goal_target_distance", &env->goal_target_distance) != 0 ||
        override_float(kwargs, "goal_radius", &env->goal_radius) != 0 ||
        override_float(kwargs, "goal_speed", &env->goal_speed) != 0) {
        return -1;
    }
    env->vision_range = kwargs && PyDict_GetItemString(kwargs, "vision_range") ? (int)unpack(kwargs, "vision_range")
                                                                               : conf.vision_range;
    env->force_zero_trailer_articulation_at_init = (int)unpack(kwargs, "force_zero_trailer_articulation_at_init");
    env->override_non_kinematic_vehicle_params = 0;
    for (int i = 0; i < 13; i++) {
        env->non_kinematic_vehicle_params_override[i] = 0.0f;
    }
    if (unpack_non_kinematic_override(kwargs, env->non_kinematic_vehicle_params_override,
                                      &env->override_non_kinematic_vehicle_params) != 0) {
        return -1;
    }
    char *map_path = kwargs && PyDict_GetItemString(kwargs, "map_path") ? unpack_str(kwargs, "map_path") : NULL;
    char *map_dir = kwargs && PyDict_GetItemString(kwargs, "map_dir") ? unpack_str(kwargs, "map_dir") : NULL;
    int map_id = kwargs && PyDict_GetItemString(kwargs, "map_id") ? unpack(kwargs, "map_id") : 0;
    int max_agents = unpack(kwargs, "max_agents");
    int init_steps = unpack(kwargs, "init_steps");
    char map_file[512];
    env->num_agents = max_agents;
    if (map_path != NULL && map_path[0] != '\0') {
        env->map_name = strdup(map_path);
    } else {
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
        env->map_name = strdup(map_file);
    }
    env->init_steps = init_steps;
    env->timestep = init_steps;
    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "offroad_per_agent", log->offroad_per_agent);
    assign_to_dict(dict, "collisions_per_agent", log->collisions_per_agent);
    assign_to_dict(dict, "goals_sampled_this_episode", log->goals_sampled_this_episode);
    assign_to_dict(dict, "goals_reached_this_episode", log->goals_reached_this_episode);
    assign_to_dict(dict, "speed_at_goal", log->speed_at_goal);
    // assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    return 0;
}

static PyObject *inspect_map(PyObject *self, PyObject *args, PyObject *kwargs) {
    char *map_path = unpack_str(kwargs, "map_path");
    int dynamics_model = unpack(kwargs, "dynamics_model");
    int init_mode = unpack(kwargs, "init_mode");
    int control_mode = unpack(kwargs, "control_mode");
    int init_steps = unpack(kwargs, "init_steps");
    int max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    int goal_behavior = unpack(kwargs, "goal_behavior");
    float goal_target_distance = unpack(kwargs, "goal_target_distance");
    float non_kinematic_override[13] = {0};
    int override_non_kinematic = 0;
    if (unpack_non_kinematic_override(kwargs, non_kinematic_override, &override_non_kinematic) != 0) {
        return NULL;
    }
    int force_zero_trailer_articulation_at_init = unpack(kwargs, "force_zero_trailer_articulation_at_init");

    Drive *env = calloc(1, sizeof(Drive));
    env->map_name = strdup(map_path);
    env->init_mode = init_mode;
    env->control_mode = control_mode;
    env->init_steps = init_steps;
    env->dynamics_model = dynamics_model;
    env->max_controlled_agents = max_controlled_agents;
    env->goal_behavior = goal_behavior;
    env->goal_target_distance = goal_target_distance;
    env->override_non_kinematic_vehicle_params = override_non_kinematic;
    env->force_zero_trailer_articulation_at_init = force_zero_trailer_articulation_at_init;
    if (override_non_kinematic) {
        for (int j = 0; j < 13; j++) {
            env->non_kinematic_vehicle_params_override[j] = non_kinematic_override[j];
        }
    }

    env->entities = load_map_binary(map_path, env);
    if (env->entities == NULL) {
        free(env->map_name);
        free(env);
        PyErr_SetString(PyExc_FileNotFoundError, map_path);
        return NULL;
    }

    set_active_agents(env);
    set_start_position(env);
    int invalid_initial_trailer_state = has_invalid_initial_sdc_trailer_collision(env);
    int active_agent_count = env->active_agent_count;

    PyObject *result = PyDict_New();
    PyDict_SetItemString(result, "active_agent_count", PyLong_FromLong(active_agent_count));
    PyDict_SetItemString(result, "invalid_initial_trailer_state", invalid_initial_trailer_state ? Py_True : Py_False);
    PyDict_SetItemString(result, "valid_for_sampling",
                         (active_agent_count > 0 && !invalid_initial_trailer_state) ? Py_True : Py_False);

    for (int j = 0; j < env->num_entities; j++) {
        free_entity(&env->entities[j]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->static_agent_indices);
    free(env->expert_static_agent_indices);
    free(env->tracks_to_predict_indices);
    free(env->map_name);
    free(env);
    return result;
}

static PyObject *vec_has_invalid_initial_trailer_state(PyObject *self, PyObject *args) {
    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    for (int i = 0; i < vec->num_envs; i++) {
        if (vec->envs[i]->invalid_initial_trailer_state) {
            Py_RETURN_TRUE;
        }
    }
    Py_RETURN_FALSE;
}

static PyObject *startup_timing_enable(PyObject *self, PyObject *args) {
    int enabled = 0;
    if (!PyArg_ParseTuple(args, "p", &enabled)) {
        return NULL;
    }
    startup_timing_enabled = enabled;
    Py_RETURN_NONE;
}

static PyObject *startup_timing_reset(PyObject *self, PyObject *args) {
    startup_timing_load_map_binary = 0.0;
    startup_timing_set_means = 0.0;
    startup_timing_init_grid_map = 0.0;
    startup_timing_init_neighbor_offsets = 0.0;
    startup_timing_cache_neighbor_offsets = 0.0;
    startup_timing_set_active_agents = 0.0;
    startup_timing_remove_bad_trajectories = 0.0;
    startup_timing_set_start_position = 0.0;
    startup_timing_invalid_initial_trailer_state = 0.0;
    startup_timing_init_goal_positions = 0.0;
    startup_timing_alloc_logs = 0.0;
    startup_timing_init_calls = 0;
    Py_RETURN_NONE;
}

static PyObject *startup_timing_get(PyObject *self, PyObject *args) {
    PyObject *dict = PyDict_New();
    if (!dict) {
        return NULL;
    }
    PyDict_SetItemString(dict, "enabled", startup_timing_enabled ? Py_True : Py_False);
    PyDict_SetItemString(dict, "init_calls", PyLong_FromLong(startup_timing_init_calls));
    PyDict_SetItemString(dict, "load_map_binary", PyFloat_FromDouble(startup_timing_load_map_binary));
    PyDict_SetItemString(dict, "set_means", PyFloat_FromDouble(startup_timing_set_means));
    PyDict_SetItemString(dict, "init_grid_map", PyFloat_FromDouble(startup_timing_init_grid_map));
    PyDict_SetItemString(dict, "init_neighbor_offsets", PyFloat_FromDouble(startup_timing_init_neighbor_offsets));
    PyDict_SetItemString(dict, "cache_neighbor_offsets", PyFloat_FromDouble(startup_timing_cache_neighbor_offsets));
    PyDict_SetItemString(dict, "set_active_agents", PyFloat_FromDouble(startup_timing_set_active_agents));
    PyDict_SetItemString(dict, "remove_bad_trajectories", PyFloat_FromDouble(startup_timing_remove_bad_trajectories));
    PyDict_SetItemString(dict, "set_start_position", PyFloat_FromDouble(startup_timing_set_start_position));
    PyDict_SetItemString(dict, "invalid_initial_trailer_state",
                         PyFloat_FromDouble(startup_timing_invalid_initial_trailer_state));
    PyDict_SetItemString(dict, "init_goal_positions", PyFloat_FromDouble(startup_timing_init_goal_positions));
    PyDict_SetItemString(dict, "alloc_logs", PyFloat_FromDouble(startup_timing_alloc_logs));
    return dict;
}

static PyObject *env_get_partner_types(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "env_get_partner_types requires 2 arguments");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }

    PyObject *types_arr = PyTuple_GetItem(args, 1);
    if (!PyArray_Check(types_arr)) {
        PyErr_SetString(PyExc_TypeError, "Output array must be a NumPy array");
        return NULL;
    }

    c_get_partner_types(env, (int *)PyArray_DATA((PyArrayObject *)types_arr));
    Py_RETURN_NONE;
}

static PyObject *env_get_ego_trailer_obs_features(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 5) {
        PyErr_SetString(PyExc_TypeError, "env_get_ego_trailer_obs_features requires 5 arguments");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }

    PyObject *rel_x_arr = PyTuple_GetItem(args, 1);
    PyObject *rel_y_arr = PyTuple_GetItem(args, 2);
    PyObject *rel_heading_x_arr = PyTuple_GetItem(args, 3);
    PyObject *rel_heading_y_arr = PyTuple_GetItem(args, 4);
    if (!PyArray_Check(rel_x_arr) || !PyArray_Check(rel_y_arr) || !PyArray_Check(rel_heading_x_arr) ||
        !PyArray_Check(rel_heading_y_arr)) {
        PyErr_SetString(PyExc_TypeError, "All output arrays must be NumPy arrays");
        return NULL;
    }

    c_get_ego_trailer_obs_features(env, (float *)PyArray_DATA((PyArrayObject *)rel_x_arr),
                                   (float *)PyArray_DATA((PyArrayObject *)rel_y_arr),
                                   (float *)PyArray_DATA((PyArrayObject *)rel_heading_x_arr),
                                   (float *)PyArray_DATA((PyArrayObject *)rel_heading_y_arr));
    Py_RETURN_NONE;
}

static PyObject *env_get_config(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "env_get_config requires 1 argument");
        return NULL;
    }

    Env *env = unpack_env(args);
    if (!env) {
        return NULL;
    }

    PyObject *dict = PyDict_New();
    if (!dict) {
        return NULL;
    }
    assign_to_dict(dict, "action_type", env->action_type);
    assign_to_dict(dict, "dynamics_model", env->dynamics_model);
    assign_to_dict(dict, "observation_mode", env->observation_mode);
    assign_to_dict(dict, "extend_classic_action_space", env->extend_classic_action_space);
    assign_to_dict(dict, "reward_vehicle_collision", env->reward_vehicle_collision);
    assign_to_dict(dict, "reward_offroad_collision", env->reward_offroad_collision);
    assign_to_dict(dict, "reward_goal", env->reward_goal);
    assign_to_dict(dict, "reward_goal_post_respawn", env->reward_goal_post_respawn);
    assign_to_dict(dict, "goal_radius", env->goal_radius);
    assign_to_dict(dict, "goal_speed", env->goal_speed);
    assign_to_dict(dict, "goal_behavior", env->goal_behavior);
    assign_to_dict(dict, "goal_target_distance", env->goal_target_distance);
    assign_to_dict(dict, "collision_behavior", env->collision_behavior);
    assign_to_dict(dict, "offroad_behavior", env->offroad_behavior);
    assign_to_dict(dict, "dt", env->dt);
    assign_to_dict(dict, "episode_length", env->episode_length);
    assign_to_dict(dict, "termination_mode", env->termination_mode);
    assign_to_dict(dict, "init_steps", env->init_steps);
    assign_to_dict(dict, "init_mode", env->init_mode);
    assign_to_dict(dict, "control_mode", env->control_mode);
    assign_to_dict(dict, "max_controlled_agents", env->max_controlled_agents);
    return dict;
}
