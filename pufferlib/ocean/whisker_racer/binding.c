#include "whisker_racer.h"

#define Env WhiskerRacer
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->frameskip = unpack(kwargs, "frameskip");
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->llw_ang = unpack(kwargs, "llw_ang");
    env->flw_ang = unpack(kwargs, "flw_ang");
    env->frw_ang = unpack(kwargs, "frw_ang");
    env->rrw_ang = unpack(kwargs, "rrw_ang");
    env->max_whisker_length = unpack(kwargs, "max_whisker_length");
    env->turn_pi_frac = unpack(kwargs, "turn_pi_frac");
    env->maxv = unpack(kwargs, "maxv");
    env->continuous = unpack(kwargs, "continuous");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
