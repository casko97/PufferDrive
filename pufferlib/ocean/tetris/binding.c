#include "tetris.h"

#define Env Tetris
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->n_rows = unpack(kwargs, "n_rows");
    env->n_cols = unpack(kwargs, "n_cols");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    return 0;
}