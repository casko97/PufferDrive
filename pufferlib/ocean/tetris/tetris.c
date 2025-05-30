#include <time.h>
#include "tetris.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

void demo() {
    Tetris env = {
        .n_rows = 10,
        .n_cols = 10,
    };
    allocate(&env);

    env.client = make_client(&env);

    int target_rotation = 0;
    int target_col = env.n_cols/2;
    c_reset(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = 0.0;
        if (IsKeyPressed(KEY_RIGHT)  || IsKeyPressed(KEY_D)){
            if (TETROMINOES_FILLS_COLUMN[env.current_tetromino][target_rotation] + target_col + 1 <= env.n_cols){
                target_col = min(env.n_cols - 1,target_col+1);
            }
        }
        if (IsKeyPressed(KEY_LEFT)  || IsKeyPressed(KEY_A)) target_col = max(0, target_col - 1);
        if (IsKeyPressed(KEY_R)) {
            if (TETROMINOES_FILLS_COLUMN[env.current_tetromino][(target_rotation + 1) % NUM_ROTATIONS] + target_col <= env.n_cols){
                target_rotation = (target_rotation + 1) % NUM_ROTATIONS;
            }
        }
        if (IsKeyPressed(KEY_ENTER)){
            env.actions[0] = target_col + env.n_cols * target_rotation;
            c_step(&env);
            target_rotation = 0;
            target_col = env.n_cols/2;
        }
        preview_new_tetromino(&env, target_rotation, target_col);
        c_render(&env);
    }
    free_allocated(&env);
    close_client(env.client);
}

int main() {
    demo();
}
