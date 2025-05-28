#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include "tetrominoes.h"

// Rendering related
#define HALF_LINEWIDTH 2
#define SQUARE_SIZE 64

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
typedef struct Tetris Tetris;
struct Tetris {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* human_actions;
    float* rewards;
    unsigned char* terminals;

    int step;
    int n_rows;
    int n_cols;
    int* preview_grid;
    int* grid;
    int score;
    int* action_mask;
    int* row_is_free;
    float ep_return;
    int current_tetrimino;
};

void init(Tetris* env) {
    env->action_mask = (int*)calloc(env->n_cols * NUM_ROTATIONS, sizeof(int));
    env->row_is_free = (int*)calloc(env->n_rows, sizeof(int));
    env->grid = (int*)calloc(env->n_rows * env->n_cols, sizeof(int));
    env->preview_grid = (int*)calloc(env->n_cols * SIZE, sizeof(int));
}

void allocate(Tetris* env) {
    init(env);
    env->observations = (float*)calloc(1 + NUM_TETROMINOES + env->n_cols*NUM_ROTATIONS + env->n_rows * env->n_cols, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void c_close(Tetris* env) {
    free(env->grid);
    free(env->preview_grid);
    free(env->action_mask);
    free(env->row_is_free);
}

void restore_grid(Tetris* env) {
    for (int r = 0; r < env->n_rows; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            env->grid[r * env->n_cols + c] = 0;
        }
    }
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            env->preview_grid[r * env->n_cols + c] = 0;
        }
    }
    for (int r = 0; r < SIZE; r++) {
        env->row_is_free[r] = 1;
    }
}

void draw_new_tetromino(Tetris* env) {
    env->current_tetrimino = rand() % NUM_TETROMINOES;
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            env->preview_grid[r * env->n_cols + c + env->n_cols/2] = 0;
        }
    }
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (TETROMINOES[env->current_tetrimino][0][r][c] == 1) {
                env->preview_grid[r * env->n_cols + c + env->n_cols/2] = env->current_tetrimino + 1;
            }
        }
    }
}


void free_allocated(Tetris* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(Tetris* env) {
    env->log.episode_length += env->step;
    env->log.episode_return += env->ep_return;
    env->log.score += env->score;
    env->log.perf += env->score;
    env->log.n += 1;
}

int is_valid_tetromino(Tetris* env, int tetrimino_idx, int col, int rotation) {
    int landing_row = -1;
    for (int r = 0; r < SIZE; r++) {
        if (env->row_is_free[r]) {
            landing_row = r;
        }
        else {
            break;
        }
    }
    for (int test_row = 0; test_row < env->n_rows; test_row++) {
        bool can_place = true;
        
        for (int r = 0; r < SIZE && can_place; r++) {
            for (int c = 0; c < SIZE && can_place; c++) {
                if (TETROMINOES[tetrimino_idx][rotation][r][c] == 1) {
                    int grid_row = test_row + r;
                    int grid_col = col + c;
                    
                    if (grid_row >= env->n_rows || grid_col < 0 || grid_col >= env->n_cols) {
                        can_place = false;
                    }
                    else if (env->grid[grid_row * env->n_cols + grid_col] != 0) {
                        can_place = false;
                    }
                }
            }
        }
        
        if (can_place) {
            landing_row = test_row;
        } else {
            break;
        }
    }
    
    return landing_row;
}

void place_tetromino(Tetris* env, int tetrimino_idx, int col, int rotation, int landing_row) {
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (TETROMINOES[tetrimino_idx][rotation][r][c] == 1) {
                int grid_row = landing_row + r;
                int grid_col = col + c;

                if (grid_row < env->n_rows && grid_col >= 0 && grid_col < env->n_cols) {
                    env->grid[grid_row * env->n_cols + grid_col] = tetrimino_idx + 1;
                    env->row_is_free[grid_row] = 0;
                }
            }
        }
    }
}

void compute_action_mask(Tetris* env, int tetrimino_idx) {
    bool valid;
    if (tetrimino_idx == 0){
        for (int c = 0; c < env->n_cols; c++) {
            valid = is_valid_tetromino(env, tetrimino_idx, c, 0) > 0;
            for (int rot = 0; rot < NUM_ROTATIONS; rot++) {
                env->action_mask[rot * env->n_cols + c] = valid;
            }
        }
    }

    else {
        for (int c = 0; c < env->n_cols; c++) {
            for (int rot = 0; rot < NUM_ROTATIONS; rot++) {
                valid = is_valid_tetromino(env, tetrimino_idx, c, rot) > 0;
                env->action_mask[rot * env->n_cols + c] = valid;
            }
        }
    }
}

void compute_observations(Tetris* env) {
    env->observations[0] = env->score;
    for (int i = 0; i < NUM_TETROMINOES; i++) {
        env->observations[1+i] = 0;
    }
    env->observations[1+env->current_tetrimino] = 1; 
    for (int i = 0; i < NUM_TETROMINOES; i++) {
        env->observations[1+i] = 0;
    }
    env->observations[1+env->current_tetrimino] = 1; 
    for (int i = 0; i < env->n_cols*NUM_ROTATIONS; i++) {
        env->observations[1+NUM_TETROMINOES+i] = env->action_mask[i];
    }
    for (int i = 0; i < env->n_cols*env->n_rows; i++) {
        env->observations[1+NUM_TETROMINOES+env->n_cols*NUM_ROTATIONS+i] = env->grid[i];
    }
}

void c_reset(Tetris* env) {
    env->score = 0.0f;
    env->ep_return = 0.0;
    env->step = 0;
    draw_new_tetromino(env);
    restore_grid(env);
    compute_action_mask(env, env->current_tetrimino);
    compute_observations(env);
}

void c_step(Tetris* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
    int col = ((int) env->actions[0])%env->n_cols;
    int rotation = ((int) env->actions[0])/env->n_cols;
    
    int landing_row = -1;
    if (env->action_mask[rotation * env->n_cols + col]) {
        landing_row = is_valid_tetromino(env, env->current_tetrimino, col, rotation);
        place_tetromino(env, env->current_tetrimino, col, rotation, landing_row);
        draw_new_tetromino(env);
        compute_action_mask(env, env->current_tetrimino);

    }
    int end = true;
    for (int i = 0; i < NUM_ROTATIONS * env->n_cols; i++) {
        if (env->action_mask[i]) {
            end = false;
            break;
        }
    }
    if (end){
        env->terminals[0] = 1;
        env->rewards[0] = env->score;
        c_reset(env);
    } 
    else{
        compute_observations(env);
    };
}

typedef struct Client Client;
struct Client {
};

Client* make_client(Tetris* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    InitWindow(SQUARE_SIZE * env->n_cols, SQUARE_SIZE * (5 + env->n_rows), "PufferLib Tetris");
    SetTargetFPS(60);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

Color BG_COLOR = (Color){100, 100, 100, 255};
Color DASH_COLOR = (Color){170, 170, 170, 255};
Color DASH_COLOR2 = (Color){150, 150, 150, 255};

void c_render(Tetris* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    BeginDrawing();
    ClearBackground(BG_COLOR);

    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            int x = c * SQUARE_SIZE;
            int y = (r + 1) * SQUARE_SIZE;
            Color color = (env->preview_grid[r*env->n_cols + c] == 0) ? BG_COLOR : TETROMINOES_COLORS[env->preview_grid[r*env->n_cols + c] - 1];
            DrawRectangle(
                x + HALF_LINEWIDTH, 
                y + HALF_LINEWIDTH,
                SQUARE_SIZE-2*HALF_LINEWIDTH, SQUARE_SIZE-2*HALF_LINEWIDTH, color
            );
            DrawRectangle(
                x - HALF_LINEWIDTH, 
                y - HALF_LINEWIDTH,
                SQUARE_SIZE, 2*HALF_LINEWIDTH, DASH_COLOR2
            );
            DrawRectangle(
                x - HALF_LINEWIDTH, 
                y + SQUARE_SIZE- HALF_LINEWIDTH,
                SQUARE_SIZE, 2*HALF_LINEWIDTH, DASH_COLOR2
            );
            DrawRectangle(
                x - HALF_LINEWIDTH, 
                y - HALF_LINEWIDTH,
                2*HALF_LINEWIDTH, + SQUARE_SIZE, DASH_COLOR2
            );
            DrawRectangle(
                x + SQUARE_SIZE - HALF_LINEWIDTH, 
                y + SQUARE_SIZE- HALF_LINEWIDTH,
                2*HALF_LINEWIDTH, + SQUARE_SIZE, DASH_COLOR2
            );
        }         
    }
    
    for (int r = 0; r < env->n_rows; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            int x = c * SQUARE_SIZE;
            int y = (r + SIZE + 1) * SQUARE_SIZE;
            Color color = (env->grid[r*env->n_cols + c] == 0) ? BG_COLOR : TETROMINOES_COLORS[env->grid[r*env->n_cols + c] - 1];
            DrawRectangle(
                x + HALF_LINEWIDTH, 
                y + HALF_LINEWIDTH,
                SQUARE_SIZE-2*HALF_LINEWIDTH, SQUARE_SIZE-2*HALF_LINEWIDTH, color
            );
            DrawRectangle(
                x - HALF_LINEWIDTH, 
                y - HALF_LINEWIDTH,
                SQUARE_SIZE, 2*HALF_LINEWIDTH, DASH_COLOR
            );
            DrawRectangle(
                x - HALF_LINEWIDTH, 
                y + SQUARE_SIZE- HALF_LINEWIDTH,
                SQUARE_SIZE, 2*HALF_LINEWIDTH, DASH_COLOR
            );
            DrawRectangle(
                x - HALF_LINEWIDTH, 
                y - HALF_LINEWIDTH,
                2*HALF_LINEWIDTH, + SQUARE_SIZE, DASH_COLOR
            );
            DrawRectangle(
                x + SQUARE_SIZE - HALF_LINEWIDTH, 
                y + SQUARE_SIZE- HALF_LINEWIDTH,
                2*HALF_LINEWIDTH, + SQUARE_SIZE, DASH_COLOR
            );
        }         
    }

    DrawRectangle(
        0, 
        5 * SQUARE_SIZE - HALF_LINEWIDTH,
        env->n_cols * SQUARE_SIZE, 2*HALF_LINEWIDTH, WHITE
    );

    DrawRectangle(
        0, 
        SQUARE_SIZE - HALF_LINEWIDTH,
        env->n_cols * SQUARE_SIZE, 2*HALF_LINEWIDTH, WHITE
    );
    // Draw UI
    DrawText(TextFormat("Score: %i", env->score), 16, 16, 40, (Color) {255, 160, 160, 255});
    EndDrawing();

}