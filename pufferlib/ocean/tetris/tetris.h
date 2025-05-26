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
    int* grid;
    int score;
    float ep_return;
};

void init(Tetris* env) {
    env->grid = (int*)calloc(env->n_rows * env->n_cols, sizeof(int));
}

void allocate(Tetris* env) {
    init(env);
    env->observations = (float*)calloc(2, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void c_close(Tetris* env) {
    free(env->grid);
}

void restore_grid(Tetris* env) {
    for (int r = 0; r < env->n_rows; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            env->grid[r * env->n_cols + c] = 0;
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

void place_tetromino(Tetris* env, int tetrimino_idx, int col, int rotation) {
    int row = 0;
    for (int r = 0; r < env->n_rows; r++) {
        if (env->grid[r * env->n_cols + col] == 0) {
            row = r;
            break;
        }
    }
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (TETROMINOES[tetrimino_idx][rotation][r][c] == 1) {
                if (row + r < env->n_rows && col + c < env->n_cols) {
                    env->grid[(row + r) * env->n_cols + (col + c)] = tetrimino_idx + 1;
                }
            }
        }
    }
}

void compute_observations(Tetris* env) {
    env->observations[0] = env->score;
    env->observations[1] = env->step;
}

void c_reset(Tetris* env) {
    env->score = 0.0f;
    env->ep_return = 0.0;
    env->step = 0;
    restore_grid(env);
    place_tetromino(env, rand() % NUM_TETROMINOES, 3, 0);
    compute_observations(env);
}

void c_step(Tetris* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
    compute_observations(env);
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

    for (int r = env->n_rows; r < env->n_rows + 4; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            int x = (env->n_cols - c -1) * SQUARE_SIZE;
            int y = (env->n_rows + 5 - r -1) * SQUARE_SIZE;
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
            int x = (env->n_cols - c -1) * SQUARE_SIZE;
            int y = (env->n_rows + 5 - r -1) * SQUARE_SIZE;
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