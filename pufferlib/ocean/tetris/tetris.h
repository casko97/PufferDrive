#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include "tetrominoes.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#define HALF_LINEWIDTH 1
#define SQUARE_SIZE 32
#define MAX_STEPS 400

const int REWARDS[5] = {0, 40, 100, 300, 1200};
const int REWARD_INVALID_ACTION = -100;

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float avg_combo;
    float episode_length;
    float lines_deleted;
    float n;
} Log;

typedef struct Client {
    int total_cols;
    int total_rows;
    int ui_rows;
    int deck_rows;
    int preview_rows;
    int preview_target_rotation;
    int preview_target_col;
} Client;

typedef struct Tetris {
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
    int deck_size;
    int* action_mask;
    int* row_is_free;
    int* grid;
    int* tetromino_deck;
    int current_position_in_deck;
    int current_tetromino;
    
    int score;
    int ep_return;
    int lines_deleted;
    int combos;
} Tetris;

void init(Tetris* env) {
    env->action_mask = (int*)calloc(env->n_cols * NUM_ROTATIONS, sizeof(int));
    env->row_is_free = (int*)calloc(env->n_rows, sizeof(int));
    env->grid = (int*)calloc(env->n_rows * env->n_cols, sizeof(int));
    env->tetromino_deck = calloc(env->deck_size, sizeof(int));
}

void allocate(Tetris* env) {
    init(env);
    env->observations = (float*)calloc(NUM_ROTATIONS * SIZE * SIZE + env->n_cols*env->n_rows + NUM_TETROMINOES * (env->deck_size -1) + 1 + NUM_ROTATIONS * env->n_cols, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void c_close(Tetris* env) {
    free(env->grid);
    free(env->tetromino_deck);
    free(env->action_mask);
    free(env->row_is_free);
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
    env->log.lines_deleted += env->lines_deleted;
    env->log.avg_combo += env->combos > 0 ? ((float)env->lines_deleted) / ((float)env->combos) : 1.0f;
    env->log.n += 1;
}

void compute_observations(Tetris* env) {
    // first tetromino, with the 4 possible rotations
    for (int target_rotation = 0; target_rotation < NUM_ROTATIONS; target_rotation++){
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                env->observations[target_rotation * SIZE * SIZE + r * SIZE + c] = TETROMINOES[env->current_tetromino][target_rotation][r][c];
            }
        }
    }
    int offset = NUM_ROTATIONS * SIZE * SIZE;
    // content of the grid
    for (int i = 0; i < env->n_cols*env->n_rows; i++) {
        env->observations[offset+i] = env->grid[i];
    }

    // other tetrominoes, one hot endoded
    int tetromino_id;
    offset = NUM_ROTATIONS * SIZE * SIZE + env->n_cols*env->n_rows;
    for (int j = 1; j<env->deck_size; j++){
        tetromino_id = env->tetromino_deck[(env->current_position_in_deck+j)%env->deck_size];
        for (int i = 0; i < NUM_TETROMINOES; i++) {
            env->observations[offset + j*NUM_TETROMINOES + i] = 0;
        }
        env->observations[offset + j*NUM_TETROMINOES + tetromino_id] = 1; 
    }
    env->observations[0] = env->score;

    // action mask
    offset = NUM_ROTATIONS * SIZE * SIZE + env->n_cols*env->n_rows + NUM_TETROMINOES * (env->deck_size -1) + 1;
    for (int i = 0; i < env->n_cols*NUM_ROTATIONS; i++) {
        env->observations[offset+i] = env->action_mask[i];
    }

}

void restore_grid(Tetris* env) {
    for (int r = 0; r < env->n_rows; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            env->grid[r * env->n_cols + c] = 0;
        }
    }
    for (int r = 0; r < SIZE; r++) {
        env->row_is_free[r] = 1;
    }
}

void initialize_deck(Tetris* env) {
    for (int i = 0; i < env->deck_size; i++) {
        env->tetromino_deck[i] = rand() % NUM_TETROMINOES;
    }
    env->current_position_in_deck = 0;
    env->current_tetromino = env->tetromino_deck[env->current_position_in_deck];
}

void update_deck(Tetris* env) {
    env->tetromino_deck[env->current_position_in_deck] = rand() % NUM_TETROMINOES;
    env->current_position_in_deck = (env->current_position_in_deck +1) % env->deck_size;
    env->current_tetromino = env->tetromino_deck[env->current_position_in_deck];
}

int get_landing_row(Tetris* env, int tetromino_idx, int target_col, int target_rotation) {
    int landing_row = -1;
    if (target_col + TETROMINOES_FILLS_COLUMN[tetromino_idx][target_rotation] > env->n_cols){
        return -1;
    }
    for (int r = TETROMINOES_FILLS_ROW[tetromino_idx][target_rotation]; r < env->n_cols; r++) {
        if (env->row_is_free[r]){
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
                if (TETROMINOES[tetromino_idx][target_rotation][r][c] == 1) {
                    int grid_row = test_row + r;
                    int grid_col = target_col + c;
                    
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

bool is_full_row(Tetris* env, int row){
    for (int c = 0; c < env->n_cols; c++){
        if (env->grid[row * env->n_cols + c] == 0){
            return false;
        } 
    }
    return true;
}

void clear_row(Tetris* env, int row){
    for (int r = row; r > 0; r--){
        for (int c = 0; c < env->n_cols; c++){
            env->grid[r * env->n_cols + c] = env->grid[ (r-1) * env->n_cols + c];
        }
    }
    for (int c = 0; c < env->n_cols; c++){
        env->grid[c] = 0; 
    }
}

int place_tetromino(Tetris* env, int tetromino_idx, int col, int rotation, int landing_row) {
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            if (TETROMINOES[tetromino_idx][rotation][r][c] == 1) {
                int grid_row = landing_row + r;
                int grid_col = col + c;

                if (grid_row < env->n_rows && grid_col >= 0 && grid_col < env->n_cols) {
                    env->grid[grid_row * env->n_cols + grid_col] = tetromino_idx + 1;
                    env->row_is_free[grid_row] = 0;
                }
            }
        }
    }
    int lines_deleted = 0;
    int row_to_check = landing_row + TETROMINOES_FILLS_ROW[tetromino_idx][rotation] - 1;
    for (int r = 0; r < TETROMINOES_FILLS_ROW[tetromino_idx][rotation]; r++){
        if (is_full_row(env, row_to_check)){
            clear_row(env, row_to_check);
            lines_deleted+=1;
        }
        else{
            row_to_check-=1;
        }
    }
    return lines_deleted;
}

void compute_action_mask(Tetris* env, int tetromino_idx) {
    int landing_row;

    if (tetromino_idx == 0){
        for (int c = 0; c < env->n_cols; c++) {
            landing_row = get_landing_row(env, tetromino_idx, c, 0);
            for (int rot = 0; rot < NUM_ROTATIONS; rot++) {
                env->action_mask[rot * env->n_cols + c] = landing_row;
            }
        }
    }

    else {
        for (int c = 0; c < env->n_cols; c++) {
            for (int rot = 0; rot < NUM_ROTATIONS; rot++) {
                landing_row = get_landing_row(env, tetromino_idx, c, rot);
                env->action_mask[rot * env->n_cols + c] = landing_row;
            }
        }
    }
}

void c_reset(Tetris* env) {
    env->score = 0;
    env->ep_return = 0;
    env->step = 0;
    env->lines_deleted = 0;
    env->combos = 0;
    restore_grid(env);
    initialize_deck(env);
    compute_action_mask(env, env->current_tetromino);
    compute_observations(env);
}

bool is_game_done(Tetris* env){
    for (int i = 0; i < NUM_ROTATIONS * env->n_cols; i++) {
        if (env->action_mask[i] > -1) {
            return false;
        }
    }
    return true;
}

void c_step(Tetris* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
    env->step += 1;
    int action = env->actions[0];
    int col = action%env->n_cols;
    int rotation = action/env->n_cols;
    int lines_deleted = 0;
    int landing_row;

    if (env->action_mask[action] > -1) {
        landing_row = get_landing_row(env, env->current_tetromino, col, rotation);
        lines_deleted = place_tetromino(env, env->current_tetromino, col, rotation, landing_row);
    
        if (lines_deleted>0){
            env->score += REWARDS[lines_deleted];
            env->lines_deleted += lines_deleted;
            env->combos += 1;
            env->rewards[0] = REWARDS[lines_deleted];
            env->ep_return+= REWARDS[lines_deleted];
        }

        update_deck(env);
        compute_action_mask(env, env->current_tetromino);
    }
    else{
        env->rewards[0] = REWARD_INVALID_ACTION;
        env->ep_return += REWARD_INVALID_ACTION;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
    
    if (is_game_done(env) || (env->step > MAX_STEPS)){
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
    else{
        compute_observations(env);
    };
}



Client* make_client(Tetris* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->ui_rows = 1;
    client->deck_rows = SIZE;
    client->preview_rows = SIZE;
    client->total_rows = 1 + client->ui_rows + 1 + client->deck_rows + 1 + client->preview_rows + 1 + env->n_cols + 1;
    client->total_cols = max(1 + env->n_rows + 1, 1 + 3 * (env->deck_size - 1));
    client->preview_target_col = env->n_cols/2;
    client->preview_target_rotation = 0;
    InitWindow(SQUARE_SIZE * client->total_cols, SQUARE_SIZE * client->total_rows, "PufferLib Tetris");
    SetTargetFPS(3);
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

Color BORDER_COLOR = (Color){100, 100, 100, 255};
Color DASH_COLOR = (Color){80, 80, 80, 255};
Color DASH_COLOR_BRIGHT = (Color){150, 150, 150, 255};
Color DASH_COLOR_DARK = (Color){50, 50, 50, 255};

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
    ClearBackground(BLACK);
    int x, y;
    Color color;
    // outer grid
    for (int r = 0; r < client->total_rows; r++) {
        for (int c = 0; c < client->total_cols; c++) {
            x = c * SQUARE_SIZE;
            y = r * SQUARE_SIZE;
            if (
                    (c == 0) || (c == client->total_cols - 1) ||
                    ((r >= 1 + client->ui_rows + 1) && (r < 1 + client->ui_rows + 1 + client->deck_rows))|| 
                    ((r >= 1 + client->ui_rows + 1 + client->deck_rows + 1) && (c>=env->n_rows))|| 
                    (r == 0) || (r == 1 + client->ui_rows) || 
                    (r == 1 + client->ui_rows + 1 + client->deck_rows) || 
                    (r == 1 + client->ui_rows + 1 + client->deck_rows + 1 + client->preview_rows) || 
                    (r == client->total_rows - 1)
                ){
                DrawRectangle(
                    x + HALF_LINEWIDTH, 
                    y + HALF_LINEWIDTH,
                    SQUARE_SIZE-2*HALF_LINEWIDTH, SQUARE_SIZE-2*HALF_LINEWIDTH, BORDER_COLOR
                );
                DrawRectangle(
                    x - HALF_LINEWIDTH, 
                    y - HALF_LINEWIDTH,
                    SQUARE_SIZE, 2*HALF_LINEWIDTH, DASH_COLOR_DARK
                );
                DrawRectangle(
                    x - HALF_LINEWIDTH, 
                    y + SQUARE_SIZE- HALF_LINEWIDTH,
                    SQUARE_SIZE, 2*HALF_LINEWIDTH, DASH_COLOR_DARK
                );
                DrawRectangle(
                    x - HALF_LINEWIDTH, 
                    y - HALF_LINEWIDTH,
                    2*HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR_DARK
                );
                DrawRectangle(
                    x + SQUARE_SIZE - HALF_LINEWIDTH, 
                    y - HALF_LINEWIDTH,
                    2*HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR_DARK
                );
            }
        }         
    }
    // lower grid
    for (int r = 0; r < env->n_rows; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            x = (c + 1) * SQUARE_SIZE;
            y = (1 + client->ui_rows + 1 + client->deck_rows + 1 + client->preview_rows + 1 + r) * SQUARE_SIZE;
            color = (env->grid[r*env->n_cols + c] == 0) ? BLACK : TETROMINOES_COLORS[env->grid[r*env->n_cols + c] - 1];
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
                2*HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR
            );
            DrawRectangle(
                x + SQUARE_SIZE - HALF_LINEWIDTH, 
                y - HALF_LINEWIDTH,
                2*HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR
            );
        }         
    }
    // upper grid
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < env->n_cols; c++) {
            x = (c + 1) * SQUARE_SIZE;
            y = (1 + client->ui_rows + 1 + client->deck_rows  + 1 + r) * SQUARE_SIZE;
            color = (
                (c<client->preview_target_col) || 
                c>= (client->preview_target_col + SIZE) || 
                TETROMINOES[env->current_tetromino][client->preview_target_rotation][r][c-client->preview_target_col] == 0
            ) ?  BLACK : TETROMINOES_COLORS[env->current_tetromino];
               
            DrawRectangle(
                x + HALF_LINEWIDTH, 
                y + HALF_LINEWIDTH,
                SQUARE_SIZE-2*HALF_LINEWIDTH, SQUARE_SIZE-2*HALF_LINEWIDTH, color
            );
            DrawRectangle(
                x - HALF_LINEWIDTH, 
                y - HALF_LINEWIDTH,
                SQUARE_SIZE, 2*HALF_LINEWIDTH, DASH_COLOR_BRIGHT
            );
            DrawRectangle(
                x - HALF_LINEWIDTH, 
                y + SQUARE_SIZE- HALF_LINEWIDTH,
                SQUARE_SIZE, 2*HALF_LINEWIDTH, DASH_COLOR_BRIGHT
            );
            DrawRectangle(
                x - HALF_LINEWIDTH, 
                y - HALF_LINEWIDTH,
                2*HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR_BRIGHT
            );
            DrawRectangle(
                x + SQUARE_SIZE - HALF_LINEWIDTH, 
                y - HALF_LINEWIDTH,
                2*HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR_BRIGHT
            );
        }         
    }
    // Deck grid
    int tetromino_id;
    for (int i = 0; i < env->deck_size - 1; i ++){
        tetromino_id = env->tetromino_deck[(env->current_position_in_deck + 1 + i)%env->deck_size];
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < 2; c++) {
                x = (c + 1 + 3 * i) * SQUARE_SIZE;
                y = (1 + client->ui_rows + 1 + r) * SQUARE_SIZE;
                int r_offset = (SIZE - TETROMINOES_FILLS_ROW[tetromino_id][0]);
                if (r < r_offset){
                    color = BLACK;
                }
                else{
                    color = (TETROMINOES[tetromino_id][0][r - r_offset][c] == 0) ? BLACK : TETROMINOES_COLORS[tetromino_id];
                }
                DrawRectangle(
                    x + HALF_LINEWIDTH, 
                    y + HALF_LINEWIDTH,
                    SQUARE_SIZE-2*HALF_LINEWIDTH, SQUARE_SIZE-2*HALF_LINEWIDTH, color
                );
                DrawRectangle(
                    x - HALF_LINEWIDTH, 
                    y - HALF_LINEWIDTH,
                    SQUARE_SIZE, 2*HALF_LINEWIDTH, DASH_COLOR_BRIGHT
                );
                DrawRectangle(
                    x - HALF_LINEWIDTH, 
                    y + SQUARE_SIZE- HALF_LINEWIDTH,
                    SQUARE_SIZE, 2*HALF_LINEWIDTH, DASH_COLOR_BRIGHT
                );
                DrawRectangle(
                    x - HALF_LINEWIDTH, 
                    y - HALF_LINEWIDTH,
                    2*HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR_BRIGHT
                );
                DrawRectangle(
                    x + SQUARE_SIZE - HALF_LINEWIDTH, 
                    y - HALF_LINEWIDTH,
                    2*HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR_BRIGHT
                );
            }   
        }
    }     

    // Draw UI
    DrawText(TextFormat("Score: %i", env->score), SQUARE_SIZE+4, SQUARE_SIZE+4, 30, (Color) {255, 160, 160, 255});
    EndDrawing();
}