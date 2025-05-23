#include "raylib.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PX_PADDING_TOP 40         // 40px padding on top of the window

#define TICK_RATE 1.0f / 60.0f

#define LOG_BUFFER_SIZE 1024

#define MAX_STEPS 10000              // max steps before truncation

#define GHOST_OBSERVATIONS_COUNT 9
#define PLAYER_OBSERVATIONS_COUNT 11
#define NUM_GHOSTS 4

#define NUM_DOTS 244
#define OBSERVATIONS_COUNT (PLAYER_OBSERVATIONS_COUNT + GHOST_OBSERVATIONS_COUNT * NUM_GHOSTS + NUM_DOTS)

#define PINKY_CORNER (IVector2){3, -3}
#define BLINKY_CORNER (IVector2){MAP_WIDTH - 4, -3}
#define INKY_CORNER (IVector2){MAP_WIDTH - 1, MAP_HEIGHT}
#define CLYDE_CORNER (IVector2){0, MAP_HEIGHT}

#define PINKY_TARGET_LEAD 4
#define INKY_TARGET_LEAD 2
#define CLYDE_TARGET_RADIUS 8

typedef enum Direction {
  DOWN = 0,
  UP = 1,
  RIGHT = 2,
  LEFT = 3,
} Direction;

typedef struct Log Log;
struct Log {
  float episode_return;
  float episode_length;
  float score;
  float n;
};

typedef enum Tile {
  WALL_TILE = '#',
  DOT_TILE = '.',
  POWER_TILE = 'x',
  PLAYER_TILE = 'p',
  EMPTY_TILE = ' ',

  INKY_TILE = '1',
  BLINKY_TILE = '2',
  PINKY_TILE = '3',
  CLYDE_TILE = '4',
} Tile;

#define MAP_HEIGHT 31
#define MAP_WIDTH 28

static const char original_map[MAP_HEIGHT][MAP_WIDTH] = {
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#x####.#####.##.#####.####x#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "######.##### ## #####.######",
    "######.##   1234   ##.######",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "      .   ########   .      ",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "######.##          ##.######",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#x..##.......p .......##..x#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################",
};

typedef struct IVector2 {
  int x;
  int y;
} IVector2;

typedef struct Ghost {
  IVector2 spawn_pos;
  IVector2 pos;
  IVector2 target;
  Direction direction;
  int start_timeout;     // randomized delay before moving
  bool frightened;      // whether the ghost is frightened
  bool return_to_spawn; // whether to return to spawn position
} Ghost;

typedef struct Client Client;
typedef struct PacmanEnv {
  Client* client;
  bool randomize_starting_position; // randomize player starting position
  int min_start_timeout; // randomized ghost delay range
  int max_start_timeout;
  int frightened_time;   // ghost frighten time
  int max_mode_changes;
  int scatter_mode_length;
  int chase_mode_length;

  float *observations;
  int *actions;
  float *rewards;
  char *terminals;
  Log log;

  int step_count;
  int score;

  Tile *game_map;

  int remaining_dots;
  Tile **dots;

  IVector2 player_pos;
  Direction player_direction;
  
  bool reverse_directions;
  bool scatter_mode;
  int mode_time_left;
  int mode_changes;

  int frightened_time_left;

  bool player_caught;

  Ghost pinky;
  Ghost blinky;
  Ghost inky;
  Ghost clyde;
} PacmanEnv;

void add_log(PacmanEnv* env) {
  env->log.score += env->score;
  env->log.episode_return += env->score;
  env->log.episode_length += env->step_count;
  env->log.n++;
}

void init(PacmanEnv *env) {
  int dot_count = 0;
  Tile tile;

  env->game_map = (Tile *)calloc(MAP_WIDTH * MAP_HEIGHT, sizeof(Tile));
  env->dots = (Tile **)calloc(NUM_DOTS, sizeof(Tile *));

  // collect dot pointers
  for (int y = 0; y < MAP_HEIGHT; y++) {
    for (int x = 0; x < MAP_WIDTH; x++) {
      tile = original_map[y][x];
      if (tile == DOT_TILE || tile == POWER_TILE) {
        env->dots[dot_count++] = &env->game_map[y * MAP_WIDTH + x];
      }
    }
  }
}

void allocate(PacmanEnv *env) {
  init(env);
  env->observations = (float *)calloc(OBSERVATIONS_COUNT, sizeof(float));
  env->actions = (int *)calloc(1, sizeof(int));
  env->rewards = (float *)calloc(1, sizeof(float));
  env->terminals = (char *)calloc(1, sizeof(char));
}

void c_close(PacmanEnv *env) {
  free(env->game_map);
  free(env->dots);
}

void free_allocated(PacmanEnv *env) {
  free(env->actions);
  free(env->observations);
  free(env->terminals);
  free(env->rewards);
  c_close(env);
}

bool vec2_equal(IVector2 a, IVector2 b) {
  return a.x == b.x && a.y == b.y;
}

int vec2_distance_squared(IVector2 pos, IVector2 target) {
  int dx = pos.x - target.x;
  int dy = pos.y - target.y;
  return dx * dx + dy * dy;
}

IVector2 vec2_move(IVector2 pos, Direction direction, int distance) {
  switch (direction) {
  case UP:
    pos.y -= distance;
    break;
  case DOWN:
    pos.y += distance;
    break;
  case LEFT:
    pos.x -= distance;
    break;
  case RIGHT:
    pos.x += distance;
    break;
  }

  return pos;
}

IVector2 vec2_move_wrapped(IVector2 pos, Direction direction, int distance) {
  pos = vec2_move(pos, direction, distance);
  pos.x = (pos.x + MAP_WIDTH) % MAP_WIDTH;
  return pos;
}

Tile *tile_at(PacmanEnv *env, IVector2 pos) {
  return &env->game_map[pos.y * MAP_WIDTH + pos.x];
}

bool can_move_in_direction(PacmanEnv *env, IVector2 pos, Direction direction) {
  return *tile_at(env, vec2_move_wrapped(pos, direction, 1)) != WALL_TILE;
}

void compute_ghost_observations(PacmanEnv *env, Ghost *ghost, float *observations, int i) {
  observations[i + 0] = ghost->pos.x / (float)MAP_WIDTH;
  observations[i + 1] = ghost->pos.y / (float)MAP_HEIGHT;
  observations[i + 2] = (float)(ghost->direction == UP);
  observations[i + 3] = (float)(ghost->direction == DOWN);
  observations[i + 4] = (float)(ghost->direction == LEFT);
  observations[i + 5] = (float)(ghost->direction == RIGHT);
  observations[i + 6] = (float)(!ghost->frightened && !ghost->return_to_spawn);
  observations[i + 7] = (float)(ghost->frightened);
  observations[i + 8] = (float)(ghost->return_to_spawn);
}

void compute_observations(PacmanEnv *env) {
  // player observations
  env->observations[0] = env->player_pos.x / (float)MAP_WIDTH;
  env->observations[1] = env->player_pos.y / (float)MAP_HEIGHT;
  env->observations[2] = env->player_direction == UP;
  env->observations[3] = env->player_direction == DOWN;
  env->observations[4] = env->player_direction == LEFT;
  env->observations[5] = env->player_direction == RIGHT;
  env->observations[6] = can_move_in_direction(env, env->player_pos, UP);
  env->observations[7] = can_move_in_direction(env, env->player_pos, DOWN);
  env->observations[8] = can_move_in_direction(env, env->player_pos, LEFT);
  env->observations[9] = can_move_in_direction(env, env->player_pos, RIGHT);
  env->observations[10] = env->frightened_time_left / (float)env->frightened_time;

  // ghost observations
  compute_ghost_observations(env, &env->pinky, env->observations, PLAYER_OBSERVATIONS_COUNT);
  compute_ghost_observations(env, &env->blinky, env->observations, PLAYER_OBSERVATIONS_COUNT + GHOST_OBSERVATIONS_COUNT);
  compute_ghost_observations(env, &env->inky, env->observations, PLAYER_OBSERVATIONS_COUNT + (GHOST_OBSERVATIONS_COUNT * 2));
  compute_ghost_observations(env, &env->clyde, env->observations, PLAYER_OBSERVATIONS_COUNT + (GHOST_OBSERVATIONS_COUNT * 3));

  // dot observations
  for (int i = 0; i < NUM_DOTS; i++) {
    env->observations[i + PLAYER_OBSERVATIONS_COUNT + (GHOST_OBSERVATIONS_COUNT * 4)] = *env->dots[i] != EMPTY_TILE;
  }
}

int rand_range(int min, int max) {
  if (min == max) {
    return min;
  }

  return min + (rand() % (max - min));
}

void reset_ghost(PacmanEnv *env, Ghost *ghost) {
  ghost->direction = UP;
  ghost->start_timeout = rand_range(env->min_start_timeout, env->max_start_timeout);
  ghost->frightened = false;
  ghost->return_to_spawn = false;
}

void reset_round(PacmanEnv *env) {
  Tile original_tile;
  Tile *map_tile;
  int player_randomizer = rand() % NUM_DOTS;

  env->scatter_mode = false;
  env->mode_time_left = 0;
  env->mode_changes = 0;
  env->frightened_time_left = 0;

  env->step_count = 0;
  env->remaining_dots = NUM_DOTS;

  env->player_direction = RIGHT;
  reset_ghost(env, &env->pinky);
  reset_ghost(env, &env->blinky);
  reset_ghost(env, &env->inky);
  reset_ghost(env, &env->clyde);

  for (int y = 0; y < MAP_HEIGHT; y++) {
    for (int x = 0; x < MAP_WIDTH; x++) {
      original_tile = original_map[y][x];
      map_tile = tile_at(env, (IVector2){x, y});

      *map_tile = EMPTY_TILE;

      switch (original_tile) {
      case PINKY_TILE:
        env->pinky.pos = (IVector2){x, y};
        env->pinky.spawn_pos = (IVector2){x, y};
        *map_tile = EMPTY_TILE;
        break;
      case BLINKY_TILE:
        env->blinky.pos = (IVector2){x, y};
        env->blinky.spawn_pos = (IVector2){x, y};
        *map_tile = EMPTY_TILE;
        break;
      case INKY_TILE:
        env->inky.pos = (IVector2){x, y};
        env->inky.spawn_pos = (IVector2){x, y};
        *map_tile = EMPTY_TILE;
        break;
      case CLYDE_TILE:
        env->clyde.pos = (IVector2){x, y};
        env->clyde.spawn_pos = (IVector2){x, y};
        *map_tile = EMPTY_TILE;
        break;
      case PLAYER_TILE:
        if (!env->randomize_starting_position) {
          env->player_pos = (IVector2){x, y};
        }
        *map_tile = EMPTY_TILE;
        break;
      case DOT_TILE:
        if (env->randomize_starting_position && player_randomizer-- == 0) {
          env->player_pos = (IVector2){x, y};
          *map_tile = EMPTY_TILE;
        } else {
          *map_tile = original_tile;
        }
        break;
      default:
        *map_tile = original_tile;
      }
    }
  }
}

void c_reset(PacmanEnv *env) {
  env->score = 0;
  reset_round(env);
  compute_observations(env);
}

void set_frightened(PacmanEnv *env) {
  env->frightened_time_left = env->frightened_time;
  env->reverse_directions = true;

  env->pinky.frightened = !env->pinky.return_to_spawn;
  env->blinky.frightened = !env->blinky.return_to_spawn;
  env->inky.frightened = !env->inky.return_to_spawn;
  env->clyde.frightened = !env->clyde.return_to_spawn;
}

void player_move(PacmanEnv *env, Direction action) {
  IVector2 new_pos = vec2_move_wrapped(env->player_pos, action, 1);
  Tile *new_tile = tile_at(env, new_pos);

  // if the player action is into a wall, move in the current direction
  if (*new_tile == WALL_TILE) {
    new_pos = vec2_move_wrapped(env->player_pos, env->player_direction, 1);
    new_tile = tile_at(env, new_pos);
  } else {
    env->player_direction = action;
  }

  if (*new_tile != WALL_TILE) {
    env->player_pos = new_pos;

    if (*new_tile == DOT_TILE) {
      env->score++;
      env->rewards[0] += 1.0f;

      env->remaining_dots--;
      *new_tile = EMPTY_TILE;
    } else if (*new_tile == POWER_TILE) {
      env->score += 5;
      env->rewards[0] += 1.0f;

      env->remaining_dots--;
      *new_tile = EMPTY_TILE;

      set_frightened(env);
    }   
  }
}

int ghost_movement_options(PacmanEnv *env, Ghost *ghost, Direction *directions) {
  int option_count = 0;

  if (ghost->direction != DOWN && can_move_in_direction(env, ghost->pos, UP)) {
    directions[option_count++] = UP;
  }
  if (ghost->direction != RIGHT && can_move_in_direction(env, ghost->pos, LEFT)) {
    directions[option_count++] = LEFT;
  }
  if (ghost->direction != UP && can_move_in_direction(env, ghost->pos, DOWN)) {
    directions[option_count++] = DOWN;
  }
  if (ghost->direction != LEFT && can_move_in_direction(env, ghost->pos, RIGHT)) {
    directions[option_count++] = RIGHT;
  }

  return option_count;
}

Direction random_direction(PacmanEnv *env, Ghost *ghost) {
  Direction directions[4];
  int option_count = ghost_movement_options(env, ghost, directions);
  if (option_count == 0) return ghost->direction;   // nowhere to go
  int random_index = rand() % option_count;
  return directions[random_index];
}

int min_index(int *array, int length) {
  int min_index = 0;
  for (int i = 1; i < length; i++) {
    if (array[i] < array[min_index]) {
      min_index = i;
    }
  }
  return min_index;
}

Direction direction_to_target(PacmanEnv *env, Ghost *ghost) {
  Direction directions[4];
  int distances[4];
  int option_count = ghost_movement_options(env, ghost, directions);

  if (option_count == 1) {
    return directions[0];
  }

  for (int i = 0; i < option_count; i++) {
    distances[i] = vec2_distance_squared(vec2_move(ghost->pos, directions[i], 1), ghost->target);
  }

  return directions[min_index(distances, option_count)];
}

void set_scatter_targets(PacmanEnv *env) {
  env->pinky.target = PINKY_CORNER;
  env->blinky.target = BLINKY_CORNER;
  env->inky.target = INKY_CORNER;
  env->clyde.target = CLYDE_CORNER;
}

void set_chase_targets(PacmanEnv *env) {
  int clyde_distance = vec2_distance_squared(env->player_pos, env->clyde.pos);

  env->blinky.target = env->player_pos;

  env->pinky.target = vec2_move(env->player_pos, env->player_direction, PINKY_TARGET_LEAD);
  
  env->inky.target = vec2_move(env->player_pos, env->player_direction, INKY_TARGET_LEAD);
  env->inky.target.x -= env->blinky.pos.x - env->inky.target.x;
  env->inky.target.y -= env->blinky.pos.y - env->inky.target.y;

  if (clyde_distance > CLYDE_TARGET_RADIUS * CLYDE_TARGET_RADIUS) {
    env->clyde.target = env->player_pos;
  } else {
    env->clyde.target = CLYDE_CORNER;
  }
}

bool check_collision(IVector2 a, IVector2 old_a, IVector2 b, IVector2 old_b) {
  return (a.x >= b.x - 1 && a.x <= b.x + 1 && a.y == b.y) || (a.y >= b.y - 1 && a.y <= b.y + 1 && a.x == b.x);
}

Direction reverse_direction(Direction direction) {
  switch (direction) {
  case UP:
    return DOWN;
  case DOWN:
    return UP;
  case LEFT:
    return RIGHT;
  case RIGHT:
    return LEFT;
  }
  return direction;
}

void ghost_move(PacmanEnv *env, Ghost *ghost, IVector2 old_player_pos) {
  IVector2 next_ghost_position;
  IVector2 old_ghost_position = ghost->pos;

  if (ghost->return_to_spawn) {
    ghost->target = ghost->spawn_pos;
  }
  
  if (env->reverse_directions && !ghost->return_to_spawn) {
    ghost->direction = reverse_direction(ghost->direction);
  } else if (ghost->frightened) {
    ghost->direction = random_direction(env, ghost);
  } else {
    ghost->direction = direction_to_target(env, ghost);
  }
  
  if (ghost->start_timeout > 0) {
    ghost->start_timeout--;
  } else {
    next_ghost_position = vec2_move_wrapped(ghost->pos, ghost->direction, 1);
    if (*tile_at(env, next_ghost_position) != WALL_TILE) {
        ghost->pos = next_ghost_position;
    }
  }
  
  if (ghost->return_to_spawn) {
    if (vec2_equal(ghost->pos, ghost->spawn_pos)) {
      ghost->return_to_spawn = false;
    }
  } else if (check_collision(ghost->pos, old_ghost_position, env->player_pos, old_player_pos)) {
    if (ghost->frightened) {
        ghost->frightened = false;
        ghost->return_to_spawn = true;
        
        env->rewards[0] += 1.0f;
    } else {
        env->player_caught = true;
    }
  }
}

void check_mode_change(PacmanEnv *env) {
  if (env->mode_changes > env->max_mode_changes) {
    return;
  }

  if (--env->mode_time_left <= 0) {
    env->scatter_mode = !env->scatter_mode;
    env->reverse_directions = true;
    env->mode_changes++;

    if (env->scatter_mode) {
      env->mode_time_left = env->scatter_mode_length;
    } else {
      env->mode_time_left = env->chase_mode_length;
    }
  }
}

void c_step(PacmanEnv *env) {
  IVector2 old_player_pos = env->player_pos;
  int action = env->actions[0];

  env->step_count += 1;
  env->terminals[0] = 0;
  env->rewards[0] = 0.0f;

  env->reverse_directions = false;
  env->player_caught = false;

  if (env->frightened_time_left > 0) {
    env->frightened_time_left--;
  } else {
    env->pinky.frightened = false;
    env->blinky.frightened = false;
    env->inky.frightened = false;
    env->clyde.frightened = false;
  }

  check_mode_change(env);
  if (env->scatter_mode) {
    set_scatter_targets(env);
  } else {
    set_chase_targets(env);
  }
  
  player_move(env, action);

  ghost_move(env, &env->pinky, old_player_pos);
  ghost_move(env, &env->blinky, old_player_pos);
  ghost_move(env, &env->inky, old_player_pos);
  ghost_move(env, &env->clyde, old_player_pos);
 
  compute_observations(env);

  if (env->player_caught || env->step_count >= MAX_STEPS || env->remaining_dots <= 0) {
    env->log.score = env->score;
    env->log.episode_return = env->score;
    env->log.episode_length = env->step_count;
    add_log(env);

    env->terminals[0] = 1;
    c_reset(env);
  }
}

typedef struct DirectionSprites {
  Texture2D up;
  Texture2D down;
  Texture2D left;
  Texture2D right;
} DirectionSprites;

typedef struct Client Client;
struct Client {
  int tile_size;
  int frame;
  
  float step_time;
  float time_accumulator;

  IVector2 previous_player_pos;
  IVector2 previous_pinky_pos;
  IVector2 previous_blinky_pos;
  IVector2 previous_inky_pos;
  IVector2 previous_clyde_pos;
  
  Texture2D tileset;
  Texture2D pacman;
  Texture2D frightened;
  DirectionSprites pinky;
  DirectionSprites blinky;
  DirectionSprites inky;
  DirectionSprites clyde;
  DirectionSprites eyes;
};

Vector2 lerp_ivector2(IVector2 a, IVector2 b, float progress) {
  if (abs(a.x - b.x) > 1) {
    b.x = a.x;
  }

  float a_x = (float)a.x;
  float a_y = (float)a.y;
  float b_x = (float)b.x;
  float b_y = (float)b.y;

  return (Vector2){a_x + (b_x - a_x) * progress, a_y + (b_y - a_y) * progress};
}

void draw_tiled(Client *client, Texture2D texture, Vector2 position, float rotation, bool flip_x, float source_width, float source_height) {
  Rectangle source = (Rectangle){0, 0, flip_x ? -source_width : source_width, source_height};

  DrawTexturePro(
      texture,
      source,
      (Rectangle){
        (position.x + 0.50f) * client->tile_size,
        (position.y + 0.50f) * client->tile_size + PX_PADDING_TOP,
        client->tile_size * 1.5f,
        client->tile_size * 1.5f
      },
      (Vector2){client->tile_size * 0.75f, client->tile_size * 0.75f},
      rotation,
      WHITE
    );
}

void draw_entity(Client *client, Texture2D texture, IVector2 previous_pos, IVector2 pos, float progress, float rotation, bool flip_x, float source_width, float source_height) {
  Vector2 position;

  if (pos.x == 0 && previous_pos.x == MAP_WIDTH - 1) {
    position = lerp_ivector2((IVector2){-1, previous_pos.y}, pos, progress);
    draw_tiled(client, texture, position, rotation, flip_x, source_width, source_height);

    position.x += (float)MAP_WIDTH;
    draw_tiled(client, texture, position, rotation, flip_x, source_width, source_height);
  } else if (previous_pos.x == 0 && pos.x == MAP_WIDTH - 1) {
    position = lerp_ivector2((IVector2){MAP_WIDTH, previous_pos.y}, pos, progress);
    draw_tiled(client, texture, position, rotation, flip_x, source_width, source_height);

    position.x -= (float)MAP_WIDTH;
    draw_tiled(client, texture, position, rotation, flip_x, source_width, source_height);
  } else {
    position = lerp_ivector2(previous_pos, pos, progress);
    draw_tiled(client, texture, position, rotation, flip_x, source_width, source_height);
  }
}

void update_interpolation(Client *client, PacmanEnv *env) {
  client->previous_player_pos = env->player_pos;
  client->previous_pinky_pos = env->pinky.pos;
  client->previous_blinky_pos = env->blinky.pos;
  client->previous_inky_pos = env->inky.pos;
  client->previous_clyde_pos = env->clyde.pos;
}

Client *make_client(PacmanEnv *env) {
  Client *client = (Client *)calloc(1, sizeof(Client));
  env->client = client;
  client->time_accumulator = 0.0f;

  client->tile_size = 20;
  client->step_time = 1.0f / 7.0f;

  update_interpolation(client, env);

  srand(time(NULL));

  InitWindow(client->tile_size * MAP_WIDTH, client->tile_size * MAP_HEIGHT + PX_PADDING_TOP, "PufferLib Pacman");
  SetTargetFPS(60);
  
  client->tileset = LoadTexture("resources/pacman/tileset.png");
  client->pacman = LoadTexture("resources/puffers_128.png");
  client->frightened = LoadTexture("resources/pacman/scared.png");

  client->pinky.up = LoadTexture("resources/pacman/pinky_up.png");
  client->pinky.down = LoadTexture("resources/pacman/pinky_down.png");
  client->pinky.left = LoadTexture("resources/pacman/pinky_left.png");
  client->pinky.right = LoadTexture("resources/pacman/pinky_right.png");

  client->blinky.up = LoadTexture("resources/pacman/blinky_up.png");
  client->blinky.down = LoadTexture("resources/pacman/blinky_down.png");
  client->blinky.left = LoadTexture("resources/pacman/blinky_left.png");
  client->blinky.right = LoadTexture("resources/pacman/blinky_right.png");

  client->inky.up = LoadTexture("resources/pacman/inky_up.png");
  client->inky.down = LoadTexture("resources/pacman/inky_down.png");
  client->inky.left = LoadTexture("resources/pacman/inky_left.png");
  client->inky.right = LoadTexture("resources/pacman/inky_right.png");

  client->clyde.up = LoadTexture("resources/pacman/clyde_up.png");
  client->clyde.down = LoadTexture("resources/pacman/clyde_down.png");
  client->clyde.left = LoadTexture("resources/pacman/clyde_left.png");
  client->clyde.right = LoadTexture("resources/pacman/clyde_right.png");

  client->eyes.up = LoadTexture("resources/pacman/eyes_up.png");
  client->eyes.down = LoadTexture("resources/pacman/eyes_down.png");
  client->eyes.left = LoadTexture("resources/pacman/eyes_left.png");
  client->eyes.right = LoadTexture("resources/pacman/eyes_right.png");

  return client;
}

#define WALL_COLOR (Color){33, 33, 255, 255}
#define DOT_COLOR (Color){255, 185, 176, 255}
#define PLAYER_COLOR (Color){255, 255, 0, 255}

#define PINKY_COLOR (Color){255, 185, 255, 255}
#define BLINKY_COLOR (Color){255, 0, 0, 255}
#define INKY_COLOR (Color){0, 255, 255, 255}
#define CLYDE_COLOR (Color){255, 185, 80, 255}

void render_ghost(Client *client, Ghost *ghost, DirectionSprites *sprites, IVector2 previous_pos, float progress) {
  Texture2D texture;
  
  if (ghost->frightened) {
    texture = client->frightened;
  } else {
    if (ghost->return_to_spawn) {
      sprites = &client->eyes;
    }

    switch (ghost->direction) {
    case UP:
      texture = sprites->up;
      break;
    case DOWN:
      texture = sprites->down;
      break;
    case LEFT:
      texture = sprites->left;
      break;
    case RIGHT:
      texture = sprites->right;
      break;
    }
  }

  draw_entity(client, texture, previous_pos, ghost->pos, progress, 0.0f, false, 16, 16);
}

void render_player(Client *client, PacmanEnv *env, float progress) {
  float rotation = 0.0f;

  if (env->player_direction == UP) {
    rotation = 270.0f;
  } else if (env->player_direction == DOWN) {
    rotation = 90.0f;
  }

  draw_entity(client, client->pacman, client->previous_player_pos, env->player_pos, progress, rotation, env->player_direction == LEFT, 128, 128);
}

bool is_wall(PacmanEnv *env, IVector2 pos) {
  if (pos.x < 0 || pos.x >= MAP_WIDTH || pos.y < 0 || pos.y >= MAP_HEIGHT) {
    return true;
  }

  return *tile_at(env, pos) == WALL_TILE;
}

#define TILE_N (1 << 0)
#define TILE_NE (1 << 1)
#define TILE_E (1 << 2)
#define TILE_SE (1 << 3)
#define TILE_S (1 << 4)
#define TILE_SW (1 << 5)
#define TILE_W (1 << 6)
#define TILE_NW (1 << 7)

int get_tile_index(PacmanEnv *env, IVector2 pos) {
  int tile_bits = 0;

  if (is_wall(env, (IVector2){pos.x, pos.y - 1}))     tile_bits |= TILE_N;
  if (is_wall(env, (IVector2){pos.x + 1, pos.y - 1})) tile_bits |= TILE_NE;
  if (is_wall(env, (IVector2){pos.x + 1, pos.y}))     tile_bits |= TILE_E;
  if (is_wall(env, (IVector2){pos.x + 1, pos.y + 1})) tile_bits |= TILE_SE;
  if (is_wall(env, (IVector2){pos.x, pos.y + 1}))     tile_bits |= TILE_S;
  if (is_wall(env, (IVector2){pos.x - 1, pos.y + 1})) tile_bits |= TILE_SW;
  if (is_wall(env, (IVector2){pos.x - 1, pos.y}))     tile_bits |= TILE_W;
  if (is_wall(env, (IVector2){pos.x - 1, pos.y - 1})) tile_bits |= TILE_NW;

  switch (tile_bits) {
    case TILE_NW | TILE_N | TILE_NE | TILE_W | TILE_E | TILE_SE:
    case TILE_NW | TILE_N | TILE_NE | TILE_W | TILE_E | TILE_SW:
    case TILE_NW | TILE_N | TILE_NE | TILE_W | TILE_E: return 13;
    
    case TILE_NE | TILE_E | TILE_SE | TILE_N | TILE_S | TILE_NW:
    case TILE_NE | TILE_E | TILE_SE | TILE_N | TILE_S | TILE_SW:
    case TILE_NE | TILE_E | TILE_SE | TILE_N | TILE_S: return 6;
    
    case TILE_SE | TILE_S | TILE_SW | TILE_E | TILE_W | TILE_NW:
    case TILE_SE | TILE_S | TILE_SW | TILE_E | TILE_W | TILE_NE:
    case TILE_SE |TILE_S | TILE_SW | TILE_E | TILE_W: return 1;
    
    case TILE_SW | TILE_W | TILE_NW | TILE_S | TILE_N | TILE_NE:
    case TILE_SW | TILE_W | TILE_NW | TILE_S | TILE_N | TILE_SE:
    case TILE_SW | TILE_W | TILE_NW | TILE_S | TILE_N: return 8;
    case TILE_S | TILE_E | TILE_SE: return 0;
    case TILE_S | TILE_W | TILE_SW: return 2;
    case TILE_N | TILE_E | TILE_NE: return 12;
    case TILE_N | TILE_W | TILE_NW: return 14;

    case 255 & ~TILE_NW: return 25;
    case 255 & ~TILE_NE: return 26;
    case 255 & ~TILE_SW: return 31;
    case 255 & ~TILE_SE: return 32;
  }

  return 7;
}

void render_tile(Client *client, PacmanEnv *env, IVector2 pos) {
  int tile_index = get_tile_index(env, pos);
  int tile_x = tile_index % 6;
  int tile_y = tile_index / 6;
  Rectangle source = (Rectangle){tile_x * 9, tile_y * 9, 8, 8};

  DrawTexturePro(
    client->tileset,
    source,
    (Rectangle){
      pos.x * client->tile_size,
      pos.y * client->tile_size + PX_PADDING_TOP,
      client->tile_size, client->tile_size
    },
    (Vector2){0, 0},
    0.0f,
    WHITE
  );
}

void render_map(Client *client, PacmanEnv *env) {
  for (int y = 0; y < MAP_HEIGHT; y++) {
    for (int x = 0; x < MAP_WIDTH; x++) {
      char tile = env->game_map[y * MAP_WIDTH + x];
      if (tile == WALL_TILE) {
        render_tile(client, env, (IVector2){x, y});
      } else if (tile == DOT_TILE) {
        float width = client->tile_size / 4.0f;
        float height = client->tile_size / 4.0f;
        DrawRectangle(x * client->tile_size + client->tile_size / 2.0f - width / 2.0f,
                      y * client->tile_size + client->tile_size / 2.0f - height / 2.0f + PX_PADDING_TOP,
                      width, height, DOT_COLOR);
      } else if (tile == POWER_TILE) {
        DrawCircle(x * client->tile_size + client->tile_size / 2.0f,
                   y * client->tile_size + client->tile_size / 2.0f + PX_PADDING_TOP,
                   client->tile_size / 3.0f, DOT_COLOR);
      }
    }
  }
}

void handle_input(PacmanEnv *env) {
  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }
  if (IsKeyPressed(KEY_TAB)) {
    ToggleFullscreen();
  }
}

void c_render(PacmanEnv *env) {
  if (env->client == NULL) {
    env->client = make_client(env);
  }
  Client* client = env->client;

  float progress = client->time_accumulator / client->step_time;
  update_interpolation(env->client, env);
  //float progress = client->frame / 8.0f;
  //client->frame = (client->frame + 1) % 8;

  handle_input(env);

  BeginDrawing();
  ClearBackground(BLACK);

  render_map(client, env);

  render_player(client, env, progress);
  render_ghost(client, &env->pinky, &client->pinky, client->previous_pinky_pos, progress);
  render_ghost(client, &env->blinky, &client->blinky, client->previous_blinky_pos, progress);
  render_ghost(client, &env->inky, &client->inky, client->previous_inky_pos, progress);
  render_ghost(client, &env->clyde, &client->clyde, client->previous_clyde_pos, progress);
    
  DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
  DrawText(TextFormat("Mode: %s", env->scatter_mode ? "Scatter" : "Chase"), 150, 10, 20, WHITE);

  EndDrawing();
}

void unload_direction_sprites(DirectionSprites *sprites) {
  UnloadTexture(sprites->up);
  UnloadTexture(sprites->down);
  UnloadTexture(sprites->left);
  UnloadTexture(sprites->right);
}

void close_client(Client *client) {
  CloseWindow();

  UnloadTexture(client->tileset);
  UnloadTexture(client->pacman);
  UnloadTexture(client->frightened);
  unload_direction_sprites(&client->pinky);
  unload_direction_sprites(&client->blinky);
  unload_direction_sprites(&client->inky);
  unload_direction_sprites(&client->clyde);
  unload_direction_sprites(&client->eyes);
  free(client);
}
