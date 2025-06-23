#pragma once

#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define MAX_PARTICLES 10
#define MAX_ASTEROIDS 20

const unsigned char FORWARD = 0;
const unsigned char TURN_LEFT = 1;
const unsigned char TURN_RIGHT = 2;
const unsigned char SHOOT = 3;

const float FRICTION = 0.95f;
const float SPEED = 0.6f;
const float PARTICLE_SPEED = 7.0f;
const float ROTATION_SPEED = 0.1f;
const float ASTEROID_SPEED = 3.0f;
const float SHOOT_DELAY = 0.3f;

const int MAX_TICK = 3600;

const int DEBUG = 0;

// for render only game over state
static int global_game_over_timer = 0;
static int global_game_over_started = 0;

typedef struct {
  float perf;
  float score;
  float episode_return;
  float episode_length;
  float n;
} Log;

typedef struct {
  Vector2 position;
  Vector2 velocity;
} Particle;

typedef struct {
  Vector2 position;
  Vector2 velocity;
  int radius;
  Vector2 shape[12];
  int num_vertices;
} Asteroid;

typedef struct {
  Log log;
  float *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int size;
  Vector2 player_position;
  Vector2 player_vel;
  float player_angle;
  int player_radius;
  int thruster_on;
  Particle particles[MAX_PARTICLES];
  int particle_index;
  Asteroid asteroids[MAX_ASTEROIDS];
  int asteroid_index;
  struct timeval last_shot;
  int tick;
  int score;
} Asteroids;

float random_float(float low, float high) {
  return low + (high - low) * ((float)rand() / (float)RAND_MAX);
}

void generate_asteroid_shape(Asteroid *as) {
  as->num_vertices = 8 + (as->radius / 10);

  for (int v = 0; v < as->num_vertices; v++) {
    float angle = (2.0f * PI * v) / as->num_vertices;
    float radius_variation =
        as->radius * (0.7f + 0.6f * random_float(0.0f, 1.0f));
    as->shape[v].x = cosf(angle) * radius_variation;
    as->shape[v].y = sinf(angle) * radius_variation;
  }
}

float clamp(float val, float low, float high) {
  return fmin(fmax(val, low), high);
}

Vector2 rotate_vector(Vector2 point, Vector2 center, float angle) {
  float s = sinf(angle);
  float c = cosf(angle);

  // Translate point back to origin:
  point.x -= center.x;
  point.y -= center.y;

  // Rotate point
  float xnew = point.x * c - point.y * s;
  float ynew = point.x * s + point.y * c;

  // Translate point back:
  point.x = xnew + center.x;
  point.y = ynew + center.y;
  return point;
}

Vector2 get_direction_vector(Asteroids *env) {
  float px = env->player_position.x;
  float py = env->player_position.y;
  Vector2 dir = (Vector2){px, py - 1};
  dir = rotate_vector(dir, env->player_position, env->player_angle);
  return (Vector2){dir.x - px, dir.y - py};
}

void move_particles(Asteroids *env) {
  Particle p;
  for (int i = 0; i < MAX_PARTICLES; i++) {
    p = env->particles[i];
    p.position.x += p.velocity.x * PARTICLE_SPEED;
    p.position.y += p.velocity.y * PARTICLE_SPEED;
    env->particles[i] = p;
  }
}

void move_asteroids(Asteroids *env) {
  Asteroid as;
  for (int i = 0; i < MAX_ASTEROIDS; i++) {
    as = env->asteroids[i];
    if (as.radius == 0)
      continue;

    as.position.x += as.velocity.x * ASTEROID_SPEED;
    as.position.y += as.velocity.y * ASTEROID_SPEED;
    env->asteroids[i] = as;
  }
}

Vector2 angle_to_vector(float angle) {
  Vector2 v;
  v.x = cosf(angle);
  v.y = sinf(angle);
  return v;
}

void spawn_asteroids(Asteroids *env) {
  float px, py;
  float angle;
  if (rand() % 10 == 0) {
    switch (rand() % 4) {
    case 0:
      // left edge
      px = 0;
      py = rand() % env->size;
      angle = random_float(-PI / 2, PI / 2);
      break;
    case 1:
      // right edge
      px = env->size;
      py = rand() % env->size;
      angle = random_float(PI / 2, 3 * PI / 2);
      break;
    case 2:
      // top edge
      px = rand() % env->size;
      py = 0;
      angle = random_float(PI, 2 * PI);
      break;
    default:
      // bottom edge
      px = rand() % env->size;
      py = env->size;
      angle = random_float(0, PI);
      break;
    }

    Vector2 direction = angle_to_vector(angle);
    Vector2 start_pos = (Vector2){px, py};
    Asteroid as;
    switch (rand() % 3) {
    case 0:
      // small
      as = (Asteroid){start_pos, direction, 10};
      break;
    case 1:
      // medium
      as = (Asteroid){start_pos, direction, 20};
      break;
    default:
      // big
      as = (Asteroid){start_pos, direction, 40};
      break;
    }
    env->asteroid_index = (env->asteroid_index + 1) % MAX_ASTEROIDS;
    env->asteroids[env->asteroid_index] = as;
    generate_asteroid_shape(&env->asteroids[env->asteroid_index]);
  }
}

int particle_asteroid_collision(Asteroids *env, Particle p, Asteroid as) {
  float dx = p.position.x - as.position.x;
  float dy = p.position.y - as.position.y;
  return as.radius * as.radius > dx * dx + dy * dy;
}

void split_asteroid(Asteroids *env, int size, int j) {
  Asteroid as = env->asteroids[j];
  int new_radius = size == 40 ? 20 : 10;

  float original_angle = atan2f(as.velocity.y, as.velocity.x);

  float offset1 = random_float(-PI / 4, PI / 4);
  float offset2 = random_float(-PI / 4, PI / 4);

  float angle1 = original_angle + offset1;
  float angle2 = original_angle + offset2;

  Vector2 direction1 = angle_to_vector(angle1);
  Vector2 direction2 = angle_to_vector(angle2);

  float len1 = sqrtf(direction1.x * direction1.x + direction1.y * direction1.y);
  float len2 = sqrtf(direction2.x * direction2.x + direction2.y * direction2.y);
  if (len1 > 0) {
    direction1.x /= len1;
    direction1.y /= len1;
  }
  if (len2 > 0) {
    direction2.x /= len2;
    direction2.y /= len2;
  }

  Vector2 start_pos = (Vector2){as.position.x, as.position.y};

  int new_index1 = (env->asteroid_index + 1) % MAX_ASTEROIDS;
  int new_index2 = (new_index1 + 1) % MAX_ASTEROIDS;

  env->asteroids[j] = (Asteroid){start_pos, direction1, new_radius};
  env->asteroids[new_index1] = (Asteroid){start_pos, direction2, new_radius};
  env->asteroid_index = new_index2;

  // Generate shapes for the new asteroids
  generate_asteroid_shape(&env->asteroids[j]);
  generate_asteroid_shape(&env->asteroids[new_index1]);
}

void check_particle_asteroid_collision(Asteroids *env) {
  Particle p;
  Asteroid as;
  for (int i = 0; i < MAX_PARTICLES; i++) {
    p = env->particles[i];
    if (p.position.x == 0 && p.position.y == 0)
      continue;

    for (int j = 0; j < MAX_ASTEROIDS; j++) {
      as = env->asteroids[j];
      if (as.radius == 0)
        continue;

      if (particle_asteroid_collision(env, p, as)) {
        env->particles[i] = (Particle){};

        switch (as.radius) {
        case 10:
          env->score += 1;
          env->rewards[0] += 1.0f;
          env->asteroids[j] = (Asteroid){};
          break;
        case 20:
          env->score += 1;
          env->rewards[0] += 1.0f;
          split_asteroid(env, as.radius, j);
          break;
        default:
          env->score += 1;
          env->rewards[0] += 1.0f;
          split_asteroid(env, as.radius, j);
          break;
        }
        break;
      }
    }
  }
}

void check_player_asteroid_collision(Asteroids *env) {
  float min_dist;
  float dx, dy;
  Asteroid as;
  for (int i = 0; i < MAX_ASTEROIDS; i++) {
    as = env->asteroids[i];
    if (as.radius == 0)
      continue;

    min_dist = env->player_radius + as.radius;
    dx = env->player_position.x - as.position.x;
    dy = env->player_position.y - as.position.y;
    if (min_dist * min_dist > dx * dx + dy * dy) {
      env->terminals[0] = 1;
      return;
    }
  }
}

void compute_observations(Asteroids *env) {
  int observation_indx = 0;
  env->observations[observation_indx++] = env->player_position.x / env->size;
  env->observations[observation_indx++] = env->player_position.y / env->size;
  env->observations[observation_indx++] = env->player_vel.x;
  env->observations[observation_indx++] = env->player_vel.y;
  Asteroid as;
  for (int i = 0; i < MAX_ASTEROIDS; i++) {
    as = env->asteroids[i];
    if (as.radius == 0)
      continue;
    env->observations[observation_indx++] =
        (as.position.x - env->player_position.x) / env->size;
    env->observations[observation_indx++] =
        (as.position.y - env->player_position.y) / env->size;
  }
}

void add_log(Asteroids *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

void c_reset(Asteroids *env) {
  env->player_position = (Vector2){env->size / 2.0f, env->size / 2.0f};
  env->player_angle = 0.0f;
  env->player_radius = 12;
  env->player_vel = (Vector2){0, 0};
  env->thruster_on = 0;
  memset(env->particles, 0, sizeof(Particle) * MAX_PARTICLES);
  memset(env->asteroids, 0, sizeof(Asteroid) * MAX_ASTEROIDS);
  env->particle_index = 0;
  env->asteroid_index = 0;
  env->tick = 0;
  env->score = 0;
  gettimeofday(&env->last_shot, NULL);
}

void c_step(Asteroids *env) {
  env->rewards[0] = 0;
  env->terminals[0] = 0;
  env->thruster_on = 0;
  env->tick += 1;

  if (global_game_over_timer > 0)
    return;

  // slow down each step
  env->player_vel.x *= FRICTION;
  env->player_vel.y *= FRICTION;

  Vector2 dir = get_direction_vector(env);

  int action = env->actions[0];
  if (action == TURN_LEFT)
    env->player_angle -= ROTATION_SPEED;
  if (action == TURN_RIGHT)
    env->player_angle += ROTATION_SPEED;
  if (action == FORWARD) {
    env->player_vel.x += dir.x * SPEED;
    env->player_vel.y += dir.y * SPEED;
    env->thruster_on = 1;
  }

  struct timeval now;
  struct timeval start = env->last_shot;
  gettimeofday(&now, NULL);

  double elapsed =
      (now.tv_sec - start.tv_sec) + (now.tv_usec - start.tv_usec) / 1e6;
  if (action == SHOOT && elapsed >= SHOOT_DELAY) {
    gettimeofday(&env->last_shot, NULL);
    env->particle_index = (env->particle_index + 1) % MAX_PARTICLES;
    Vector2 start_pos = (Vector2){env->player_position.x + 20 * dir.x,
                                  env->player_position.y + 20 * dir.y};
    env->particles[env->particle_index] = (Particle){start_pos, dir};
  }

  // Explicit Euler
  env->player_position.x += env->player_vel.x;
  env->player_position.y += env->player_vel.y;

  move_particles(env);
  spawn_asteroids(env);
  move_asteroids(env);
  check_particle_asteroid_collision(env);
  check_player_asteroid_collision(env);

  if (env->player_position.x < 0)
    env->player_position.x = env->size;
  if (env->player_position.y < 0)
    env->player_position.y = env->size;
  if (env->player_position.x > env->size)
    env->player_position.x = 0;
  if (env->player_position.y > env->size)
    env->player_position.y = 0;

  if (env->terminals[0] == 1 || env->tick > MAX_TICK) {
    env->terminals[0] = 1;
    add_log(env);
    c_reset(env);
    return;
  }

  env->rewards[0] = env->score;
  compute_observations(env);
}

void draw_player(Asteroids *env) {
  if (global_game_over_timer > 0)
    return;

  float px = env->player_position.x;
  float py = env->player_position.y;

  if (DEBUG) {
    DrawPixel(px, py, RED);
    Vector2 dir = get_direction_vector(env);
    dir = (Vector2){dir.x * 10.0f, dir.y * 10.f};
    Vector2 t = (Vector2){dir.x + px, dir.y + py};
    DrawLineV(env->player_position, t, RED);
    DrawCircleLines(px, py, env->player_radius, RED);
  }

  Vector2 ps[8];

  // ship
  ps[0] = (Vector2){px - 10, py + 10};
  ps[1] = (Vector2){px + 10, py + 10};
  ps[2] = (Vector2){px, py - 20};
  ps[3] = (Vector2){px - 9, py + 6};
  ps[4] = (Vector2){px + 9, py + 6};
  ps[5] = (Vector2){px - 5, py + 6};
  ps[6] = (Vector2){px + 5, py + 6};
  ps[7] = (Vector2){px, py + 14};

  for (int i = 0; i < 8; i++)
    ps[i] = rotate_vector(ps[i], env->player_position, env->player_angle);

  DrawLineV(ps[0], ps[2], RAYWHITE);
  DrawLineV(ps[1], ps[2], RAYWHITE);

  DrawLineV(ps[3], ps[4], RAYWHITE);

  if (env->thruster_on) {
    DrawLineV(ps[5], ps[7], RAYWHITE);
    DrawLineV(ps[6], ps[7], RAYWHITE);
  }
}

void draw_particles(Asteroids *env) {
  for (int i = 0; i < MAX_PARTICLES; i++) {
    DrawPixel(env->particles[i].position.x, env->particles[i].position.y,
              RAYWHITE);
  }
}

void draw_asteroids(Asteroids *env) {
  Asteroid as;
  for (int i = 0; i < MAX_ASTEROIDS; i++) {
    as = env->asteroids[i];
    if (as.radius == 0)
      continue;

    if (DEBUG)
      DrawCircleLines(as.position.x, as.position.y, as.radius, RED);

    for (int v = 0; v < as.num_vertices; v++) {
      int next_v = (v + 1) % as.num_vertices;
      Vector2 pos1 = {as.position.x + as.shape[v].x,
                      as.position.y + as.shape[v].y};
      Vector2 pos2 = {as.position.x + as.shape[next_v].x,
                      as.position.y + as.shape[next_v].y};
      DrawLineV(pos1, pos2, (Color){245, 245, 245, 175});
    }
  }
}

void c_render(Asteroids *env) {
  if (!IsWindowReady()) {
    InitWindow(env->size, env->size, "PufferLib Asteroids");
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    SetTargetFPS(60);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  if (env->terminals[0] == 1 && !global_game_over_started) {
    global_game_over_started = 1;
    global_game_over_timer = 120;
  }

  if (global_game_over_timer > 0) {
    global_game_over_timer--;
  } else {
    global_game_over_started = 0;
  }

  BeginDrawing();
  ClearBackground(BLACK);
  draw_player(env);
  draw_particles(env);
  draw_asteroids(env);

  DrawText(TextFormat("Score: %d", env->score), 10, 10, 20, RAYWHITE);
  DrawText(TextFormat("%d s", (int)(env->tick / 60)), env->size - 40, 10, 20,
           RAYWHITE);

  if (global_game_over_timer > 0) {
    const char *game_over_text = "GAME OVER";
    int text_width = MeasureText(game_over_text, 40);
    int x = (env->size - text_width) / 2;
    int y = env->size / 2 - 20;

    float alpha = (float)global_game_over_timer / 120.0f;
    int alpha_value = (int)(alpha * 255);

    Color text_color = ColorAlpha(RED, alpha_value);
    DrawTextEx(GetFontDefault(), game_over_text, (Vector2){x, y}, 40, 2,
               text_color);
  }

  EndDrawing();
}

void c_close(Asteroids *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
