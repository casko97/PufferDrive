#pragma once

#include "raylib.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

const unsigned char FORWARD = 0;
const unsigned char TURN_LEFT = 1;
const unsigned char TURN_RIGHT = 2;
const unsigned char SHOOT = 3;

const float FRICTION = 0.95f;
const float SPEED = 0.6f;
const float PARTICLE_SPEED = 7.0f;
const float ROTATION_SPEED = 0.1f;
const float SHOOT_DELAY = 0.3f;

const int MAX_PARTICLES = 50;

const int DEBUG = 1;

// Only use floats!
typedef struct {
  float score;
  float n; // Required as the last field
} Log;

typedef struct {
  Vector2 position;
  Vector2 velocity;
} Particle;

typedef struct {
  Vector2 position;
  Vector2 velocity;
  int radius;
} Asteroid;

typedef struct {
  Log log;
  unsigned char *observations;
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
  struct timeval last_shot;
} Asteroids;

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

void c_reset(Asteroids *env) {
  env->player_position = (Vector2){env->size / 2.0f, env->size / 2.0f};
  env->player_angle = 0.5f;
  env->player_radius = 12;
  env->thruster_on = 0;
  env->particle_index = 0;
  env->particle_index = 0.0f;
}

void c_step(Asteroids *env) {
  env->rewards[0] = 0;
  env->terminals[0] = 0;
  env->thruster_on = 0;

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
    env->particles[env->particle_index] = (Particle){env->player_position, dir};
  }

  // Explicit Euler
  env->player_position.x += env->player_vel.x;
  env->player_position.y += env->player_vel.y;

  move_particles(env);

  if (env->player_position.x < 0)
    env->player_position.x = env->size;
  if (env->player_position.y < 0)
    env->player_position.y = env->size;
  if (env->player_position.x > env->size)
    env->player_position.x = 0;
  if (env->player_position.y > env->size)
    env->player_position.y = 0;
}

void draw_player(Asteroids *env) {
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

void c_render(Asteroids *env) {
  if (!IsWindowReady()) {
    InitWindow(env->size, env->size, "PufferLib Asteroids");
    SetTargetFPS(60);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(BLACK);
  draw_player(env);
  draw_particles(env);
  EndDrawing();
}

void c_close(Asteroids *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
