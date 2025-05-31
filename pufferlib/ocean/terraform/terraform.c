#include "terraform.h"

void allocate(Terraform* env) {
    env->observations = (unsigned char*)calloc(env->num_agents*246, sizeof(unsigned char));
    env->actions = (int*)calloc(3*env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    init(env);
}

void free_allocated(Terraform* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free_initialized(env);
}

void handle_camera_controls(Client* client) {
    static Vector2 prev_mouse_pos = {0};
    static bool is_dragging = false;
    float camera_move_speed = 0.5f;

    // Handle mouse drag for camera movement
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        prev_mouse_pos = GetMousePosition();
        is_dragging = true;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        is_dragging = false;
    }

    if (is_dragging) {
        Vector2 current_mouse_pos = GetMousePosition();
        Vector2 delta = {
            -(current_mouse_pos.x - prev_mouse_pos.x) * camera_move_speed,
            (current_mouse_pos.y - prev_mouse_pos.y) * camera_move_speed
        };

        // Apply 45-degree rotation to the movement
        // For a -45 degree rotation (clockwise)
        float cos45 = -0.7071f;  // cos(-45°)
        float sin45 = 0.7071f; // sin(-45°)
        Vector2 rotated_delta = {
            delta.x * cos45 - delta.y * sin45,
            delta.x * sin45 + delta.y * cos45
        };

        // Update camera position (only X and Y)
        client->camera.position.z += rotated_delta.x;
        client->camera.position.x += rotated_delta.y;

        // Update camera target (only X and Y)
        client->camera.target.z += rotated_delta.x;
        client->camera.target.x += rotated_delta.y;

        prev_mouse_pos = current_mouse_pos;
    }

    // Handle mouse wheel for zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        float zoom_factor = 1.0f - (wheel * 0.1f);
        // Calculate the current direction vector from target to position
        Vector3 direction = {
            client->camera.position.x - client->camera.target.x,
            client->camera.position.y - client->camera.target.y,
            client->camera.position.z - client->camera.target.z
        };

        // Scale the direction vector by the zoom factor
        direction.x *= zoom_factor;
        direction.y *= zoom_factor;
        direction.z *= zoom_factor;

        // Update the camera position based on the scaled direction
        client->camera.position.x = client->camera.target.x + direction.x;
        client->camera.position.y = client->camera.target.y + direction.y;
        client->camera.position.z = client->camera.target.z + direction.z;
    }
}

void demo() {
    //Weights* weights = load_weights("resources/pong_weights.bin", 133764);
    //LinearLSTM* net = make_linearlstm(weights, 1, 8, 3);

    Terraform env = {.size = 512, .num_agents = 8};
    allocate(&env);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        handle_camera_controls(env.client);
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[3*i] = 4; //rand() % 5;
            env.actions[3*i + 1] = rand() % 5;
            env.actions[3*i + 2] = rand() % 3;
        }
        env.actions[0] = 2;
        env.actions[1] = 2;
        env.actions[2] = 0;
        if (IsKeyDown(KEY_UP)    || IsKeyPressed(KEY_W)) env.actions[0] = 4;
        if (IsKeyDown(KEY_DOWN)  || IsKeyPressed(KEY_S)) env.actions[0] = 0;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[1] = 4;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[1] = 0;
        if (IsKeyDown(KEY_SPACE)) env.actions[2] = 1;
        if (IsKeyPressed(KEY_LEFT_SHIFT)) {
            env.actions[2] = 2;
        }
        DrawText(TextFormat("Bucket load: %f", env.dozers[0].load), 10, 80, 20, WHITE);

        c_step(&env);
        c_render(&env);
    }
    //free_linearlstm(net);
    //free(weights);
    free_allocated(&env);
}

void test_performance(int timeout) {
    Terraform env = {
        .size = 128,
        .num_agents = 8,
    };
    allocate(&env);
    c_reset(&env);

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        for (int i = 0; i < env.num_agents; i++) {
            env.actions[3*i] = rand() % 5;
            env.actions[3*i + 1] = rand() % 5;
            env.actions[3*i + 2] = rand() % 3;
        }

        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free_allocated(&env);
}

int main() {
    //test_performance(10);
    demo();
}

