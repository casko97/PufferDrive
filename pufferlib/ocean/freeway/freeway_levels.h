#define BASE_ROAD_SPEED 1.0f/13.0f // inverse number of seconds to go from the left to the right (slowest enemy)
#define MULT_ROAD_SPEED 1.39 // Factor of increase for the road speed (approximates the ratio between min speed and max speed of lvl1 (13/2.5)^(1/5))
#define BASE_PLAYER_SPEED 1.0f/3.5f // inverse number of seconds to go from the bottom to the top of the screen for the player
#define MAX_ENEMIES_PER_LANE 3
#define NUM_LEVELS 2

const int HUMAN_HIGH_SCORE[] = {25, 12};

const int ENEMIES_IS_ENABLED[] = {
    // Level 0
    1, 0, 0, // lane 0 to 9
    1, 0, 0,
    1, 0, 0,
    1, 0, 0,
    1, 0, 0,
    1, 0, 0,
    1, 0, 0,
    1, 0, 0,
    1, 0, 0,   
    1, 0, 0,
    // Level 1
    1, 0, 0, // lane 0 to 9
    1, 1, 0,
    1, 1, 0,
    1, 1, 1,
    1, 1, 0,
    1, 0, 0,
    1, 1, 0,
    1, 1, 1,
    1, 1, 0, 
    1, 1, 0,
};
const float ENEMIES_TYPES[] = {
    // Level 0
    0, 0, 0, // lane 0 to 9
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,

    // Level 1
    0, 0, 0, // lane 0 to 9
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    1, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
};

const float ENEMIES_INITIAL_X[] = {
    // Level 0
    0.0, 0.0, 0.0, // lane 0 to 9
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    // Level 1
    0.0, 0.0, 0.0, // lane 0 to 9
    0.0, 0.1, 0.0,
    0.0, 0.2, 0.0,
    0.0, 0.1, 0.2,
    0.0, 0.4, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.4, 0.0,
    0.0, 0.1, 0.2,
    0.0, 0.2, 0.0, 
    0.0, 0.1, 0.0, 
};

const float ENEMIES_INITIAL_VX[] = {
    // Level 0
    BASE_ROAD_SPEED,// lane 0 to 9
    MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    BASE_ROAD_SPEED,

    // Level 1
    BASE_ROAD_SPEED, // lane 0 to 9
    MULT_ROAD_SPEED*BASE_ROAD_SPEED, 
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    MULT_ROAD_SPEED*BASE_ROAD_SPEED,
    BASE_ROAD_SPEED,

};