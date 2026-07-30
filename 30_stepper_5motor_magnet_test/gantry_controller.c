#include "gantry_controller.h"

#include "gantry_config.h"
#include "mos_switch.h"
#include "rpi_link.h"
#include "stepper_cm.h"
#include "stepper_cm_config.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TOKEN_COUNT_MAX 12U

typedef struct {
    int32_t sourceX10;
    int32_t sourceY10;
    int32_t targetX10;
    int32_t targetY10;
    int32_t angleMdeg;
    bool received;
} GantryJob;

typedef enum {
    STATE_IDLE = 0,
    STATE_WAIT_PLAN,
    STATE_START_ITEM,
    STATE_WAIT_PICK_XY,
    STATE_COMMAND_PICK_DOWN,
    STATE_WAIT_PICK_DOWN,
    STATE_WAIT_MAGNET_PICK,
    STATE_COMMAND_PICK_UP,
    STATE_WAIT_PICK_UP,
    STATE_COMMAND_PLACE_XY_ROTATE,
    STATE_WAIT_PLACE_XY_ROTATE,
    STATE_COMMAND_PLACE_DOWN,
    STATE_WAIT_PLACE_DOWN,
    STATE_WAIT_MAGNET_RELEASE,
    STATE_COMMAND_PLACE_UP,
    STATE_WAIT_PLACE_UP,
    STATE_COMMAND_ROTATE_ZERO,
    STATE_WAIT_ROTATE_ZERO,
    STATE_NEXT_ITEM,
    STATE_COMMAND_RETURN_HOME,
    STATE_WAIT_RETURN_HOME,
    STATE_WAIT_MANUAL_XY,
    STATE_ESTOP,
    STATE_FAULT
} ControllerState;

typedef enum {
    MOTION_FAILED = 0,
    MOTION_SKIPPED,
    MOTION_STARTED
} MotionStart;

typedef struct {
    ControllerState state;
    uint32_t deadlineMs;
    uint32_t nextRequestId;
    uint32_t requestId;
    uint32_t lastCompletedRequestId;
    uint8_t mode;
    uint8_t jobCount;
    uint8_t jobIndex;
    GantryJob jobs[GANTRY_MAX_PLAN_ITEMS];
    int32_t currentX10;
    int32_t currentY10;
    int32_t pendingX10;
    int32_t pendingY10;
    int32_t rotationMdeg;
    int32_t pendingRotationMdeg;
    bool positionKnown;
    bool zSafe;
    bool completionEvent;
    bool rejectionEvent;
} ControllerContext;

static ControllerContext g_controller;

static bool deadline_reached(uint32_t nowMs)
{
    return (int32_t)(nowMs - g_controller.deadlineMs) >= 0;
}

static uint32_t magnitude_i32(int32_t value)
{
    if (value >= 0) {
        return (uint32_t)value;
    }
    if (value == INT32_MIN) {
        return ((uint32_t)INT32_MAX + 1U);
    }
    return (uint32_t)(-value);
}

static uint64_t divide_round_up_u64(uint64_t numerator, uint64_t denominator)
{
    if (denominator == 0ULL) {
        return UINT64_MAX;
    }
    return (numerator + denominator - 1ULL) / denominator;
}

static uint32_t square_root_round_up_u64(uint64_t value)
{
    uint64_t original = value;
    uint64_t result = 0ULL;
    uint64_t bit = 1ULL << 62;

    while (bit > value) {
        bit >>= 2;
    }
    while (bit != 0ULL) {
        if (value >= (result + bit)) {
            value -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    if ((result * result) < original) {
        result++;
    }
    return (result > UINT32_MAX) ? UINT32_MAX : (uint32_t)result;
}

/* Emm V5 curve rule: changing speed by 1 RPM takes
 * (256 - acceleration) * 50 us.  The move is triangular when it is too short
 * to reach speedRpm, otherwise it is trapezoidal.  distanceMilliRev is motor
 * travel in 0.001 revolutions, so the same function supports linear and
 * rotary axes.  Every division and square root rounds upward.
 */
static uint32_t motion_profile_duration_ms(uint32_t distanceMilliRev,
                                           uint16_t speedRpm,
                                           uint8_t acceleration,
                                           uint32_t settleMs)
{
    uint64_t baseMs;
    uint64_t totalMs;

    if ((distanceMilliRev == 0U) || (speedRpm == 0U)) {
        return 0U;
    }

    if (acceleration == 0U) {
        baseMs = divide_round_up_u64((uint64_t)distanceMilliRev * 60ULL,
                                     (uint64_t)speedRpm);
    } else {
        uint32_t accelerationDifference = 256U - (uint32_t)acceleration;
        uint64_t rampDistanceNumerator =
            (uint64_t)speedRpm * (uint64_t)speedRpm *
            (uint64_t)accelerationDifference;

        if (((uint64_t)distanceMilliRev * 1200ULL) >=
            rampDistanceNumerator) {
            uint64_t constantSpeedEquivalentMs = divide_round_up_u64(
                (uint64_t)distanceMilliRev * 60ULL,
                (uint64_t)speedRpm);
            uint64_t accelerationMs = divide_round_up_u64(
                (uint64_t)speedRpm *
                    (uint64_t)accelerationDifference,
                20ULL);

            baseMs = constantSpeedEquivalentMs + accelerationMs;
        } else {
            uint64_t durationSquaredMs =
                (uint64_t)distanceMilliRev *
                (uint64_t)accelerationDifference * 12ULL;
            baseMs = square_root_round_up_u64(durationSquaredMs);
        }
    }

    totalMs = baseMs + (uint64_t)GANTRY_MOTION_MARGIN_MS +
              (uint64_t)settleMs;
    return (totalMs > UINT32_MAX) ? UINT32_MAX : (uint32_t)totalMs;
}

static uint32_t linear_profile_duration_ms(uint32_t distanceX10,
                                           uint16_t speedRpm,
                                           uint8_t acceleration,
                                           uint32_t settleMs)
{
    uint64_t distanceMilliRev;

    if (distanceX10 == 0U) {
        return 0U;
    }
    distanceMilliRev = divide_round_up_u64(
        (uint64_t)distanceX10 * 100ULL,
        (uint64_t)STEPPER_SLIDE_LEAD_MM);
    if (distanceMilliRev > UINT32_MAX) {
        return UINT32_MAX;
    }
    return motion_profile_duration_ms((uint32_t)distanceMilliRev,
                                      speedRpm, acceleration, settleMs);
}

static bool coordinate_is_reachable(int32_t x10, int32_t y10)
{
    return (x10 >= GANTRY_MIN_X_X10) && (x10 <= GANTRY_MAX_X_X10) &&
           (y10 >= GANTRY_MIN_Y_X10) && (y10 <= GANTRY_MAX_Y_X10);
}

static bool at_home(void)
{
    return g_controller.positionKnown &&
           (g_controller.currentX10 == GANTRY_HOME_X_X10) &&
           (g_controller.currentY10 == GANTRY_HOME_Y_X10);
}

static const char *state_name(ControllerState state)
{
    switch (state) {
        case STATE_IDLE: return "IDLE";
        case STATE_WAIT_PLAN: return "WAIT_PLAN";
        case STATE_START_ITEM: return "START_ITEM";
        case STATE_WAIT_PICK_XY: return "MOVE_PICK";
        case STATE_COMMAND_PICK_DOWN:
        case STATE_WAIT_PICK_DOWN: return "Z_DOWN_PICK";
        case STATE_WAIT_MAGNET_PICK: return "MAGNET_PICK";
        case STATE_COMMAND_PICK_UP:
        case STATE_WAIT_PICK_UP: return "Z_UP_PICK";
        case STATE_COMMAND_PLACE_XY_ROTATE:
        case STATE_WAIT_PLACE_XY_ROTATE: return "MOVE_PLACE_ROTATE";
        case STATE_COMMAND_PLACE_DOWN:
        case STATE_WAIT_PLACE_DOWN: return "Z_DOWN_PLACE";
        case STATE_WAIT_MAGNET_RELEASE: return "MAGNET_RELEASE";
        case STATE_COMMAND_PLACE_UP:
        case STATE_WAIT_PLACE_UP: return "Z_UP_PLACE";
        case STATE_COMMAND_ROTATE_ZERO:
        case STATE_WAIT_ROTATE_ZERO: return "ROTATE_ZERO";
        case STATE_NEXT_ITEM: return "NEXT_ITEM";
        case STATE_COMMAND_RETURN_HOME:
        case STATE_WAIT_RETURN_HOME: return "RETURN_HOME";
        case STATE_WAIT_MANUAL_XY: return "MOVE_MANUAL";
        case STATE_ESTOP: return "ESTOP";
        case STATE_FAULT: return "FAULT";
        default: return "UNKNOWN";
    }
}

static void send_state(const char *name)
{
    (void)RpiLink_SendFormat("STATE %lu %s ITEM %u",
                             (unsigned long)g_controller.requestId,
                             name,
                             (unsigned int)(g_controller.jobIndex + 1U));
}

static void clear_jobs(void)
{
    uint8_t index;

    for (index = 0U; index < GANTRY_MAX_PLAN_ITEMS; index++) {
        memset(&g_controller.jobs[index], 0, sizeof(g_controller.jobs[index]));
    }
    g_controller.jobCount = 0U;
    g_controller.jobIndex = 0U;
}

static void enter_fault(const char *code)
{
    uint32_t request = g_controller.requestId;

    /* Removing magnet power is deliberately the first fault action. */
    MOS_Switch_EmergencyOff();
    (void)StepperCm_StopAll();
    g_controller.positionKnown = false;
    g_controller.zSafe = false;
    g_controller.state = STATE_FAULT;
    (void)RpiLink_SendFormat("ERR %lu %s",
                             (unsigned long)request,
                             (code == NULL) ? "MOTION" : code);
}

static uint16_t proportional_speed(uint32_t distanceX10,
                                   uint32_t maximumX10)
{
    uint32_t speed;

    if ((distanceX10 == 0U) || (maximumX10 == 0U)) {
        return GANTRY_XY_MIN_SPEED_RPM;
    }
    speed = ((uint32_t)GANTRY_XY_MAX_SPEED_RPM * distanceX10 +
             (maximumX10 / 2U)) / maximumX10;
    if (speed < GANTRY_XY_MIN_SPEED_RPM) {
        speed = GANTRY_XY_MIN_SPEED_RPM;
    }
    if (speed > GANTRY_XY_MAX_SPEED_RPM) {
        speed = GANTRY_XY_MAX_SPEED_RPM;
    }
    return (uint16_t)speed;
}

static MotionStart start_xy_move(int32_t targetX10, int32_t targetY10,
                                 uint32_t nowMs)
{
    StepperCmProfiledMove moves[3];
    uint8_t count = 0U;
    int32_t deltaX10;
    int32_t deltaY10;
    uint32_t absX10;
    uint32_t absY10;
    uint32_t maximumX10;
    uint16_t xSpeed;
    uint16_t ySpeed;
    uint32_t xDuration;
    uint32_t yDuration;

    if (!g_controller.positionKnown || !g_controller.zSafe ||
        !coordinate_is_reachable(targetX10, targetY10)) {
        return MOTION_FAILED;
    }
    deltaX10 = targetX10 - g_controller.currentX10;
    deltaY10 = targetY10 - g_controller.currentY10;
    absX10 = magnitude_i32(deltaX10);
    absY10 = magnitude_i32(deltaY10);
    maximumX10 = (absX10 > absY10) ? absX10 : absY10;
    if (maximumX10 == 0U) {
        g_controller.pendingX10 = targetX10;
        g_controller.pendingY10 = targetY10;
        return MOTION_SKIPPED;
    }

    xSpeed = proportional_speed(absX10, maximumX10);
    ySpeed = proportional_speed(absY10, maximumX10);
    if (deltaX10 != 0) {
        moves[count].motor = STEPPER_MOTOR_1;
        /* Motor 1 positive moves left, opposite A4 +X. */
        moves[count].centimeters = -(float)deltaX10 / 100.0F;
        moves[count].speedRpm = xSpeed;
        moves[count].acceleration = GANTRY_XY_ACCELERATION;
        count++;
    }
    if (deltaY10 != 0) {
        float yCentimeters = (float)deltaY10 / 100.0F;

        moves[count].motor = STEPPER_MOTOR_2;
        moves[count].centimeters = yCentimeters;
        moves[count].speedRpm = ySpeed;
        moves[count].acceleration = GANTRY_XY_ACCELERATION;
        count++;
        moves[count].motor = STEPPER_MOTOR_3;
        moves[count].centimeters = yCentimeters;
        moves[count].speedRpm = ySpeed;
        moves[count].acceleration = GANTRY_XY_ACCELERATION;
        count++;
    }
    if (!StepperCm_MoveProfiledSynchronized(moves, count)) {
        return MOTION_FAILED;
    }

    xDuration = linear_profile_duration_ms(absX10, xSpeed,
                                           GANTRY_XY_ACCELERATION,
                                           GANTRY_XY_SETTLE_MS);
    yDuration = linear_profile_duration_ms(absY10, ySpeed,
                                           GANTRY_XY_ACCELERATION,
                                           GANTRY_XY_SETTLE_MS);
    g_controller.deadlineMs = nowMs +
        ((xDuration > yDuration) ? xDuration : yDuration);
    g_controller.pendingX10 = targetX10;
    g_controller.pendingY10 = targetY10;
    return MOTION_STARTED;
}

static MotionStart start_z_move(bool upward, uint32_t nowMs)
{
    uint32_t duration;

    if (!StepperCm_MoveWithParams(STEPPER_MOTOR_4,
                                  upward ? GANTRY_Z_TRAVEL_CM
                                         : -GANTRY_Z_TRAVEL_CM,
                                  GANTRY_Z_SPEED_RPM,
                                  GANTRY_Z_ACCELERATION)) {
        return MOTION_FAILED;
    }
    duration = linear_profile_duration_ms(80U, GANTRY_Z_SPEED_RPM,
                                          GANTRY_Z_ACCELERATION, 0U);
    g_controller.deadlineMs = nowMs + duration;
    return MOTION_STARTED;
}

static uint32_t rotation_duration_ms(int32_t angleMdeg)
{
    uint64_t numerator;
    uint64_t denominator;
    uint64_t distanceMilliRev;

    numerator = (uint64_t)magnitude_i32(angleMdeg) *
                (uint64_t)STEPPER_MOTOR_5_PULSES_PER_OUTPUT_REV;
    denominator = 360ULL * (uint64_t)STEPPER_PULSES_PER_REV;
    distanceMilliRev = divide_round_up_u64(numerator, denominator);
    if (distanceMilliRev > UINT32_MAX) {
        return UINT32_MAX;
    }
    return motion_profile_duration_ms((uint32_t)distanceMilliRev,
                                      STEPPER_MOTOR_5_SPEED_RPM,
                                      STEPPER_MOTOR_5_ACCELERATION, 0U);
}

static MotionStart start_rotation(int32_t relativeAngleMdeg, uint32_t nowMs)
{
    float commandDegrees;

    if (relativeAngleMdeg == 0) {
        g_controller.pendingRotationMdeg = g_controller.rotationMdeg;
        return MOTION_SKIPPED;
    }
    commandDegrees = (float)relativeAngleMdeg / 1000.0F;
    if (!Motor5_RotateDegWithParams(commandDegrees,
                                    STEPPER_MOTOR_5_SPEED_RPM,
                                    STEPPER_MOTOR_5_ACCELERATION)) {
        return MOTION_FAILED;
    }
    g_controller.pendingRotationMdeg =
        g_controller.rotationMdeg + relativeAngleMdeg;
    g_controller.deadlineMs = nowMs +
        rotation_duration_ms(relativeAngleMdeg);
    return MOTION_STARTED;
}

static MotionStart start_place_xy_and_rotation(const GantryJob *job,
                                               uint32_t nowMs)
{
    MotionStart rotationResult;
    MotionStart xyResult;
    uint32_t rotationDeadline = nowMs;
    uint32_t xyDeadline = nowMs;

    if (job == NULL) {
        return MOTION_FAILED;
    }

    /* Motor 5 is addressed independently, so start it first and then launch
     * the synchronized X/Y group.  Both motions run concurrently. */
    rotationResult = start_rotation(job->angleMdeg, nowMs);
    if (rotationResult == MOTION_FAILED) {
        return MOTION_FAILED;
    }
    if (rotationResult == MOTION_STARTED) {
        rotationDeadline = g_controller.deadlineMs;
    }

    xyResult = start_xy_move(job->targetX10, job->targetY10, nowMs);
    if (xyResult == MOTION_FAILED) {
        return MOTION_FAILED;
    }
    if (xyResult == MOTION_STARTED) {
        xyDeadline = g_controller.deadlineMs;
    }

    /* Z may descend only after the slower of XY and rotation has completed. */
    g_controller.deadlineMs =
        ((int32_t)(rotationDeadline - xyDeadline) > 0)
            ? rotationDeadline : xyDeadline;
    if ((rotationResult == MOTION_STARTED) ||
        (xyResult == MOTION_STARTED)) {
        return MOTION_STARTED;
    }
    return MOTION_SKIPPED;
}

static bool parse_u32(const char *text, uint32_t *value)
{
    char *end;
    unsigned long parsed;

    if ((text == NULL) || (value == NULL) || (*text == '\0') ||
        (*text == '-')) {
        return false;
    }
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if ((errno == ERANGE) || (*end != '\0') || (parsed > UINT32_MAX)) {
        return false;
    }
    *value = (uint32_t)parsed;
    return true;
}

static bool parse_i32(const char *text, int32_t *value)
{
    char *end;
    long parsed;

    if ((text == NULL) || (value == NULL) || (*text == '\0')) {
        return false;
    }
    errno = 0;
    parsed = strtol(text, &end, 10);
    if ((errno == ERANGE) || (*end != '\0') ||
        (parsed < INT32_MIN) || (parsed > INT32_MAX)) {
        return false;
    }
    *value = (int32_t)parsed;
    return true;
}

static uint8_t tokenize(char *line, char *tokens[TOKEN_COUNT_MAX])
{
    uint8_t count = 0U;
    char *token = strtok(line, " \t");

    while ((token != NULL) && (count < TOKEN_COUNT_MAX)) {
        tokens[count++] = token;
        token = strtok(NULL, " \t");
    }
    if (token != NULL) {
        return TOKEN_COUNT_MAX;
    }
    return count;
}

static void send_status(void)
{
    (void)RpiLink_SendFormat(
        "STATUS %s REQ %lu ITEM %u POS %ld %ld Z %s MAG %u ROT %ld KNOWN %u",
        state_name(g_controller.state),
        (unsigned long)g_controller.requestId,
        (unsigned int)(g_controller.jobIndex + 1U),
        (long)g_controller.currentX10,
        (long)g_controller.currentY10,
        g_controller.zSafe ? "SAFE" : "DOWN_OR_UNKNOWN",
        MOS_Switch_IsOn() ? 1U : 0U,
        (long)g_controller.rotationMdeg,
        g_controller.positionKnown ? 1U : 0U);
}

static void finish_request(void)
{
    uint32_t completed = g_controller.requestId;

    g_controller.lastCompletedRequestId = completed;
    g_controller.requestId = 0U;
    g_controller.mode = 0U;
    clear_jobs();
    g_controller.state = STATE_IDLE;
    g_controller.completionEvent = true;
    (void)RpiLink_SendFormat("DONE %lu", (unsigned long)completed);
}

static void begin_plan_run(uint32_t nowMs)
{
    (void)nowMs;
    g_controller.jobIndex = 0U;
    g_controller.state = STATE_START_ITEM;
    (void)RpiLink_SendFormat("ACK %lu COMMIT",
                             (unsigned long)g_controller.requestId);
}

void GantryController_Init(uint32_t nowMs)
{
    memset(&g_controller, 0, sizeof(g_controller));
    g_controller.state = STATE_IDLE;
    g_controller.deadlineMs = nowMs;
    g_controller.nextRequestId = 1U;
    g_controller.currentX10 = GANTRY_HOME_X_X10;
    g_controller.currentY10 = GANTRY_HOME_Y_X10;
    g_controller.pendingX10 = GANTRY_HOME_X_X10;
    g_controller.pendingY10 = GANTRY_HOME_Y_X10;
    g_controller.positionKnown = true;
    g_controller.zSafe = true;
    MOS_Switch_Off();

    (void)RpiLink_SendFormat(
        "READY GANTRY_V3_XY_ROTATE HOME %ld %ld RANGE %ld %ld %ld %ld",
        (long)GANTRY_HOME_X_X10, (long)GANTRY_HOME_Y_X10,
        (long)GANTRY_MIN_X_X10, (long)GANTRY_MAX_X_X10,
        (long)GANTRY_MIN_Y_X10, (long)GANTRY_MAX_Y_X10);
}

bool GantryController_IsIdle(void)
{
    return g_controller.state == STATE_IDLE;
}

bool GantryController_HasFault(void)
{
    return g_controller.state == STATE_FAULT;
}

bool GantryController_IsEmergencyStopped(void)
{
    return g_controller.state == STATE_ESTOP;
}

bool GantryController_TakeCompletionEvent(void)
{
    bool event = g_controller.completionEvent;

    g_controller.completionEvent = false;
    return event;
}

bool GantryController_TakeRejectionEvent(void)
{
    bool event = g_controller.rejectionEvent;

    g_controller.rejectionEvent = false;
    return event;
}

bool GantryController_RequestMode(uint8_t mode, uint32_t nowMs)
{
    if ((mode < 1U) || (mode > 3U)) {
        return false;
    }
    if (g_controller.state != STATE_IDLE) {
        (void)RpiLink_SendLine("ERR 0 BUSY");
        return false;
    }
    if (!at_home() || !g_controller.zSafe || MOS_Switch_IsOn() ||
        (g_controller.rotationMdeg != 0)) {
        (void)RpiLink_SendLine("ERR 0 NOT_HOME");
        return false;
    }

    clear_jobs();
    g_controller.requestId = g_controller.nextRequestId++;
    if (g_controller.nextRequestId == 0U) {
        g_controller.nextRequestId = 1U;
    }
    g_controller.mode = mode;
    g_controller.deadlineMs = nowMs + GANTRY_PLAN_TIMEOUT_MS;
    g_controller.state = STATE_WAIT_PLAN;
    (void)RpiLink_SendFormat("KEY %lu %u",
                             (unsigned long)g_controller.requestId,
                             (unsigned int)mode);
    return true;
}

static void process_plan(char *tokens[TOKEN_COUNT_MAX], uint8_t count,
                         uint32_t nowMs)
{
    uint32_t request;
    uint32_t mode;
    uint32_t itemCount;

    if ((count != 4U) || !parse_u32(tokens[1], &request) ||
        !parse_u32(tokens[2], &mode) ||
        !parse_u32(tokens[3], &itemCount)) {
        (void)RpiLink_SendLine("ERR 0 BAD_PLAN");
        return;
    }
    if ((request == 0U) || (mode < 1U) || (mode > 3U) ||
        (itemCount < GANTRY_MIN_PLAN_ITEMS) ||
        (itemCount > GANTRY_MAX_PLAN_ITEMS)) {
        (void)RpiLink_SendFormat("ERR %lu PLAN_SHAPE",
                                 (unsigned long)request);
        return;
    }

    /* A web request has no preceding physical KEY event.  It is accepted only
     * from the exact same safe start condition required by KEY1/KEY2.
     */
    if (g_controller.state == STATE_IDLE) {
        if (!at_home() || !g_controller.zSafe || MOS_Switch_IsOn() ||
            MOS_Switch_IsInhibited() || (g_controller.rotationMdeg != 0)) {
            (void)RpiLink_SendFormat("ERR %lu NOT_HOME",
                                     (unsigned long)request);
            return;
        }
        if (request == g_controller.lastCompletedRequestId) {
            (void)RpiLink_SendFormat("DONE %lu DUPLICATE",
                                     (unsigned long)request);
            return;
        }
        clear_jobs();
        g_controller.requestId = request;
        g_controller.mode = (uint8_t)mode;
        g_controller.deadlineMs = nowMs + GANTRY_PLAN_TIMEOUT_MS;
        g_controller.state = STATE_WAIT_PLAN;
    }
    if ((g_controller.state != STATE_WAIT_PLAN) ||
        (request != g_controller.requestId)) {
        (void)RpiLink_SendFormat("ERR %lu NOT_WAITING_PLAN",
                                 (unsigned long)request);
        return;
    }
    if (mode != g_controller.mode) {
        (void)RpiLink_SendFormat("ERR %lu PLAN_SHAPE",
                                 (unsigned long)request);
        return;
    }
    if (g_controller.jobCount == 0U) {
        clear_jobs();
        g_controller.jobCount = (uint8_t)itemCount;
    } else if (g_controller.jobCount != (uint8_t)itemCount) {
        (void)RpiLink_SendFormat("ERR %lu PLAN_CONFLICT",
                                 (unsigned long)request);
        return;
    }
    g_controller.deadlineMs = nowMs + GANTRY_PLAN_TIMEOUT_MS;
    (void)RpiLink_SendFormat("ACK %lu PLAN %lu",
                             (unsigned long)request,
                             (unsigned long)itemCount);
}

static void process_item(char *tokens[TOKEN_COUNT_MAX], uint8_t count,
                         uint32_t nowMs)
{
    GantryJob candidate;
    GantryJob *stored;
    uint32_t request;
    uint32_t index;

    memset(&candidate, 0, sizeof(candidate));
    if ((count != 8U) || !parse_u32(tokens[1], &request) ||
        !parse_u32(tokens[2], &index) ||
        !parse_i32(tokens[3], &candidate.sourceX10) ||
        !parse_i32(tokens[4], &candidate.sourceY10) ||
        !parse_i32(tokens[5], &candidate.targetX10) ||
        !parse_i32(tokens[6], &candidate.targetY10) ||
        !parse_i32(tokens[7], &candidate.angleMdeg)) {
        (void)RpiLink_SendLine("ERR 0 BAD_ITEM");
        return;
    }
    if ((g_controller.state != STATE_WAIT_PLAN) ||
        (request != g_controller.requestId) ||
        (g_controller.jobCount < GANTRY_MIN_PLAN_ITEMS) ||
        (g_controller.jobCount > GANTRY_MAX_PLAN_ITEMS)) {
        (void)RpiLink_SendFormat("ERR %lu ITEM_WITHOUT_PLAN",
                                 (unsigned long)request);
        return;
    }
    if (index >= g_controller.jobCount) {
        (void)RpiLink_SendFormat("ERR %lu ITEM_INDEX",
                                 (unsigned long)request);
        return;
    }
    if (!coordinate_is_reachable(candidate.sourceX10,
                                 candidate.sourceY10) ||
        !coordinate_is_reachable(candidate.targetX10,
                                 candidate.targetY10)) {
        (void)RpiLink_SendFormat("ERR %lu RANGE", (unsigned long)request);
        return;
    }
    /* Mode 1 normally preserves orientation, but the reachable upper band is
     * only about 77 mm high.  The Pi may therefore request +/-90 degrees for
     * a long piece so all four can be packed without overlap.  Both modes use
     * the same global rotation limit.
     */
    if ((candidate.angleMdeg < -GANTRY_MAX_ABS_ROTATION_MDEG) ||
        (candidate.angleMdeg > GANTRY_MAX_ABS_ROTATION_MDEG)) {
        (void)RpiLink_SendFormat("ERR %lu ANGLE", (unsigned long)request);
        return;
    }
    candidate.received = true;
    stored = &g_controller.jobs[index];
    if (stored->received) {
        (void)RpiLink_SendFormat("ERR %lu ITEM_DUPLICATE",
                                 (unsigned long)request);
        return;
    }
    *stored = candidate;
    g_controller.deadlineMs = nowMs + GANTRY_PLAN_TIMEOUT_MS;
    (void)RpiLink_SendFormat("ACK %lu ITEM %lu",
                             (unsigned long)request,
                             (unsigned long)index);
}

static void process_commit(char *tokens[TOKEN_COUNT_MAX], uint8_t count,
                           uint32_t nowMs)
{
    uint32_t request;
    uint8_t index;

    if ((count != 2U) || !parse_u32(tokens[1], &request)) {
        (void)RpiLink_SendLine("ERR 0 BAD_COMMIT");
        return;
    }
    if ((g_controller.state != STATE_WAIT_PLAN) ||
        (request != g_controller.requestId) ||
        (g_controller.jobCount < GANTRY_MIN_PLAN_ITEMS) ||
        (g_controller.jobCount > GANTRY_MAX_PLAN_ITEMS)) {
        (void)RpiLink_SendFormat("ERR %lu COMMIT_WITHOUT_PLAN",
                                 (unsigned long)request);
        return;
    }
    for (index = 0U; index < g_controller.jobCount; index++) {
        if (!g_controller.jobs[index].received) {
            (void)RpiLink_SendFormat("ERR %lu MISSING_ITEM %u",
                                     (unsigned long)request,
                                     (unsigned int)index);
            return;
        }
    }
    begin_plan_run(nowMs);
}

static void process_manual_move(char *tokens[TOKEN_COUNT_MAX], uint8_t count,
                                uint32_t nowMs, bool relative)
{
    MotionStart result;
    uint32_t request;
    int32_t x10;
    int32_t y10;

    if ((count != 4U) || !parse_u32(tokens[1], &request) ||
        !parse_i32(tokens[2], &x10) || !parse_i32(tokens[3], &y10) ||
        (request == 0U)) {
        (void)RpiLink_SendLine("ERR 0 BAD_MOVE");
        return;
    }
    if (g_controller.state != STATE_IDLE) {
        (void)RpiLink_SendFormat("ERR %lu BUSY", (unsigned long)request);
        return;
    }
    if (request == g_controller.lastCompletedRequestId) {
        (void)RpiLink_SendFormat("DONE %lu DUPLICATE",
                                 (unsigned long)request);
        return;
    }
    if (relative) {
        int64_t absoluteX10 = (int64_t)g_controller.currentX10 + x10;
        int64_t absoluteY10 = (int64_t)g_controller.currentY10 + y10;

        if (!g_controller.positionKnown || (absoluteX10 < INT32_MIN) ||
            (absoluteX10 > INT32_MAX) || (absoluteY10 < INT32_MIN) ||
            (absoluteY10 > INT32_MAX)) {
            (void)RpiLink_SendFormat("ERR %lu RANGE", (unsigned long)request);
            return;
        }
        x10 = (int32_t)absoluteX10;
        y10 = (int32_t)absoluteY10;
    }
    if (!coordinate_is_reachable(x10, y10)) {
        (void)RpiLink_SendFormat("ERR %lu RANGE", (unsigned long)request);
        return;
    }
    g_controller.requestId = request;
    g_controller.jobIndex = 0U;
    send_state("MOVE_MANUAL");
    result = start_xy_move(x10, y10, nowMs);
    if (result == MOTION_FAILED) {
        enter_fault("MOVE_TX");
    } else if (result == MOTION_SKIPPED) {
        g_controller.currentX10 = x10;
        g_controller.currentY10 = y10;
        finish_request();
    } else {
        g_controller.state = STATE_WAIT_MANUAL_XY;
    }
}

static void process_direct_pickplace(char *tokens[TOKEN_COUNT_MAX],
                                     uint8_t count, uint32_t nowMs)
{
    GantryJob job;
    uint32_t request;

    memset(&job, 0, sizeof(job));
    if ((count != 7U) || !parse_u32(tokens[1], &request) ||
        !parse_i32(tokens[2], &job.sourceX10) ||
        !parse_i32(tokens[3], &job.sourceY10) ||
        !parse_i32(tokens[4], &job.targetX10) ||
        !parse_i32(tokens[5], &job.targetY10) ||
        !parse_i32(tokens[6], &job.angleMdeg) || (request == 0U)) {
        (void)RpiLink_SendLine("ERR 0 BAD_PICKPLACE");
        return;
    }
    if (g_controller.state != STATE_IDLE) {
        (void)RpiLink_SendFormat("ERR %lu BUSY", (unsigned long)request);
        return;
    }
    if (!coordinate_is_reachable(job.sourceX10, job.sourceY10) ||
        !coordinate_is_reachable(job.targetX10, job.targetY10) ||
        (job.angleMdeg < -GANTRY_MAX_ABS_ROTATION_MDEG) ||
        (job.angleMdeg > GANTRY_MAX_ABS_ROTATION_MDEG)) {
        (void)RpiLink_SendFormat("ERR %lu RANGE_OR_ANGLE",
                                 (unsigned long)request);
        return;
    }
    job.received = true;
    clear_jobs();
    g_controller.jobs[0] = job;
    g_controller.jobCount = 1U;
    g_controller.jobIndex = 0U;
    g_controller.requestId = request;
    g_controller.mode = 0U;
    g_controller.state = STATE_START_ITEM;
    (void)RpiLink_SendFormat("ACK %lu PICKPLACE",
                             (unsigned long)request);
    (void)nowMs;
}

void GantryController_ProcessLine(char *line, uint32_t nowMs)
{
    char *tokens[TOKEN_COUNT_MAX];
    uint8_t count;

    if (line == NULL) {
        return;
    }
    count = tokenize(line, tokens);
    if (count == 0U) {
        return;
    }
    if (strcmp(tokens[0], "STOP") == 0) {
        if (count == 1U) {
            GantryController_EmergencyStop("UART_STOP");
        } else {
            (void)RpiLink_SendLine("ERR 0 BAD_STOP");
        }
    } else if (strcmp(tokens[0], "PING") == 0) {
        if (count == 1U) {
            if ((g_controller.state == STATE_IDLE) && at_home() &&
                g_controller.zSafe && !MOS_Switch_IsOn() &&
                (g_controller.rotationMdeg == 0)) {
                (void)RpiLink_SendLine("PONG READY");
            } else {
                (void)RpiLink_SendLine("PONG BUSY");
            }
        } else {
            (void)RpiLink_SendLine("ERR 0 BAD_PING");
        }
    } else if ((strcmp(tokens[0], "STATUS") == 0) && (count == 1U)) {
        send_status();
    } else if (strcmp(tokens[0], "PLAN") == 0) {
        process_plan(tokens, count, nowMs);
    } else if (strcmp(tokens[0], "ITEM") == 0) {
        process_item(tokens, count, nowMs);
    } else if (strcmp(tokens[0], "COMMIT") == 0) {
        process_commit(tokens, count, nowMs);
    } else if (strcmp(tokens[0], "REJECT") == 0) {
        uint32_t request;

        if ((count != 3U) || !parse_u32(tokens[1], &request)) {
            (void)RpiLink_SendLine("ERR 0 BAD_REJECT");
        } else if ((g_controller.state != STATE_WAIT_PLAN) ||
                   (request != g_controller.requestId)) {
            (void)RpiLink_SendFormat("ERR %lu REJECT_NOT_PENDING",
                                     (unsigned long)request);
        } else {
            clear_jobs();
            g_controller.requestId = 0U;
            g_controller.mode = 0U;
            g_controller.rejectionEvent = true;
            g_controller.state = STATE_IDLE;
            (void)RpiLink_SendFormat("ACK %lu REJECT %s",
                                     (unsigned long)request, tokens[2]);
        }
    } else if (strcmp(tokens[0], "MOVEA") == 0) {
        process_manual_move(tokens, count, nowMs, false);
    } else if (strcmp(tokens[0], "MOVER") == 0) {
        process_manual_move(tokens, count, nowMs, true);
    } else if (strcmp(tokens[0], "PICKPLACE") == 0) {
        process_direct_pickplace(tokens, count, nowMs);
    } else if ((strcmp(tokens[0], "HELP") == 0) && (count == 1U)) {
        (void)RpiLink_SendLine(
            "COMMANDS PING STATUS STOP PLAN ITEM COMMIT REJECT MOVEA MOVER PICKPLACE");
    } else {
        (void)RpiLink_SendLine("ERR 0 UNKNOWN_COMMAND");
    }
}

void GantryController_EmergencyStop(const char *reason)
{
    uint32_t request = g_controller.requestId;

    MOS_Switch_EmergencyOff();
    (void)StepperCm_StopAll();
    g_controller.positionKnown = false;
    g_controller.zSafe = false;
    g_controller.state = STATE_ESTOP;
    (void)RpiLink_SendFormat("ERR %lu STOPPED %s POSITION_UNKNOWN",
                             (unsigned long)request,
                             (reason == NULL) ? "EMERGENCY" : reason);
}

void GantryController_Update(uint32_t nowMs)
{
    GantryJob *job;
    MotionStart result;

    switch (g_controller.state) {
        case STATE_IDLE:
        case STATE_ESTOP:
        case STATE_FAULT:
            break;

        case STATE_WAIT_PLAN:
            if (deadline_reached(nowMs)) {
                uint32_t request = g_controller.requestId;

                clear_jobs();
                g_controller.requestId = 0U;
                g_controller.mode = 0U;
                g_controller.rejectionEvent = true;
                g_controller.state = STATE_IDLE;
                (void)RpiLink_SendFormat("ERR %lu PLAN_TIMEOUT",
                                         (unsigned long)request);
            }
            break;

        case STATE_START_ITEM:
            job = &g_controller.jobs[g_controller.jobIndex];
            send_state("MOVE_PICK");
            result = start_xy_move(job->sourceX10, job->sourceY10, nowMs);
            if (result == MOTION_FAILED) {
                enter_fault("MOVE_PICK_TX");
            } else if (result == MOTION_SKIPPED) {
                g_controller.currentX10 = job->sourceX10;
                g_controller.currentY10 = job->sourceY10;
                g_controller.state = STATE_COMMAND_PICK_DOWN;
            } else {
                g_controller.state = STATE_WAIT_PICK_XY;
            }
            break;

        case STATE_WAIT_PICK_XY:
            if (deadline_reached(nowMs)) {
                g_controller.currentX10 = g_controller.pendingX10;
                g_controller.currentY10 = g_controller.pendingY10;
                g_controller.state = STATE_COMMAND_PICK_DOWN;
            }
            break;

        case STATE_COMMAND_PICK_DOWN:
            send_state("Z_DOWN_PICK");
            g_controller.zSafe = false;
            if (start_z_move(false, nowMs) == MOTION_FAILED) {
                enter_fault("Z_DOWN_PICK_TX");
            } else {
                g_controller.state = STATE_WAIT_PICK_DOWN;
            }
            break;

        case STATE_WAIT_PICK_DOWN:
            if (deadline_reached(nowMs)) {
                MOS_Switch_On();
                if (!MOS_Switch_IsOn()) {
                    enter_fault("MAGNET_INHIBITED");
                    break;
                }
                send_state("MAGNET_PICK");
                g_controller.deadlineMs = nowMs + GANTRY_MAGNET_DWELL_MS;
                g_controller.state = STATE_WAIT_MAGNET_PICK;
            }
            break;

        case STATE_WAIT_MAGNET_PICK:
            if (deadline_reached(nowMs)) {
                g_controller.state = STATE_COMMAND_PICK_UP;
            }
            break;

        case STATE_COMMAND_PICK_UP:
            send_state("Z_UP_PICK");
            if (start_z_move(true, nowMs) == MOTION_FAILED) {
                enter_fault("Z_UP_PICK_TX");
            } else {
                g_controller.state = STATE_WAIT_PICK_UP;
            }
            break;

        case STATE_WAIT_PICK_UP:
            if (deadline_reached(nowMs)) {
                g_controller.zSafe = true;
                g_controller.state = STATE_COMMAND_PLACE_XY_ROTATE;
            }
            break;

        case STATE_COMMAND_PLACE_XY_ROTATE:
            job = &g_controller.jobs[g_controller.jobIndex];
            send_state("MOVE_PLACE_ROTATE");
            result = start_place_xy_and_rotation(job, nowMs);
            if (result == MOTION_FAILED) {
                enter_fault("MOVE_PLACE_ROTATE_TX");
            } else if (result == MOTION_SKIPPED) {
                g_controller.currentX10 = g_controller.pendingX10;
                g_controller.currentY10 = g_controller.pendingY10;
                g_controller.rotationMdeg = g_controller.pendingRotationMdeg;
                g_controller.state = STATE_COMMAND_PLACE_DOWN;
            } else {
                g_controller.state = STATE_WAIT_PLACE_XY_ROTATE;
            }
            break;

        case STATE_WAIT_PLACE_XY_ROTATE:
            if (deadline_reached(nowMs)) {
                g_controller.currentX10 = g_controller.pendingX10;
                g_controller.currentY10 = g_controller.pendingY10;
                g_controller.rotationMdeg = g_controller.pendingRotationMdeg;
                g_controller.state = STATE_COMMAND_PLACE_DOWN;
            }
            break;

        case STATE_COMMAND_PLACE_DOWN:
            send_state("Z_DOWN_PLACE");
            g_controller.zSafe = false;
            if (start_z_move(false, nowMs) == MOTION_FAILED) {
                enter_fault("Z_DOWN_PLACE_TX");
            } else {
                g_controller.state = STATE_WAIT_PLACE_DOWN;
            }
            break;

        case STATE_WAIT_PLACE_DOWN:
            if (deadline_reached(nowMs)) {
                MOS_Switch_Off();
                send_state("MAGNET_RELEASE");
                g_controller.deadlineMs = nowMs + GANTRY_MAGNET_DWELL_MS;
                g_controller.state = STATE_WAIT_MAGNET_RELEASE;
            }
            break;

        case STATE_WAIT_MAGNET_RELEASE:
            if (deadline_reached(nowMs)) {
                g_controller.state = STATE_COMMAND_PLACE_UP;
            }
            break;

        case STATE_COMMAND_PLACE_UP:
            send_state("Z_UP_PLACE");
            if (start_z_move(true, nowMs) == MOTION_FAILED) {
                enter_fault("Z_UP_PLACE_TX");
            } else {
                g_controller.state = STATE_WAIT_PLACE_UP;
            }
            break;

        case STATE_WAIT_PLACE_UP:
            if (deadline_reached(nowMs)) {
                g_controller.zSafe = true;
                g_controller.state = STATE_COMMAND_ROTATE_ZERO;
            }
            break;

        case STATE_COMMAND_ROTATE_ZERO:
            send_state("ROTATE_ZERO");
            result = start_rotation(-g_controller.rotationMdeg, nowMs);
            if (result == MOTION_FAILED) {
                enter_fault("ROTATE_ZERO_TX");
            } else if (result == MOTION_SKIPPED) {
                g_controller.rotationMdeg = 0;
                g_controller.state = STATE_NEXT_ITEM;
            } else {
                g_controller.state = STATE_WAIT_ROTATE_ZERO;
            }
            break;

        case STATE_WAIT_ROTATE_ZERO:
            if (deadline_reached(nowMs)) {
                g_controller.rotationMdeg = g_controller.pendingRotationMdeg;
                g_controller.state = STATE_NEXT_ITEM;
            }
            break;

        case STATE_NEXT_ITEM:
            g_controller.jobIndex++;
            if (g_controller.jobIndex < g_controller.jobCount) {
                g_controller.state = STATE_START_ITEM;
            } else {
                g_controller.state = STATE_COMMAND_RETURN_HOME;
            }
            break;

        case STATE_COMMAND_RETURN_HOME:
            send_state("RETURN_HOME");
            result = start_xy_move(GANTRY_HOME_X_X10, GANTRY_HOME_Y_X10,
                                   nowMs);
            if (result == MOTION_FAILED) {
                enter_fault("RETURN_HOME_TX");
            } else if (result == MOTION_SKIPPED) {
                g_controller.currentX10 = GANTRY_HOME_X_X10;
                g_controller.currentY10 = GANTRY_HOME_Y_X10;
                finish_request();
            } else {
                g_controller.state = STATE_WAIT_RETURN_HOME;
            }
            break;

        case STATE_WAIT_RETURN_HOME:
            if (deadline_reached(nowMs)) {
                g_controller.currentX10 = g_controller.pendingX10;
                g_controller.currentY10 = g_controller.pendingY10;
                finish_request();
            }
            break;

        case STATE_WAIT_MANUAL_XY:
            if (deadline_reached(nowMs)) {
                g_controller.currentX10 = g_controller.pendingX10;
                g_controller.currentY10 = g_controller.pendingY10;
                finish_request();
            }
            break;

        default:
            enter_fault("STATE");
            break;
    }
}
