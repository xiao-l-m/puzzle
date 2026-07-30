#ifndef STEPPER_CM_CONFIG_H
#define STEPPER_CM_CONFIG_H

/* Emm driver addresses configured on the five motor driver boards. */
#define STEPPER_MOTOR_1_ADDRESS          1U
#define STEPPER_MOTOR_2_ADDRESS          2U
#define STEPPER_MOTOR_3_ADDRESS          3U
#define STEPPER_MOTOR_4_ADDRESS          4U
#define STEPPER_MOTOR_5_ADDRESS          5U

/*
 * Set a direction XOR to 1 only if that motor's mechanical positive direction
 * is opposite to the other motors. Motors 2 and 3 currently use the same
 * direction because they drive the same Y axis without mirroring.
 */
#define STEPPER_MOTOR_1_DIRECTION_XOR    0U
#define STEPPER_MOTOR_2_DIRECTION_XOR    0U
#define STEPPER_MOTOR_3_DIRECTION_XOR    0U
#define STEPPER_MOTOR_4_DIRECTION_XOR    0U
#define STEPPER_MOTOR_5_DIRECTION_XOR    0U

/* Linear distance calibration from the measured full-range moves:
 * X: requested 20.0 cm, measured 20.1 cm -> 20.0 / 20.1
 * Y: requested 22.5 cm, measured 22.6 cm -> 22.5 / 22.6
 */
#define STEPPER_MOTOR_1_DISTANCE_SCALE   0.995025F
#define STEPPER_MOTOR_2_DISTANCE_SCALE   0.995575F
#define STEPPER_MOTOR_3_DISTANCE_SCALE   0.995575F
#define STEPPER_MOTOR_4_DISTANCE_SCALE   1.000000F
#define STEPPER_MOTOR_5_DISTANCE_SCALE   1.000000F

/* Current driver subdivision: 200 full steps/rev * 16 = 3200 pulses/rev. */
#define STEPPER_PULSES_PER_REV           3200U

/* Measured travel: 10 revolutions = 40 mm, therefore lead = 4 mm/rev. */
#define STEPPER_SLIDE_LEAD_MM            4U

#define STEPPER_DEFAULT_SPEED_RPM        300U
#define STEPPER_DEFAULT_ACCELERATION     10U

/*
 * Motor 5 is a rotary axis, not a lead-screw axis.  The current test assumes
 * direct drive at 16 microsteps: 3200 command pulses = one output revolution.
 * If a gearbox or belt reduction is added, change this to the measured number
 * of motor-command pulses required for one 360-degree output revolution.
 */
#define STEPPER_MOTOR_5_PULSES_PER_OUTPUT_REV 3200U
#define STEPPER_MOTOR_5_SPEED_RPM              120U
#define STEPPER_MOTOR_5_ACCELERATION           5U
#define STEPPER_MOTOR_5_MAX_ABS_DEGREES        720.0F

/* Prevent an accidental command longer than the physical slide. */
#define STEPPER_MAX_ABS_DISTANCE_CM      200.0F

#if ((STEPPER_DEFAULT_SPEED_RPM == 0U) || \
     (STEPPER_DEFAULT_SPEED_RPM > 5000U))
#error "STEPPER_DEFAULT_SPEED_RPM must be in the range 1..5000"
#endif

#if (STEPPER_DEFAULT_ACCELERATION > 255U)
#error "STEPPER_DEFAULT_ACCELERATION must be in the range 0..255"
#endif

#if ((STEPPER_PULSES_PER_REV == 0U) || (STEPPER_SLIDE_LEAD_MM == 0U))
#error "Stepper pulse and slide lead settings must be greater than zero"
#endif

#if (STEPPER_MOTOR_5_PULSES_PER_OUTPUT_REV == 0U)
#error "Motor 5 pulses per output revolution must be greater than zero"
#endif

#endif
