#ifndef GANTRY_CONFIG_H
#define GANTRY_CONFIG_H

/* All A4 coordinates use 0.1 mm integer units: origin at A4 top-left,
 * +X right, +Y down.  The mechanism is manually placed at HOME before reset.
 */
#define GANTRY_HOME_X_X10             50L
#define GANTRY_HOME_Y_X10             610L
#define GANTRY_MIN_X_X10              50L
#define GANTRY_MAX_X_X10              2050L
#define GANTRY_MIN_Y_X10              610L
#define GANTRY_MAX_Y_X10              2860L

#define GANTRY_REQUIRED_PLAN_ITEMS    4U
/* Pi may wait up to 15 s for a stable multi-frame vision snapshot. */
#define GANTRY_PLAN_TIMEOUT_MS        20000U

#define GANTRY_XY_MAX_SPEED_RPM       600U
#define GANTRY_XY_MIN_SPEED_RPM       60U
#define GANTRY_XY_ACCELERATION        220U

/* The driver TX lines are not connected back to the MCU, so there is no
 * electrical position-arrived feedback.  The controller calculates the full
 * triangular/trapezoidal acceleration profile and then adds this settle guard
 * before Z is allowed to descend.
 */
#define GANTRY_XY_SETTLE_MS           500U

/* Motor 4 positive moves upward. */
#define GANTRY_Z_TRAVEL_CM             0.8F
#define GANTRY_Z_SPEED_RPM             220U
#define GANTRY_Z_ACCELERATION          220U

#define GANTRY_MAGNET_DWELL_MS         400U
#define GANTRY_MOTION_MARGIN_MS        800U

/* ITEM carries the already signed motor-5 command angle.  Image-clockwise to
 * motor-direction calibration belongs to the Pi's --rotation-sign setting.
 */
#define GANTRY_MAX_ABS_ROTATION_MDEG   360000L

#endif
