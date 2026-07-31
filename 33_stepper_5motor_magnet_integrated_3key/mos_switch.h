#ifndef MOS_SWITCH_H_
#define MOS_SWITCH_H_

#include <stdbool.h>

/* Active-high PA14 control for the external MOS/electromagnet module. */
void MOS_Switch_Init(void);
void MOS_Switch_On(void);
void MOS_Switch_Off(void);
/* Immediately removes power and prevents re-energizing until MCU reset. */
void MOS_Switch_EmergencyOff(void);
void MOS_Switch_Set(bool enabled);
bool MOS_Switch_IsOn(void);
bool MOS_Switch_IsInhibited(void);

#endif /* MOS_SWITCH_H_ */
