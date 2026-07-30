#ifndef RPI_LINK_H
#define RPI_LINK_H

#include <stdbool.h>
#include <stddef.h>

#define RPI_LINK_LINE_MAX 128U

/* UART3, PB12 TX / PB13 RX, 115200 8N1. */
void RpiLink_Init(void);

/* Returns one newline-terminated ASCII command without CR/LF. */
bool RpiLink_ReadLine(char *line, size_t capacity);

/* True once for an RX-ring overflow or an overlong input line. */
bool RpiLink_TakeRxError(void);

bool RpiLink_SendLine(const char *line);
bool RpiLink_SendFormat(const char *format, ...);

#endif
