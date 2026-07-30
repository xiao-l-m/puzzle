# 30工程：树莓派联调与安全抓放协议

## 接线

树莓派和天猛星分别供电，只连接交叉UART和共地：

| Raspberry Pi 5 | 天猛星 MSPM0G3507 |
|---|---|
| GPIO14/TXD，物理8脚 | PB13/UART3 RX |
| GPIO15/RXD，物理10脚 | PB12/UART3 TX |
| GND，物理6脚 | GND |

不要连接5V，也不需要连接双方3.3V。UART3固定为115200、8N1。
PA10/UART0仍专门连接五台Emm驱动器RX，两路UART互不共用。

## 启动条件

每次复位前必须人工确认：

- 磁铁中心位于A4坐标 `(5.0, 61.0) mm`；
- Z轴处于上方初始安全高度；
- 5号旋转轴处于0度；
- 电磁铁关闭。

本机没有原点/限位传感器。复位后程序只能按上述人工初始位置建立开环绝对坐标。
任何急停都会使坐标失效，不能继续运行；必须人工回初始位置后重新复位。

## 按键

- KEY1/PB17：向树莓派发送 `KEY seq 1`，请求第一问小问1方案。
- KEY2/PB18：向树莓派发送 `KEY seq 2`，请求第一问小问2拼图方案。
- PB21：急停；GPIO下降沿会先立即关闭PA14电磁铁，再由主循环停止1~5号电机。

KEY1/KEY2不会直接移动。树莓派没有返回完整、合法的4片计划时，所有电机保持不动。

## 文本协议

ASCII文本，115200 8N1，每条命令以换行结束。坐标为A4左上角坐标，X向右、Y向下。
坐标字段使用0.1mm整数，角度字段使用0.001度整数。

心跳：

```text
Pi  -> MCU: PING
MCU -> Pi : PONG READY
```

只有控制器空闲、位于HOME、Z安全、旋转轴为0且磁铁关闭时才回复`READY`，否则回复`PONG BUSY`。

按键请求后，树莓派冻结同一份稳定视觉方案并发送：

```text
MCU -> Pi : KEY 17 2
Pi  -> MCU: PLAN 17 2 4
Pi  -> MCU: ITEM 17 0 pick_x10 pick_y10 place_x10 place_y10 angle_mdeg
Pi  -> MCU: ITEM 17 1 pick_x10 pick_y10 place_x10 place_y10 angle_mdeg
Pi  -> MCU: ITEM 17 2 pick_x10 pick_y10 place_x10 place_y10 angle_mdeg
Pi  -> MCU: ITEM 17 3 pick_x10 pick_y10 place_x10 place_y10 angle_mdeg
Pi  -> MCU: COMMIT 17
```

两种模式都允许正负旋转角。模式1优先保持原方向，但上半区空间不足时，
树莓派会给个别长铁片发送`+90000`或`-90000`，先旋转90度再紧凑排布；
模式2按题图2发送任意合法角度。所有角度都受`GANTRY_MAX_ABS_ROTATION_MDEG`限制。
MCU会先验证四片所有源/目标坐标和角度；只有收到合法`COMMIT`后才开始移动。
网页按钮没有前置`KEY`事件，因此也可从树莓派主动发送同样的`PLAN`。MCU仅在
`IDLE + HOME有效 + Z安全 + 磁铁关闭 + 旋转轴为0`时接受主动计划；其他状态一律拒绝。

树莓派视觉不能生成安全计划时发送：

```text
REJECT 17 VISION_NOT_READY
```

MCU取消该请求并保持不动。

运行状态：

```text
ACK 17 PLAN 4
ACK 17 ITEM 0
ACK 17 COMMIT
STATE 17 MOVE_PICK ITEM 1
STATE 17 Z_DOWN_PICK ITEM 1
...
DONE 17
ERR 17 RANGE
```

调试命令：

```text
STATUS
MOVEA seq target_x10 target_y10
MOVER seq delta_x10 delta_y10
PICKPLACE seq pick_x10 pick_y10 place_x10 place_y10 angle_mdeg
STOP
```

XY可达范围固定为：`X=5.0..205.0mm`、`Y=61.0..286.0mm`。

## 单片安全动作

1. Z保持安全高度，XY移动至铁片质心；
2. 4号轴负向下降8mm；
3. PA14开启磁铁并等待400ms；
4. 4号轴正向上升8mm；
5. 在安全高度用5号轴旋转；
6. XY移动到目标质心；
7. 下降8mm、关闭磁铁并等待400ms；
8. 上升8mm；
9. 5号轴反向回0度；
10. 完成四片后XY返回HOME。

当前驱动器TX没有接回MCU，所以不能读取真实到位返回。固件不再采用简单的“距离除以RPM”估算，而是依据Emm V5加速度档位公式自动计算三角形或梯形速度曲线，并向上取整，再叠加通信余量和XY静置余量。等待结束后才允许Z下降；Z下降曲线等待结束后才允许吸取或释放。改变速度或加速度时不需要重写状态机。

## 烧录前必须实测

1. 发送5号轴`+90度`，确认从相机俯视时是否顺时针。树莓派发送的
   `ITEM angle_mdeg`已经是5号电机命令角；如果相反，只把树莓派启动参数
   `--rotation-sign`改为`-1`，MCU不要再做第二次符号补偿。
2. 确认5号轴3200脉冲是否准确对应末端360度；有减速机构时应修改
   `STEPPER_MOTOR_5_PULSES_PER_OUTPUT_REV`。
3. 单独确认4号轴负向0.8cm确实下降8mm、正向0.8cm能回到相同初始高度。
4. 首次整套测试不要放铁片，手放急停旁，逐状态确认方向和行程。

## 模式3：1～4片可变计划

```text
Pi  -> MCU: PLAN request_id 3 item_count
Pi  -> MCU: ITEM request_id index pick_x10 pick_y10 place_x10 place_y10 angle_mdeg
Pi  -> MCU: COMMIT request_id
```

`item_count` 必须是1～4，`index` 必须是 `0..item_count-1`。MCU会在任何运动前验证全部ITEM；重复ITEM返回 `ITEM_DUPLICATE`，缺失ITEM返回 `MISSING_ITEM`，0片/5片返回 `PLAN_SHAPE`。模式1、2仍可使用原来4片计划。
