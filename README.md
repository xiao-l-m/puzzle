# A4 镀锌铁片三模式拼图系统

本仓库整合了当前可直接联调的三种任务模式，包含树莓派视觉/网页端与天猛星 MSPM0G3507 五轴滑台执行端。

- 模式 1（第一大问第 1 小问）：识别下半区 4 片灰色镀锌铁片，全部搬到分界线上半区。
- 模式 2（第一大问第 2 小问）：识别、匹配并旋转题图 2 的 4 片铁片，在上半区拼成 `100 mm × 60 mm` 矩形。
- 模式 3（第二大问第 1 小问）：识别下半区 1～4 片未知白色碎片，不依赖固定模板重建规定尺寸矩形，再搬到分界线上半区。网页按钮会一键等待稳定识别、求解并启动电机。

当前实物因滑台行程采用“下半区取件、上半区放置”。模式 3 的目标矩形在分界线一侧水平居中，所有吸取点和放置点都会先做可达性检查。

## 工程目录

- `28_puzzle_vision/`：相机去畸变、A4 透视标定、碎片识别、质心/安全吸取点、模式 1/2 排布、模式 3 未知矩形重建、网页控制、串口协议和回归测试。
- `30_stepper_5motor_magnet_test/`：天猛星 CCS 工程，控制 X、双 Y、Z、旋转轴和电磁铁，支持模式 1/2 固定四片计划及模式 3 的 1～4 片可变计划。

## 坐标和机械标定

- A4 物理左上角为 `(0,0)`，X 向右、Y 向下，单位 mm。
- 固定 HOME 电磁铁中心：A4 `(5,61) mm`。
- 当前实测可达质心范围：`X=5..205 mm`、`Y=61..286 mm`。
- 电机 1：X；电机 2、3：同步 Y；电机 4：Z；电机 5：铁片旋转。
- 每次抓取或放置，Z 轴下降 8 mm 后返回安全高度。

没有原点或限位开关。每次上电/复位前必须人工回到固定 HOME，并确认 Z 在安全高度、5 号轴为 0°、电磁铁关闭。急停、断电或卡住后坐标作废，必须重新回 HOME 并复位。

## 通信接线

| Raspberry Pi 5 | 天猛星 MSPM0G3507 |
|---|---|
| GPIO14 / TXD，物理 8 脚 | PB13 / UART3 RX |
| GPIO15 / RXD，物理 10 脚 | PB12 / UART3 TX |
| GND，物理 6 脚 | GND |

只交叉连接 TX/RX 并共地，不连接两端的 3.3 V 或 5 V。串口参数为 `115200 8N1`。

天猛星 `PA10/UART0 TX` 并接五台 Emm 驱动器 RX，地址为 1～5；驱动器 TX 当前不回接天猛星。`PA14` 连接电磁铁 MOS 模块控制输入，`PB21` 为硬件急停。

## 启动

树莓派首次配置串口：

```bash
cd /home/wxm/28_puzzle_vision/raspberry_pi
sudo sh setup_uart_for_mcu.sh
sudo reboot
```

日常重启服务：

```bash
cd /home/wxm/28_puzzle_vision/raspberry_pi
sh stop_puzzle_vision.sh
sh start_puzzle_vision.sh
```

浏览器打开 `http://<树莓派IP>:8081`。服务重启会自动加载已保存的 A4 标定；相机或 A4 位置改变后必须重新识别并锁定 A4。

网页中模式 1、2 必须等 4 片稳定后启动；模式 3 的“一键启动”允许先点击，后台会等待 A4 和 1～4 片白片稳定、完成矩形求解与可达性验证后，再一次性向 MCU 下发冻结计划。运动开始后不会因画面变化修改计划。

## 单片运动时序

1. 安全高度移动至吸取点；
2. Z 下降 8 mm，开启电磁铁并等待；
3. Z 返回安全高度；
4. XY 向放置点移动，同时 5 号轴按规划角度旋转；
5. 等 XY 与旋转中较慢的一项完成并留出停稳余量；
6. Z 下降，关闭电磁铁；
7. Z 返回安全高度，5 号轴回零；
8. 全部碎片完成后 XY 返回固定 HOME。

固件 READY 标识为 `GANTRY_V3_XY_ROTATE`。驱动器没有回传真实到位信号，固件按距离、速度和加速度曲线保守计时，因此运行时仍必须有人看护并准备按下 PB21 或网页急停。

## 构建与测试

1. 在 Code Composer Studio 中导入、构建并烧录 `30_stepper_5motor_magnet_test`。
2. 树莓派端运行回归测试：

```bash
cd 28_puzzle_vision/raspberry_pi
python3 -m unittest \
  test_arbitrary_puzzle_solver.py \
  test_gantry_serial.py \
  test_puzzle_motion.py \
  test_firmware_protocol_contract.py
python3 puzzle_vision.py --self-test
```

详细的模式 3 算法见 [28_puzzle_vision/MODE3_QUESTION2_1.md](28_puzzle_vision/MODE3_QUESTION2_1.md)，串口报文和安全状态机见 [30_stepper_5motor_magnet_test/RASPBERRY_PI_PROTOCOL.md](30_stepper_5motor_magnet_test/RASPBERRY_PI_PROTOCOL.md)。
