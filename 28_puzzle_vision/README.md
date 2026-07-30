# E题第一问：四片搬运与拼图

本工程完成题目第一大问的两个模式：

- 按键1：识别下半区的4片镀锌铁片，全部搬到分界线上方；优先保持方向，空间不足时自动旋转90°后紧凑排布。
- 按键2：识别并匹配题图2的4片，在上半区靠近分界线拼成 `100 mm × 60 mm` 矩形。

两种模式都在启动时冻结一份稳定的4片坐标。搬走第一片后，即使实时画面只剩3片，也不会改变本次计划。

## 固定坐标与行程

- A4物理左上角：`(0, 0)`
- X向右、Y向下，单位mm
- 固定初始磁铁中心：A4 `(5, 61) mm`
- 当前实测可达质心：`X=5..205 mm`，`Y=61..286 mm`
- 1号电机正转向左，所以A4 `+X` 对应1号电机反转
- 2、3号电机正转向下，与A4 `+Y` 一致
- 4号电机正转为Z向上；每次拾取/放置下降8mm、再上升8mm
- 5号电机负责铁片旋转；`--rotation-sign` 必须在 `+90°` 实测后确定

Y总行程只有225mm，不能覆盖297mm长的整张A4。本方案因此固定为“下半区取件、上半区靠分界线放置”：松散铁片质心约在 `Y=150..286 mm`，目标矩形约在 `Y=divider-65..divider-5 mm`，避开顶部 `Y<61 mm` 的机械盲区。网页仍会逐点检查，越界时整份计划拒绝。

## 树莓派与天猛星接线

树莓派使用主串口，天猛星使用独立的UART3：

| 树莓派 | 天猛星 |
|---|---|
| GPIO14/TXD，物理8脚 | PB13/RX |
| GPIO15/RXD，物理10脚 | PB12/TX |
| GND，物理6脚 | GND |

两边只共地，不连接3.3V或5V。电机驱动器继续使用UART0 PA10/PA11，不与树莓派共用。

树莓派的GPIO串口不能同时作为Linux登录控制台。首次配置执行：

```bash
cd /home/wxm/28_puzzle_vision/raspberry_pi
sudo sh setup_uart_for_mcu.sh
sudo reboot
```

脚本会先备份`/boot/firmware/cmdline.txt`，再关闭`serial-getty`并移除
`console=serial0,115200`，不会关闭UART硬件。

## 启动

```bash
cd /home/wxm/28_puzzle_vision/raspberry_pi
sh start_puzzle_vision.sh
```

停止服务使用 `sh stop_puzzle_vision.sh`。日志在 `puzzle_vision.log`。

浏览器打开：`http://10.29.23.60:8081`

固定相机后，等待A4四角稳定，再点一次“锁定当前A4物理方向”。锁定结果保存在 `a4_calibration.json`，服务重启后仍保持同一坐标方向。相机、A4或分辨率改变后，先点“重新自动识别A4”，再重新锁定。

## 启动条件

按键1或2只有同时满足下列条件才会运动：

1. A4方向已锁定，分界线稳定。
2. 连续多帧稳定识别到恰好4片。
3. 4片总面积、轮廓凸度和顶点数通过几何检查。
4. 对应模式的4个拾取质心和4个放置质心都在滑台范围内。
5. 预计执行时间不超过题目两分钟。
6. 树莓派 5 的 40 针排针 GPIO14/15 使用 `/dev/ttyAMA0`；必须已经收到天猛星回应，而不只是成功打开串口。`/dev/serial0` 在本机指向专用 Debug 接口 `/dev/ttyAMA10`，不得用于本项目。

硬件按键触发后，树莓派最多等待15秒获得稳定方案，然后一次性下发整份计划。任一点不可达时整份计划拒绝，不会执行一半。

## 串口协议

波特率115200，每行一个ASCII命令，坐标单位0.1mm、角度单位0.001°：

```text
MCU -> Pi: KEY request_id 1
MCU -> Pi: KEY request_id 2

Pi -> MCU: PLAN request_id mode 4
Pi -> MCU: ITEM request_id index pick_x10 pick_y10 place_x10 place_y10 angle_mdeg
Pi -> MCU: COMMIT request_id
Pi -> MCU: STOP
```

MCU响应 `STATE`、`DONE` 或 `ERR`。重复的 `request_id` 不会重复执行。

## 安全动作顺序

每片严格执行：安全高度到拾取XY → Z下降8mm → 磁铁开启 → Z抬升 → 5轴旋转 → 到放置XY → Z下降 → 磁铁关闭 → Z抬升 → 5轴回零。XY和旋转只允许在Z安全高度运动。

PB21或网页急停会先关闭电磁铁，再停止全部电机。急停或断电后软件坐标失效，必须人工回固定初始位置并复位，程序不会猜测当前位置。

## 镜头畸变标定

A4四角透视只能校正相机倾斜，不能消除广角镜头的桶形畸变。正式测坐标前建议打印 `10×7` 格棋盘（即 `9×6` 内角点，每格20mm），采集15～25个覆盖中央、四边和四角的不同倾斜视图：

```bash
cd /home/wxm/28_puzzle_vision/raspberry_pi
python3 calibrate_camera.py --self-test
python3 calibrate_camera.py \
  --live --headless \
  --cols 9 --rows 6 --square-mm 20 \
  --samples 20 --min-views 12 --max-seconds 180 \
  --output camera_calibration.npz
```

生成文件后重启视觉服务。程序会先对1280×720原图去畸变，再识别A4；旧的A4像素角点会被自动判为失效，必须重新锁定A4。网页会显示标定是否加载和平均重投影误差。

## 自测

```bash
cd /home/wxm/28_puzzle_vision/raspberry_pi
python3 test_puzzle_motion.py
python3 puzzle_vision.py --self-test
```

网页API：

- `GET /api/status`：识别、模式1/2候选计划、可达性、串口和MCU状态
- `GET /raw.jpg`：未画标记、未去畸变的传感器单帧，仅用于镜头标定
- `POST /api/task?mode=1`：网页模拟按键1
- `POST /api/task?mode=2`：网页模拟按键2
- `POST /api/stop`：急停
