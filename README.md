# UR5e–Can: iPhone teleoperation và thu demo

Điều khiển **UR5e / PickPlaceCan / OSC_POSE** trong robosuite bằng pose ARKit
6 DoF từ HEBI Mobile I/O; lưu demo gốc và xuất low_dim.hdf5 cho robomimic.
Chỉ mô phỏng; không kết nối UR5e thật hoặc chạy huấn luyện.

## Chuẩn bị

Đã chạy trên Ubuntu 22.04, dppo Python 3.8.20, NumPy 1.24.4, MuJoCo 3.2.3,
robosuite 1.4.1, robomimic 0.5.0, PyTorch 2.4.0. Receiver dùng Python 3.10 riêng
với hebi-py 2.16.3; không cần LeRobot.
Các phiên bản đã kiểm tra được ghi trong environment-versions.json.

Sử dụng env dppo đang hoạt động:
~~~bash
git clone https://github.com/khangnhtran/ur5e-phone-teleop.git
cd ur5e-phone-teleop
conda activate dppo
python -B check_environment.py
bash setup_receiver.sh
~~~
setup_receiver.sh chỉ tạo .venv-receiver và cài requirements-receiver.txt vào đó.
Nếu env này đã tồn tại thì giữ nguyên. Cần python3.10 và module venv của Ubuntu;
có thể chọn Python qua RECEIVER_SYSTEM_PYTHON.
Repo không tự cài hoặc nâng/hạ thư viện trong dppo. Trên máy mới cần chuẩn bị
env mô phỏng đúng phiên bản trước. Launcher dùng Python đang activate;
có thể chọn DPPO_PYTHON=/duong/dan/env/bin/python.

## Kết nối iPhone

1. Mở HEBI Mobile I/O, bật chế độ AR/pose và cấp quyền camera, Local Network khi app yêu cầu.
2. PC và iPhone cùng LAN; app ở foreground, camera thấy môi trường đủ sáng.
3. B1 cần là nút giữ: nhấn=1, nhả=0. Dùng ARKit position/orientation, không thay bằng IMU.
4. Trên mạng mới, xác minh IP và tên bằng probe:
~~~bash
.venv-receiver/bin/python -I phone_probe.py --discover-seconds 10
# Nếu broadcast không tìm được, dùng IP thực tế hiển thị trên iPhone:
.venv-receiver/bin/python -I phone_probe.py --lookup-address IP_THUC --discover-seconds 10
.venv-receiver/bin/python -I phone_probe.py --lookup-address IP_THUC --family FAMILY_THUC --name TEN_THUC
~~~
Probe in position (m), quaternion wxyz, B1–B8, tuổi pose và Hz.
Cấu hình đi kèm dùng thông tin đã nhận trước đây: 192.168.1.103 / HEBI / mobileIO;
DHCP có thể đổi IP. Sửa collection_config.json hoặc truyền --ip, --family, --name.

App đã dùng không gửi ar_quality: mặc định cho phép thiếu quality, không coi là Normal.
Nếu có quality thì chỉ nhận Normal, vẫn kiểm tra tuổi pose/nút và timeout.
Pose hữu hạn nhưng ARKit đứng mà app không báo quality không phải lúc nào cũng
phát hiện được; nhả B1 nếu thấy chuyển động không đúng.

## Chạy và thu

~~~bash
conda activate dppo
bash start_collection.sh
# Tập riêng, không ghi demo:
bash start_collection.sh --mode practice
~~~
Không mở receiver riêng: collector tự chạy/dừng receiver ~100Hz qua UDP localhost;
mô phỏng và ghi ở **20Hz**, một mẫu mỗi env.step().
Không chạy đồng thời hai collector; nếu cổng bận, dừng đúng phiên cũ hoặc chọn --port khác.
Nhả các nút trước khi bắt đầu và sau reset.

| Nút | Thao tác |
| --- | --- |
| B1 giữ | Điều khiển XYZ và xoay đầu kẹp cả 3 trục, gồm Z |
| B1 nhả | Giữ mục tiêu; giữ lại lấy mốc điện thoại/TCP mới |
| B2 nhấn | Đảo yêu cầu đóng/mở kẹp |
| B3 nhấn | Reset mới và bắt đầu ghi một lượt trong mode record |
| B8 nhấn | Hủy lượt, reset và chờ B3 |
| Ctrl-C | Dừng phiên, flush raw và xuất các lượt thành công |

Lần đầu giữ B1: màn hình hướng lên, cạnh trên theo world +X của robot
(không phải chiều ngang ảnh camera). Mỗi lần giữ lại đều lấy mốc pose mới;
hiệu chỉnh lại hệ trục tổng thể bằng cách khởi động phiên mới.
Reset giữ tư thế gắp từ trên đã hiệu chỉnh, sau đó cho xoay tự do.
Đường Z lấy từ TCP thật, kèm chấm tâm bán trong suốt.
Mất kết nối/tracking: pause mô phỏng và ghi, giữ trạng thái kẹp.
Kết nối lại phải nhả rồi giữ lại B1.

Thao tác: tiếp cận → đóng → nâng → chuyển → hạ → nhả → rút tay.
Giữ nguyên env._check_success() của PickPlaceCan 1.4.1:
tâm lon ở đúng ô nhận (X/Y trong ô, Z từ mặt đáy ô đến +0.1m), TCP đủ xa lon
theo điều kiện 1-tanh(10*distance)<0.6.
Success liên tục 0.5s (10 bước) thì nhận lượt, flush và chờ B3 tiếp.
Lượt hủy/chưa đạt vẫn giữ raw nhưng không vào data/demo_*.
Vị trí lon được task lấy mẫu lại qua reset.

## Độ nhạy

collection_config.json chứa mặc định; tham số CLI ghi đè cho từng phiên:

- pos_scale=2, rot_scale=1, orientation_control=phone.
- smooth_tau=0.025s, max_speed=0.4m/s, max_accel=2.5m/s², max_angular_speed=1.8rad/s.
- max_lag_m=0.04, max_lag_rad=0.30 giới hạn mục tiêu đi trước TCP, tránh tích lũy
  lệnh xa. Tay quá nhanh có thể bị giảm quãng đường/góc; giảm tốc hoặc nhả/giữ lại B1.
- Lấy pose mới nhất mỗi bước, giữ cạnh nhấn/nhả; không bước bù dồn sau khi chậm.
- --orientation-control locked để giữ hướng; --translation-only để tập XYZ;
  --no-eef-axes để ẩn đường/chấm.

Đây là giới hạn lệnh, không cam kết giới hạn tốc độ robot thực đo hoặc tránh va chạm.
Không đảm bảo loại bỏ drift do ARKit.

## File kết quả và xuất robomimic

Mỗi lần chạy tạo thư mục mới trong ../ur5e_can_phone_demos/ tính từ repo.
Đổi bằng --output-root /THU_MUC_RIENG; console in đường dẫn SESSION.

~~~text
record_<timestamp_uuid>/
  demo.hdf5        states/actions/XML/metadata gốc, chỉ lượt được nhận
  raw/             trạng thái ban đầu và các chunk nguyên gốc mọi lượt
  episodes.json    accepted, task_success, lý do kết thúc
  session.json     phiên bản, seed nếu có, controller và cấu hình điện thoại
  events.jsonl
  receiver.log
  robomimic.hdf5   bản sao qua convert_robosuite chính thức
  low_dim.hdf5     file huấn luyện sau dataset_states_to_obs chính thức
  export.json
  data_check.json
~~~
Sau ít nhất một lượt thành công, Ctrl-C rồi chờ dòng READY chỉ ra low_dim.hdf5.
Xuất lại hoặc chọn đầu ra mới:

~~~bash
bash export.sh /SESSION/demo.hdf5
bash export.sh /SESSION/demo.hdf5 --output-dir /THU_MUC_MOI
~~~
Không ghi đè đầu ra đã có. Raw không được chuẩn hóa hoặc sửa actions.
DataCollectionWrapper lưu N+1 states / N actions; demo.hdf5 dùng states[:-1],
giữ terminal_state riêng và ghi đúng actions đã truyền vào env.step().
low_dim chứa data/demo_*/actions, states, obs, next_obs, rewards, dones,
model_file, num_samples và env_args.
next_obs cuối lấy từ terminal_state đã ghi trên bản sao; không hard-code chiều obs/state.
Train/valid chia theo cả demo: ≥2 demo mới có valid; 1 demo chỉ có train.
Fixture/synthetic và external chưa xác minh không được xuất huấn luyện.

~~~bash
python -B check_dataset.py /SESSION/demo.hdf5 --replay states --demo demo_1 --render
~~~
Export tự kiểm tra finite, shape/thời gian, action bounds, UR5e/OSC_POSE/20Hz,
state playback một lượt và căn chỉnh observation.
Playback states khôi phục trạng thái ghi, khác với chạy lại actions.

## Xử lý lỗi

- NO_DEVICES: kiểm tra IP thực tế, cùng LAN, app foreground, quyền camera/Local Network,
  VPN/AP isolation. Không tắt toàn bộ firewall.
- PAUSED: đọc phone_error và receiver.log; nhả B1 rồi giữ lại sau khi pose bình thường.
- Thiếu .venv-receiver: chạy setup_receiver.sh; hoặc dùng
  --receiver-python /duong/dan/receiver/bin/python để tận dụng env có sẵn.
- Sai phiên bản: activate đúng dppo rồi chạy check_environment.py; không tự nâng/hạ gói.
- Không hiện cửa sổ: kiểm tra desktop DISPLAY/OpenGL, mặc định MUJOCO_GL=glx.
- Export lỗi: raw demo.hdf5 vẫn giữ nguyên; đọc lỗi rồi chạy export.sh với đầu ra mới.
- Cảnh báo HEBI thiếu matplotlib/pynput không ngăn nhận ARKit pose.

## Phạm vi bản đóng gói

Giữ collector, adapter, receiver, TCP reset/hiển thị, lưu dữ liệu, kiểm tra và xuất.
Bỏ audit cũ, test/fixture, ảnh thử, log, dataset và virtualenv khỏi bản đóng gói.
Không sửa hoặc dừng thư mục thu demo gốc.
Giữ mapping/action, tư thế reset, success và HDF5 từ mã đang dùng; thay launcher/đường dẫn
để clone sang nơi khác. Không cung cấp dataset synthetic để huấn luyện.
Các kiểm tra mô phỏng không thay cho thử iPhone trên máy/mạng mới.

## Kiểm tra bản đóng gói

Đã chạy trên env dppo thực tế:
- Kiểm tra phiên bản, cú pháp Python/shell và đầy đủ import nội bộ.
- Launcher chạy từ thư mục khác; mô phỏng headless practice chờ pose bằng cổng riêng,
  không nhận điều khiển iPhone, không ghi demo.
- Playback states một demo iPhone thật 418 bước và xuất lại bằng công cụ robomimic.
  File gốc giữ nguyên SHA-256; obs/next_obs khớp, không NaN/Inf, split theo demo.
  Một demo nên train=1, valid=0.
Chưa xác minh lại cảm giác điều khiển iPhone từ bản đóng gói; thuật toán adapter,
TCP reset và overlay được giữ nguyên từ bản đã kiểm tra 6 trục trong MuJoCo.
