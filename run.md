# run.md - G0 관문 (frozen OpenVLA가 peer 관측을 쓸 수 있는가)

서버(헤드리스) 기준 bash. 코드 폴더는 `~/Desktop/sj/peerview`, 외부 저장소는 `~/Desktop/sj/peerview_ext` 로 가정.
옛 버전은 `git show HEAD:run.md`.

## 0. 환경 (1회)

```bash
bash setup_server.sh ~/Desktop/sj/peerview_ext ysj
```

| 인자 | 기본 | 뜻 |
|---|---|---|
| 1번째 위치 인자 | `~/Desktop/sj/peerview_ext` | LIBERO·LIBERO-Occ 클론 위치 |
| 2번째 위치 인자 | `ysj` | conda env 이름. 없으면 python 3.10으로 생성 |

매 셸마다 아래를 먼저 실행. GPU는 UUID로 1장 고정(`nvidia-smi -L`), EGL 장치도 같은 카드로 맞춤.

```bash
conda activate ysj
export PYTHONPATH=$HOME/Desktop/sj/peerview_ext/LIBERO
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx
export MUJOCO_EGL_DEVICE_ID=0
cd ~/Desktop/sj/peerview
```

## 1. 렌더 검사 (정책 없음, GPU 거의 안 씀)

```bash
python render_check.py \
  --occ_root ~/Desktop/sj/peerview_ext/Libero-Occ \
  --libero_root ~/Desktop/sj/peerview_ext/LIBERO \
  --out runs/render_check \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --task_filter "" \
  --init_idx 0 \
  --n_peer 3 \
  --res 256 \
  --seed 7
```

| 인자 | 기본 | 뜻 |
|---|---|---|
| `--occ_root` | 필수 | LIBERO-Occ 클론. BDDL·init 파일을 그 자리에서 읽음(설치 스크립트 불필요) |
| `--libero_root` | 필수 | 원본 LIBERO 클론. 원본 BDDL과 비교해 폐색 물체를 찾음 |
| `--out` | 필수 | 태스크별 조건 그림 PNG + `diag.jsonl` |
| `--suites` | 4개 전부 | 대상 suite |
| `--task_filter` | `""` | 태스크 이름 부분 문자열 필터 |
| `--init_idx` | `0` | 쓸 초기 상태 번호 |
| `--n_peer` | `3` | 태스크당 peer 포즈 수 |
| `--res` | `256` | 렌더 해상도. OpenVLA LIBERO 평가와 같은 값 |
| `--seed` | `7` | peer 포즈 샘플링 시드 |

`--list_only` 를 붙이면 태스크·폐색 물체 목록만 출력. 확인할 값: `ori=flipud(0.00)`, `self≈0`, `geo` 1에 가까움, `peers=3/…`.

## 2. G0 프로브 (OpenVLA, 최소판)

```bash
python probe.py \
  --occ_root ~/Desktop/sj/peerview_ext/Libero-Occ \
  --libero_root ~/Desktop/sj/peerview_ext/LIBERO \
  --out runs/g0_spatial \
  --suites libero_spatial \
  --task_filter "" \
  --max_tasks 0 \
  --n_init 3 \
  --drive clean \
  --probe_every 20 \
  --max_probes 6 \
  --n_peer 3 \
  --peer_tries 24 \
  --min_target_px 30 \
  --az_min 30 \
  --az_max 150 \
  --el_min 15 \
  --el_max 60 \
  --dist_lo 0.8 \
  --dist_hi 1.2 \
  --hole_iters 2 \
  --res 256 \
  --num_steps_wait 10 \
  --max_steps 0 \
  --seed 7 \
  --ckpt_template "openvla/openvla-7b-finetuned-libero-{suffix}" \
  --device cuda:0 \
  --attn sdpa \
  --save_png 3
```

| 인자 | 기본 | 뜻 |
|---|---|---|
| `--out` | 필수 | `probe.jsonl`·`episodes.jsonl`·`tasks.json`·`config.json`·`png/` |
| `--suites` | 4개 전부 | suite마다 체크포인트 7B를 따로 받음(약 15GB씩). 최소판은 spatial 하나 |
| `--max_tasks` | `0` | 0이면 전부. 빠른 확인은 2 |
| `--n_init` | `10` | 태스크당 에피소드(초기 상태) 수 |
| `--drive` | `clean` | 로봇을 움직이는 입력. clean = 교사 시점이라 과제가 진행되는 상태를 모음. occ/peer/reproj도 가능 |
| `--probe_every` | `10` | 몇 step마다 4조건을 채점할지. 0이면 성공률만 잼 |
| `--max_probes` | `0` | 에피소드당 프로브 상한. 0이면 무제한 |
| `--n_peer` | `3` | 프로브마다 peer 포즈 수 |
| `--peer_tries` | `24` | 대상이 보이는 peer 포즈를 찾기 위한 최대 후보 수 |
| `--min_target_px` | `30` | peer 시점에 대상이 이만큼 보여야 채택(C2 조건) |
| `--az_min`/`--az_max` | `30`/`150` | ego 대비 방위각 오프셋 범위(도, 좌우 무작위) |
| `--el_min`/`--el_max` | `15`/`60` | peer 고도각 범위(도) |
| `--dist_lo`/`--dist_hi` | `0.8`/`1.2` | ego 거리 대비 peer 거리 배율 |
| `--hole_iters` | `2` | see-through 구멍 메우기 반복(3x3 이웃) |
| `--max_steps` | `0` | 0이면 suite별 openvla 값(220/280/300/520) |
| `--ckpt_template` | 위 값 | `{suffix}` = spatial/object/goal/10. 로컬 경로 템플릿도 가능 |
| `--attn` | `sdpa` | flash-attn 미설치 기준. Blackwell에서 막히면 `eager` |
| `--save_png` | `3` | 태스크당 저장할 조건 그림 수 |

`--dry_policy` 는 모델 없이 파이프라인만 돌림. `--no_center_crop` 은 쓰지 않음(LIBERO 체크포인트는 crop 학습).

## 3. 집계

```bash
python summarize.py runs/g0_spatial \
  --min_kl 0.5 \
  --vis_ratio 0.7
```

| 인자 | 기본 | 뜻 |
|---|---|---|
| 위치 인자 | 필수 | 2단계 `--out` |
| `--min_kl` | `0.5` | 폐색이 정책 출력을 이만큼 이상 바꾼 상태만 셈(위생 검사) |
| `--vis_ratio` | `0.7` | 폐색 시점에서 대상이 clean 대비 이 비율 미만으로 보이는 상태만 셈 |

출력: 조건별 KL·recovery·token 일치율 표, `g0_bars.png`. recovery 0 = 폐색 시점과 같음, 1 = clean과 같음.

## 주의

- 1·3단계와 2단계의 `--dry_policy` 경로는 로컬 WSL(py3.10, mujoco 3.2.7, robosuite 1.4.1)에서 실행 확인. 0단계와 실제 OpenVLA 경로는 서버에서 처음 돌림.
- `teacher_argmax_ok`(집계 첫 줄)가 1이 아니면 teacher forcing 위치가 어긋난 것. 결과를 쓰지 말 것.
- init 파일은 torch 없이 numpy 전용 unpickler로 읽음. numpy 외 객체가 들어 있으면 거부함.
- peer 카메라는 모델의 `sideview` 카메라를 빌려 씀. 정책 입력(agentview)은 건드리지 않음.
