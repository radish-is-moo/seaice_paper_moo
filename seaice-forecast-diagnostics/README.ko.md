# 해빙 예측오차 진단 코드

최신 원고의 **2023–2025년 ver2 실험**을 정리한 GitHub 업로드용 저장소입니다. 2020–2025년의 별도 실험은 포함하지 않습니다. 원본 연구 폴더를 수정하지 않고 별도 구조로 정리했습니다.

## 가장 간단한 실행

저장소 폴더에서 다음 명령을 실행합니다.

```sh
python -m pip install -e ".[maps,test]"
python -m seaice_diagnostics check-data
python -m pytest -q
python -m seaice_diagnostics figures
```

최신 Figure 1–7을 `results/figures/`에 PNG와 PDF로 생성합니다. 필요한 요약표와 Figure 3의 선택 사례·Figure 6의 진단 배열을 포함하므로 이 작업에는 전체 예측 자료가 필요하지 않습니다. 지도 배경은 Cartopy가 첫 실행에서 내려받을 수 있습니다.

## 전체 재계산

`configs/paths.example.json`을 `configs/paths.json`으로 복사해 일별 입력 캐시, 예측·기준장 아카이브, ERA5 격자 파일 등의 경로를 지정합니다. 다음 순서로 실행합니다.

1. `python -m seaice_diagnostics audit` — 기존 ver2 자료의 SHA-256 검증
2. `python -m seaice_diagnostics evaluate ocean` — 해양 마스크를 적용한 해역별 평가·상태 전이·지속성
3. `python -m seaice_diagnostics evaluate states` — 범북극 목표일 상태별 평가
4. `python -m seaice_diagnostics evaluate conditions` — 범북극 초기조건별 평가

학습 노트북은 `train_cnn.ipynb`, `train_unet.ipynb`, `train_gnn.ipynb`로 정리했습니다. 모델 구조와 학습 설정은 유지하고, 이전 평가용 셀과 출력은 제거했습니다. `prepare-cache`는 **이미 같은 격자로 전처리한 NetCDF**를 일별 캐시로 바꾸는 단계입니다. 원자료 다운로드·원격 자료 접근 권한을 제공하는 명령은 아닙니다.

해양에는 개방수역이 포함됩니다. 목표일 상태 분류와 초기조건 고정 분류를 구분하며, 모델별 RMSE를 계산한 다음 같은 비중으로 평균합니다. NSR 결과를 범북극 결과로 대체하지 않습니다.

## GitHub에 올리기

이 폴더의 **내용 전체**를 새 GitHub 저장소의 최상위에 올리면 됩니다. 압축파일 자체를 하나의 파일로 올리는 대신 압축을 풀어 업로드하세요. `.github`, `.gitignore`, `.gitattributes` 등 숨김 파일도 포함합니다. Git을 사용하면 숨김 파일을 빠뜨리지 않을 수 있습니다.

```sh
git init
git add .
git commit -m "Organize sea-ice forecast diagnostics"
```

이후 생성한 GitHub 저장소가 안내하는 원격 연결·push 명령을 사용합니다. 이 작업에서는 실제 GitHub 게시를 수행하지 않았습니다. 대용량 원자료·체크포인트·개인 경로 설정·실행 결과는 `.gitignore`로 제외됩니다.

라이선스는 **미지정**입니다. 임의로 MIT 등의 라이선스를 부여하지 않았습니다. 상세한 자료 형식·검증 범위는 [영문 README](README.md)와 `docs/`를 참고하세요.
