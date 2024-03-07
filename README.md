# 상공회의소 부산인력개발원 인텔교육 1기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/pskcci/intel-01.git
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team projects

### 제출방법

1. 팀구성 및 프로젝트 세부 논의 후, 각 팀은 프로젝트 진행을 위한 Github repository 생성

2. [doc/project/README.md](./doc/project/README.md)을 각 팀이 생성한 repository의 main README.md로 복사 후 팀 프로젝트에 맞게 수정 활용

3. 과제 제출시 `인텔교육 1기 Github repository`에 `New Issue` 생성. 생성된 Issue에 하기 내용 포함되어야 함.

    * Team name : Project Name
    * Project 소개
    * 팀원 및 팀원 역활
    * Project Github repository
    * Project 발표자료 업로드

4. 강사가 생성한 `Milestone`에 생성된 Issue에 추가 

### 평가방법

* [assessment-criteria.pdf](./doc/project/assessment-criteria.pdf) 참고

### 제출현황


### Team: 뭐먹

<"오늘 뭐 먹지?"라는 질문을 했을 때 랜덤하게 음식의 이미지와 정보를 출력해주는 프로그램>

* Members
  | Name | Role |
  |----|----|
  | 김보람 | Project leader, 프로젝트 총괄 및 데이터셋 구성, 텍스트 입력시 음식 이미지 출력 코드 작성 |
  | 김용제 | 이미지 검출 기능 및 음식 이미지 및 정보 출력 코드 작성, 코드 통합 |
  | 이경준 | 이미지 검출 기능 및 음식 이미지 및 정보 출력 코드 작성 |
  | 천호진 | 데이터셋 구성, 모델 학습, PPT제작 및 발표 |
  
* Project Github : https://github.com/brkim92/Team_project
* 발표자료 : 



### Team: IC (Inventory Controllers)

직원과 물건의 확인을 바탕으로 한 재고 관리 자동화 프로젝트

* Members
  | Name | Role |
  |----|----|
  | 공병현 | 프로젝트 총괄, 얼굴 인식 구현 |
  | 신은상 | 발표 준비, RFID 구현 |
  | 이중섭 | Github 관리, 사물 인식 구현 |
  | 최영중 | 서버 관리, 음성 인식 구현 |

* Project Github : https://github.com/Mazogorath/AI_Inventory.git

* 발표자료 : https://github.com/Mazogorath/AI_Inventory/blob/main/doc/Presentation.pptx

