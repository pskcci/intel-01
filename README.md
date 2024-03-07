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

### Team: 뭔가 센스있는 팀명

<프로젝트 요약>

* Members
  | Name | Role |
  |----|----|
  | 채치수 | Project lead, 프로젝트를 총괄하고 망하면 책임진다. |
  | 송태섭 | Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | 정대만 | UI design, 사용자 인터페이스를 정의하고 구현한다. |
  | 채소연 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 권준호 | Architect, 프로젝트의 component를 구성하고 상위 디자인을 책임진다. |

* Project Github : https://github.com/goodsense/project_awesome.git

* 발표자료 : https://github.com/goodsense/project_aewsome/doc/slide.ppt



### Team: 다입

<프로젝트 요약>  
다입은 '다 입게해줄게'라는 뜻으로, 의류 매장에 등록되어 있는 옷들을 바탕으로 옷을 데이터베이스에 넣어 직접 선택해 입혀주거나 고객이 선택하기 힘들다면 말(Speech-to-Text)로 추천을 받아 랜덤으로 그와 비슷한 옷을 입혀주고 배경을 바꾸고 싶다면 배경까지 바꿔서 본인의 의상 착용샷을 보여주는 의류 매장 AI 키오스크

* Members
  | Name | Role |
  |----|----|
  | 감다공 | Project lead, Cloth Manage, 프로젝트를 총괄, 기존의 옷으로 사진의 옷을 바꿔준다. |
  | 이송원 | STT Algorithm, Background Manage, 음성을 추출하고 그 음성을 바탕으로 배경을 만든다. |
  | 조예주 | STT Algorithm, Background Manage, 음성을 추출하고 그 음성을 바탕으로 배경을 만든다. |
  | 송시경 | Gesture, Face Modeling, 얼굴을 인식하고 제스쳐를 인식해 기능을 구현한다. |

* Project Github : https://github.com/bluenight12/da-ipp.git

* 발표자료 : https://github.com/bluenight12/da-ipp/blob/main/present.ppt


