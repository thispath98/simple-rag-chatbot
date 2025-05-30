# 🛍️ 네이버 스마트스토어 AI FAQ 챗봇

네이버 스마트스토어 FAQ 데이터를 활용한 RAG 기반 AI 챗봇입니다. 사용자의 질문에 대해 관련 문서를 검색하고, 맥락을 고려한 답변을 제공합니다.

## 주요 기능

- **Retrieval-Augmented Geration (RAG) 기반 답변**: 관련 문서 검색 및 활용을 통한 정확한 답변 생성
- **멀티턴 대화**: 이전 대화 맥락을 고려한 자연스러운 대화 가능
- **라우팅**: 질문의 성격에 따라 적절한 응답 방식 결정
- **쿼리 정제**: 사용자 질문을 FAQ 검색에 최적화된 형태로 변환
- **문서 리랭킹**: 검색된 문서의 관련성 검증 및 순위화
- **맥락 기반 응답**: 검색된 문서와 대화 맥락을 고려한 답변 생성

## 시스템 아키텍처

### 에이전트 구성

1. **Router Agent**
   - `Function calling`을 통한 질문 분석
   - `RAG` 필요 여부 판단, 필요 시 유저의 쿼리 정제
   - 적절한 처리 방식 결정

2. **Reranker Agent**
   - 검색된 문서의 관련성 평가
   - 관련성 점수 기반 문서 순위화
   - 임계값 기반 문서 필터링

3. **Responder Agent**
   - 검증된 문서와 대화 맥락 활용
   - 자연스러운 답변 생성
   - 스트리밍 방식의 응답 제공

### 기술 스택

- **UI**: Streamlit
- **LLM, Embedding**: OpenAI API
- **Vector Store**: Milvus

## 시작하기

### 환경 설정

1. Conda 가상환경 생성 및 활성화:
```bash
conda create -n smartstore_faq python=3.10 -y
conda activate smartstore_faq
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
```bash
cp .env.example .env
```
`.env` 파일에 다음 변수들을 설정:
- `OPENAI_API_KEY`: OpenAI API 키
- `OPENAI_MODEL`: 사용할 OpenAI 모델명
- `OPENAI_EMBEDDING_MODEL`: 임베딩에 사용할 모델명

4. 데이터 파일 설정:
- `final_result.pkl` 파일을 `data` 디렉토리에 위치시켜 주세요.

5. 벡터 스토어 생성:
```bash
python utils/vectorstore_util.py
```
- FAQ 데이터를 임베딩하여 `milvus_demo.db` 벡터 스토어를 생성합니다.

### 실행 방법

```bash
streamlit run main.py
```

## 프로젝트 구조

```
.
├── EDA.ipynb           # FAQ 데이터 분석 및 전처리
├── main.py             # 메인 애플리케이션
├── utils/
│   ├── chat_util.py    # 채팅 관련 유틸리티
│   ├── llms.py         # LLM 에이전트 구현
│   └── vectorstore_util.py  # 벡터 스토어 관련 유틸리티
├── configs/
│   └── prompt_templates.yaml  # 프롬프트 템플릿
└── data/
    └── final_result.pkl  # FAQ 데이터
```