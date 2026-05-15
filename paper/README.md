# KCC 2026 포스터 논문 — 컴파일 가이드

## 파일
- `main.tex` — KCC 양식에 맞춘 LaTeX 소스 (심사용 익명 버전)
- `draft.md` — Markdown 초안 (참고용)
- `outline.md` — 구조 outline

## 컴파일 방법: Overleaf 사용 (권장)

로컬에 LaTeX이 없으므로 **Overleaf** (무료) 사용을 추천합니다.

### 단계
1. https://www.overleaf.com 가입 (무료)
2. **New Project → Blank Project** 생성
3. 기본 `main.tex`를 지우고 본 프로젝트의 `main.tex` 내용을 붙여넣기
4. **Menu → Compiler → XeLaTeX**로 변경 (한글 처리 위해 필수)
5. **Recompile** 버튼 클릭 → PDF 생성

### 컴파일 확인 사항
- ⚠️ **반드시 XeLaTeX**로 설정 (pdfLaTeX은 한글 인식 안 함)
- 컴파일 후 PDF가 **2~3쪽**인지 확인 (KCC 규정)
- 제목/요약은 1단, 본문은 2단인지 확인
- 익명 심사용이므로 저자 정보가 **들어가 있지 않은지** 확인

## KCC 규정 체크리스트

| 항목 | 규정 | 본 논문 | OK |
|---|---|---|---|
| 페이지 | 2~3쪽 | 컴파일 후 확인 | ☐ |
| 용지 | A4 세로 | A4, portrait | ✅ |
| 여백 | 위 30, 아래 20, 좌/우 10mm | geometry 패키지로 설정 | ✅ |
| 제목/요약 | 1단 | `@twocolumnfalse` 사용 | ✅ |
| 본문 | 2단 | `\twocolumn` 사용 | ✅ |
| 폰트 크기 | ≥ 9pt | 10pt | ✅ |
| 그림 캡션 | 하단 "그림 N" | 표만 사용 (그림 없음) | ✅ |
| 표 캡션 | 상단 "표 N" | `\caption` 상단 | ✅ |
| 참고문헌 번호 | 인용 순서 | `\cite` + bibitem | ✅ |
| **저자 정보** | **미기재** | **익명** | ✅ |

## 페이지 분량 조정

**3쪽 초과 시** 줄이는 우선순위:
1. Section 4.4 분석 (b), (c) 축약
2. Related Work에서 VeRA, DoRA 언급 짧게
3. 제안 방법 3.2 Code listing 제거 (이미 없음)
4. 표 3에서 Symmetric, B-only SVD 행 제거

**2쪽 미만이면** 추가:
1. 학습 곡선 그림 추가
2. 추가 분석 paragraph

## 제출

KCC 제출 페이지: https://www.kiise.or.kr/conference/dissertation/dissertationReceipt.do?CC=kcc&CS=2026&PARENT_ID=020500

- Overleaf에서 PDF 다운로드 (오른쪽 상단 download 버튼)
- KCC 시스템에 PDF 업로드
- 파일명: `[이니셜]_FrozenLoRA.pdf` 등 식별 가능한 이름 (단, PDF 내부에는 익명)
