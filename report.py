"""
ChatGPT 보고서 생성기 (Responses API)
- PDF 원본 직접 업로드 + File Search (벡터 스토어 RAG)
- 웹 검색, 코드 인터프리터 지원
- 추론 깊이 제어 (reasoning_effort)
- 세션 지속 (대화 이어가기) 지원
- 환경변수 OPENAI_API_KEY 필요
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import httpx
from openai import OpenAI


# ── 설정 ──────────────────────────────────────────────
DEFAULT_MODEL = "gpt-5.4-pro"
API_TIMEOUT = httpx.Timeout(600.0, connect=30.0)  # 응답 대기 10분, 연결 30초
OUTPUT_DIR = Path(__file__).parent / "wiki" / "reports"
WIKI_DIR = Path(__file__).parent / "wiki"
SESSION_FILE = Path(__file__).parent / ".session"


# ── 위키 페이지 수집 ──────────────────────────────────
def collect_wiki_text(wiki_dir: Path = WIKI_DIR) -> str:
    """wiki/ 디렉토리의 콘텐츠 페이지를 하나의 텍스트로 합침"""
    parts = []
    for md_file in sorted(wiki_dir.rglob("*.md")):
        rel = md_file.relative_to(wiki_dir)
        if rel.name == "log.md" or rel.parts[0] == "reports":
            continue
        content = md_file.read_text(encoding="utf-8")
        label = f"wiki/{rel.as_posix()}"
        parts.append(f"=== {label} ===\n{content}")
        print(f"  [위키] {label} ({len(content):,} chars)", file=sys.stderr)
    return "\n\n".join(parts)


# ── 파일 업로드 + 벡터 스토어 ─────────────────────────
def upload_to_vector_store(client: OpenAI, file_paths: list[Path]) -> str:
    """파일들을 업로드하고 벡터 스토어를 생성하여 ID 반환"""
    # 1. 파일 업로드
    file_ids = []
    for path in file_paths:
        print(f"  [업로드] {path.name}...", file=sys.stderr, end=" ")
        with open(path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="assistants")
        file_ids.append(uploaded.id)
        print(f"완료 ({uploaded.id})", file=sys.stderr)

    # 2. 벡터 스토어 생성
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    vs = client.vector_stores.create(
        name=f"research-{timestamp}",
        file_ids=file_ids,
    )
    print(f"  [벡터스토어] 생성됨 ({vs.id}), 인덱싱 대기 중...", file=sys.stderr, end="")

    # 3. 인덱싱 완료 대기
    while True:
        vs_status = client.vector_stores.retrieve(vs.id)
        if vs_status.status == "completed":
            break
        if vs_status.status == "failed":
            print(" 실패!", file=sys.stderr)
            raise RuntimeError(f"벡터 스토어 인덱싱 실패: {vs_status}")
        print(".", file=sys.stderr, end="", flush=True)
        time.sleep(1)
    print(" 완료!", file=sys.stderr)

    return vs.id


# ── 세션 관리 ─────────────────────────────────────────
def load_session() -> str | None:
    if SESSION_FILE.exists():
        return SESSION_FILE.read_text(encoding="utf-8").strip()
    return None


def save_session(response_id: str):
    SESSION_FILE.write_text(response_id, encoding="utf-8")


def clear_session():
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


# ── Responses API 호출 ────────────────────────────────
def call_chatgpt(
    prompt: str,
    wiki_text: str | None = None,
    vector_store_id: str | None = None,
    model: str | None = None,
    continue_session: bool = False,
    tools: list[str] | None = None,
    reasoning: str = "high",
) -> tuple[str, str]:
    """
    OpenAI Responses API 호출 (스트리밍 + 백그라운드).
    Returns: (output_text, response_id)
    """
    client = OpenAI(timeout=API_TIMEOUT)
    model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    instructions = (
        "당신은 딥러닝 이론 연구 보고서를 작성하는 전문가입니다.\n"
        "수학적 엄밀성을 최우선으로 하고, 주장의 근거가 되는 정리/수식을 반드시 명시하세요.\n"
        "수학 수식은 LaTeX 형식으로 작성하되, 렌더링 오류 방지를 위해 "
        "기호 앞뒤에 반드시 한 칸씩 공백을 두세요. (예: $ x + y = z $)\n"
        "비유적 설명은 철저히 배제하고 이론적 레퍼런스와 수학적 근거만으로 설명하십시오.\n"
        "필요하다면 웹 검색으로 관련 논문/자료를 찾아 근거를 보강하세요.\n"
        "수학적 계산이 필요하면 코드 인터프리터를 활용해 검증하세요."
    )

    # input 구성
    content_parts = []

    if wiki_text:
        content_parts.append({
            "type": "input_text",
            "text": (
                "# 연구 위키 (참고 자료)\n\n"
                "'wiki/' 접두사가 붙은 자료는 연구자가 정리한 위키 페이지입니다.\n\n"
                + wiki_text
            ),
        })

    content_parts.append({
        "type": "input_text",
        "text": f"# 요청\n\n{prompt}",
    })

    input_msg = [{"role": "user", "content": content_parts}]

    # 도구 설정
    tool_list = []
    enabled_tools = tools or ["web_search", "code_interpreter", "file_search"]

    if "web_search" in enabled_tools:
        tool_list.append({"type": "web_search_preview"})
    if "code_interpreter" in enabled_tools:
        tool_list.append({
            "type": "code_interpreter",
            "container": {"type": "auto"}
        })
    if "file_search" in enabled_tools and vector_store_id:
        tool_list.append({
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
        })

    # 세션 이어가기
    kwargs = {}
    if continue_session:
        prev_id = load_session()
        if prev_id:
            kwargs["previous_response_id"] = prev_id
            print(f"  [세션] 이전 대화 이어가기 ({prev_id[:20]}...)", file=sys.stderr)

    # 스트리밍 + 백그라운드 모드
    print(f"  [스트리밍] 요청 전송 중...", file=sys.stderr)
    stream = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_msg,
        tools=tool_list if tool_list else None,
        reasoning={"effort": reasoning, "summary": "auto"},
        background=True,
        stream=True,
        **kwargs,
    )

    # 스트림 이벤트 처리
    output_text = ""
    response_id = None
    text_started = False
    current_tool = None
    char_count = 0  # 텍스트 출력 카운터 (진행률 표시용)

    for event in stream:
        event_type = event.type

        # 응답 생성됨
        if event_type == "response.created":
            response_id = event.response.id
            print(f"  [백그라운드] 요청 전송됨 (id: {response_id})", file=sys.stderr)

        # 진행 중
        elif event_type == "response.in_progress":
            print(f"  [상태] 모델 처리 중...", file=sys.stderr)

        # 추론(reasoning) 요약 델타
        elif event_type == "response.reasoning_summary_text.delta":
            print(f"{event.delta}", file=sys.stderr, end="", flush=True)

        elif event_type == "response.reasoning_summary_text.done":
            print("", file=sys.stderr)  # 줄바꿈

        # 출력 항목 추가됨
        elif event_type == "response.output_item.added":
            item_type = getattr(event.item, "type", None)
            if item_type == "reasoning":
                print("  [추론] 사고 중...", file=sys.stderr, flush=True)
            elif item_type == "web_search_call":
                query = getattr(event.item, "query", "")
                print(f"  [웹검색] {query}", file=sys.stderr, flush=True)
            elif item_type == "file_search_call":
                print(f"  [파일검색] 벡터 스토어 검색 중...", file=sys.stderr, flush=True)
            elif item_type == "code_interpreter_call":
                print(f"  [코드실행] 코드 인터프리터 실행 중...", file=sys.stderr, flush=True)
            elif item_type == "message":
                if not text_started:
                    print("  [응답] 텍스트 생성 시작", file=sys.stderr, flush=True)
                    text_started = True

        # 출력 항목 완료
        elif event_type == "response.output_item.done":
            item_type = getattr(event.item, "type", None)
            if item_type == "reasoning":
                summaries = getattr(event.item, "summary", None) or []
                if summaries:
                    print("  [추론 요약]", file=sys.stderr)
                    for s in summaries:
                        text = s.text if hasattr(s, "text") else str(s)
                        print(f"    {text[:200]}", file=sys.stderr)
            elif item_type == "web_search_call":
                print(f"  [웹검색] 완료", file=sys.stderr, flush=True)
            elif item_type == "file_search_call":
                print(f"  [파일검색] 완료", file=sys.stderr, flush=True)
            elif item_type == "code_interpreter_call":
                print(f"  [코드실행] 완료", file=sys.stderr, flush=True)

        # 텍스트 델타 (실제 응답 내용)
        elif event_type == "response.output_text.delta":
            output_text += event.delta
            char_count += len(event.delta)
            # 매 500자마다 진행률 표시
            if char_count % 500 < len(event.delta):
                print(f"  [출력] {char_count:,}자 생성됨...", file=sys.stderr, flush=True)

        # 텍스트 완료
        elif event_type == "response.output_text.done":
            print(f"  [출력] 텍스트 생성 완료 (총 {len(output_text):,}자)", file=sys.stderr)

        # 응답 완료
        elif event_type == "response.completed":
            resp = event.response
            response_id = resp.id
            print(f"  [완료] 상태: {resp.status}", file=sys.stderr)

            # 토큰 사용량 출력
            if resp.usage:
                inp = resp.usage.input_tokens
                out = resp.usage.output_tokens
                reasoning_tok = getattr(resp.usage, "output_tokens_details", None)
                r_tok = ""
                if reasoning_tok:
                    r_tok = f" (reasoning: {getattr(reasoning_tok, 'reasoning_tokens', '?')})"
                print(f"  [토큰] input={inp:,} output={out:,}{r_tok}", file=sys.stderr)

        # 응답 실패
        elif event_type == "response.failed":
            resp = event.response
            error_info = getattr(resp, "error", "알 수 없는 오류")
            raise RuntimeError(f"요청 실패: {error_info}")

        # 응답 불완전
        elif event_type == "response.incomplete":
            print("  [경고] 응답이 불완전합니다 (incomplete)", file=sys.stderr)

    # 스트림이 중간에 끊긴 경우 → 백그라운드 폴링으로 폴백
    if not output_text and response_id:
        print("  [폴백] 스트림 끊김, 백그라운드 폴링으로 전환...", file=sys.stderr)
        while True:
            response = client.responses.retrieve(response_id)
            if response.status in ("completed", "failed", "incomplete"):
                break
            print(f"  [폴링] {response.status}...", file=sys.stderr, flush=True)
            time.sleep(3)

        if response.status == "failed":
            error_info = getattr(response, "error", "알 수 없는 오류")
            raise RuntimeError(f"요청 실패: {error_info}")

        output_text = response.output_text
        print(f"  [폴백] 텍스트 수거 완료 ({len(output_text):,}자)", file=sys.stderr)

    return output_text, response_id


# ── 보고서 저장 ───────────────────────────────────────
def save_report(prompt: str, response: str, sources_used: list[str], response_id: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"report-{timestamp}.md"
    filepath = OUTPUT_DIR / filename

    frontmatter = (
        "---\n"
        f"title: 보고서 {timestamp}\n"
        f"created: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"sources: {json.dumps(sources_used, ensure_ascii=False)}\n"
        f"response_id: {response_id}\n"
        f"prompt: |\n"
    )
    for line in prompt.split("\n"):
        frontmatter += f"  {line}\n"
    frontmatter += "---\n\n"

    filepath.write_text(frontmatter + response, encoding="utf-8")
    return filepath


# ── CLI ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ChatGPT 보고서 생성기 (Responses API)")
    parser.add_argument("--prompt", "-p", required=True, help="프롬프트 (문자열 또는 파일 경로)")
    parser.add_argument("--sources", "-s", nargs="*", default=[], help="업로드할 파일 경로들 (PDF 등)")
    parser.add_argument("--wiki", "-w", action="store_true", help="wiki/ 페이지를 컨텍스트로 자동 포함")
    parser.add_argument("--model", "-m", default=None, help=f"모델 (기본: {DEFAULT_MODEL})")
    parser.add_argument("--reasoning", "-r", default="high", choices=["low", "medium", "high"],
                        help="추론 깊이 (기본: high)")
    parser.add_argument("--continue", "-c", dest="cont", action="store_true", help="이전 세션 이어가기")
    parser.add_argument("--new-session", action="store_true", help="새 세션 시작")
    parser.add_argument("--no-web", action="store_true", help="웹 검색 비활성화")
    parser.add_argument("--no-code", action="store_true", help="코드 인터프리터 비활성화")
    parser.add_argument("--no-save", action="store_true", help="보고서 저장 안 함")

    args = parser.parse_args()

    if args.new_session:
        clear_session()
        print("[세션] 새 세션 시작", file=sys.stderr)

    # 프롬프트
    prompt_path = Path(args.prompt)
    if prompt_path.exists() and prompt_path.is_file():
        prompt = prompt_path.read_text(encoding="utf-8")
        print(f"[프롬프트] {prompt_path.name} ({len(prompt):,} chars)", file=sys.stderr)
    else:
        prompt = args.prompt

    # 위키 수집
    wiki_text = None
    source_names = []
    if args.wiki:
        print("[위키 수집]", file=sys.stderr)
        wiki_text = collect_wiki_text()
        source_names.append("wiki/*")

    # 파일 업로드 → 벡터 스토어
    client = OpenAI(timeout=API_TIMEOUT)
    vector_store_id = None
    file_paths = [Path(s) for s in args.sources if Path(s).exists()]
    if file_paths:
        print("[파일 업로드 → 벡터 스토어]", file=sys.stderr)
        vector_store_id = upload_to_vector_store(client, file_paths)
        source_names.extend([p.name for p in file_paths])

    # 도구
    tools = []
    if not args.no_web:
        tools.append("web_search")
    if not args.no_code:
        tools.append("code_interpreter")
    if vector_store_id:
        tools.append("file_search")

    # 호출
    print(f"[전송] {args.model or DEFAULT_MODEL} (reasoning: {args.reasoning})", file=sys.stderr)
    output_text, response_id = call_chatgpt(
        prompt=prompt,
        wiki_text=wiki_text,
        vector_store_id=vector_store_id,
        model=args.model,
        continue_session=args.cont,
        tools=tools,
        reasoning=args.reasoning,
    )

    save_session(response_id)
    print(f"[세션] 저장됨 ({response_id[:20]}...)", file=sys.stderr)

    # 출력
    print(output_text)

    # 저장
    if not args.no_save:
        filepath = save_report(prompt, output_text, source_names, response_id)
        print(f"[저장] {filepath}", file=sys.stderr)


if __name__ == "__main__":
    main()
