from langchain_community.document_loaders import JSONLoader

def load_and_split_jsonl(file_path):
    """
    JSONL 파일을 로드하여 청크로 분할합니다.
    """
    
    # 1. 로드
    # (중요) jq_schema='.content' : "content" 필드의 값을 본문(page_content)으로 사용
    # metadata_func : 메타데이터를 어떻게 구성할지 정의
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.content',  # 본문으로 사용할 필드
        json_lines=True,       # JSONL 모드 활성화
        metadata_func=lambda record, metadata: {
            "source": record.get("source_url"),
            "title": record.get("title")
        }
    )
    
    documents = loader.load()
    
    # 2. 분할 (Split) - 이 부분은 PDF와 동일
    # text_splitter = RecursiveCharacterTextSplitter(...)
    # split_docs = text_splitter.split_documents(documents)
    
    print(f"JSONL에서 {len(documents)}개 문서 로드 완료")
    
    # (분할 로직 추가 후 split_docs를 반환)
    return documents

# --- 테스트 ---
if __name__ == "__main__":
    # my_scraped_data.jsonl 파일이 있다고 가정
    docs = load_and_split_jsonl("data/my_scraped_data.jsonl")
    
    print("\n--- 첫 번째 문서 내용 ---")
    print(docs[0].page_content)
    print("\n--- 첫 번째 문서 메타데이터 ---")
    print(docs[0].metadata)