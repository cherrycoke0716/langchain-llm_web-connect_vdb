from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
base_data_path = os.path.join(script_dir, 'crawling_data')

diffuser_file_path = os.path.join(base_data_path, '디퓨저_jsonline.jsonl')
herb_file_path = os.path.join(base_data_path, '허브티_jsonline.jsonl')

def load_and_split_jsonl():
    # TODO: JSONL 파일을 로드하고 분할하는 함수를 구현

    def _metadata_func(record: dict, metadata: dict) -> dict:
        metadata["keyword"] = record.get("matched_keyword", []) #기본값 리스트
        
        return metadata
    
    diffuser_loader = JSONLoader(
        file_path=diffuser_file_path,
        jq_schema='.',
        content_key='text_to_embed',
        json_lines=True,
        metadata_func=_metadata_func # 공통 함수 사용
    )

    herb_loader = JSONLoader(
        file_path=herb_file_path,
        jq_schema='.',
        content_key='text_to_embed',
        json_lines=True,
        metadata_func=_metadata_func # 공통 함수 사용
    )

    diffuser_documents = diffuser_loader.load()
    print(f"디퓨저에서 {len(diffuser_documents)}개 문서 로드 완료")
    herb_documents = herb_loader.load()
    print(f"허브에서 {len(herb_documents)}개 문서 로드 완료")

    all_documents = diffuser_documents + herb_documents
    print(f"총 {len(all_documents)}개 문서 로드 완료.")
    
    # 2. 분할 (Split) - 이 부분은 PDF와 동일
    # text_splitter = RecursiveCharacterTextSplitter(...)
    # split_docs = text_splitter.split_documents(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    split_docs = text_splitter.split_documents(all_documents)
    print(f"총 {len(split_docs)}개의 청크(chunk)로 분할 완료.")
    
    
    
    # (분할 로직 추가 후 split_docs를 반환)
    return split_docs

if __name__ == "__main__":
    # 수정된 함수 호출 (인자 없음)
    docs = load_and_split_jsonl()
    
    if docs:
        print("\n--- 첫 번째 청크(chunk) 내용 ---")
        print(docs[0].page_content)
        print("\n--- 첫 번째 청크(chunk) 메타데이터 ---")
        print(docs[0].metadata)
    else:
        print("로드된 문서가 없습니다.")